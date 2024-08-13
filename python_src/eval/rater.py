import json
import nltk
import threading
from rouge_score import rouge_scorer
from nltk import word_tokenize
from nltk.translate import bleu, meteor_score
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from core.data_bundle import DataBundle
from core.score import Score
from core.metric_category import MetricCategory
from utils.exceptions import InvalidArgsError
from utils.logger import Trace, log
from utils.strings import StringUtils
from utils.diff import DiffUtils
from utils.dates import DateUtils

class Rater:
    """Collection of popular scoring functions for GenAI tasks."""
    _lock = threading.Lock()
    _is_initialized = False

    @staticmethod
    def _assert_arrays(task_result: DataBundle, ground_truth: DataBundle) -> None:
        if len(task_result) != 1 or not task_result.is_json_array():
            raise InvalidArgsError(f'task_result must have a single JSON_ARRAY')
        if len(ground_truth) != 1 or not ground_truth.is_json_array():
            raise InvalidArgsError(f'ground_truth must have a single JSON_ARRAY')

    @staticmethod
    def _extract_equal_length_arrays(task_result: DataBundle, ground_truth: DataBundle, k: int = 0) -> tuple[list, list]:
        """Ranking metrics expect the two lists to be of equal length."""
        Rater._assert_arrays(task_result, ground_truth)
        pr = task_result.first().value.copy()
        if k:
            pr = pr[:k]
        gt = ground_truth.first().value.copy()
        if k:
            gt = gt[:k]
        if len(pr) < len(gt):
            pr.extend([''] * (len(gt) - len(pr)))
        if len(pr) > len(gt):
            # log(f'Precision calculation uses only the first {len(gt)} elements based on ground truth length.')
            pr = pr[:len(gt)]
        return pr, gt

    @staticmethod
    def precision(task_result: DataBundle, ground_truth: DataBundle, k: int = 5) -> Score:
        """Fraction of relevant documents among the top k search results. Both inputs must include a single JSON_ARRAY."""
        pr, gt = Rater._extract_equal_length_arrays(task_result, ground_truth, k)
        score = float(precision_score(gt, pr, average='micro', zero_division=0))
        log(f'precision@{k}={score}')
        return Score(MetricCategory.ACCURACY, 'precision', score, rater_name='sklearn', rater_version='1.4.0')

    @staticmethod
    def recall(task_result: DataBundle, ground_truth: DataBundle, k: int = 5) -> Score:
        """Measures ability to find all relevant items. Both inputs must include a single JSON_ARRAY."""
        pr, gt = Rater._extract_equal_length_arrays(task_result, ground_truth, k)
        score = float(recall_score(gt, pr, average='micro', zero_division=0))
        log(f'recall@{k}={score}')
        return Score(MetricCategory.ACCURACY, 'recall', score, rater_name='sklearn', rater_version='1.4.0')

    @staticmethod
    def F1(task_result: DataBundle, ground_truth: DataBundle) -> Score:
        """Harmonic mean of precision and recall: 2 * (Precision * Recall) / (Precision + Recall). Both inputs must include a single JSON_ARRAY."""
        pr, gt = Rater._extract_equal_length_arrays(task_result, ground_truth)
        score = float(f1_score(gt, pr, average='micro', zero_division=0))
        log(f'F1={score}')
        return Score(MetricCategory.ACCURACY, 'F1', score, rater_name='sklearn', rater_version='1.4.0')

    @staticmethod
    def search_accuracy(task_result: DataBundle, ground_truth: DataBundle) -> Score:
        """Fraction of expected documents that are among the top k search results. Both inputs must include a single JSON_ARRAY."""
        if ground_truth.is_empty():  # This query is supposed to return empty results.
            score = 1.0 if task_result.is_empty() else 0.0
        else:
            pr, gt = Rater._extract_equal_length_arrays(task_result, ground_truth)
            score = float(accuracy_score(gt, pr, normalize=True))
        log(f'search_accuracy={score}')
        return Score(MetricCategory.ACCURACY, 'search_accuracy', score, rater_name='sklearn', rater_version='1.4.0')

    @staticmethod
    def BLEU(task_result: DataBundle, ground_truth: DataBundle) -> Score:
        prediction, label = Rater._to_string(task_result, ground_truth)
        empty, score = Rater._null_check(prediction, label)
        if not empty:
            score = bleu(
                [prediction.lower()],
                label.lower(),
                smoothing_function=SmoothingFunction().method4,
            )
        return Score(MetricCategory.ACCURACY, 'bleu', score, rater_name='nltk', rater_version='3.8.1_method4')  # type: ignore

    @staticmethod
    def ROUGE_L(task_result: DataBundle, ground_truth: DataBundle, find_text: str | None = None, replace_with: str | None = None) -> Score:
        prediction, label = Rater._to_string(task_result, ground_truth)
        if find_text and replace_with:
            prediction = prediction.replace(find_text, replace_with)
            label = label.replace(find_text, replace_with)
        empty, score = Rater._null_check(prediction, label)
        if not empty:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            score, _, _ = scorer.score(prediction, label)['rougeL']
        return Score(MetricCategory.ACCURACY, 'rouge_l', score, rater_name='rouge-score', rater_version='0.1.2')

    @staticmethod
    def METEOR(task_result: DataBundle, ground_truth: DataBundle, alpha=0.9, beta=3, gamma=0.5) -> Score:
        prediction, label = Rater._to_string(task_result, ground_truth)
        empty, score = Rater._null_check(prediction, label)
        if not empty:
            Rater.initialize()
            score = meteor_score.single_meteor_score(word_tokenize(label), word_tokenize(prediction), alpha=alpha, beta=beta, gamma=gamma)
        return Score(MetricCategory.ACCURACY, 'meteor', score, rater_name='nltk', rater_version='3.8.1')

    @staticmethod
    def exact_match(task_result: DataBundle, ground_truth: DataBundle, ignore_punctuation: bool = False) -> Score:
        """Compares the values, with case insensitive string comparison."""
        if task_result.is_text() and ground_truth.is_text(): # Case insensitive comparison for strings
            prediction, label = Rater._to_string(task_result, ground_truth)
            if ignore_punctuation:
                prediction = StringUtils.remove_punctuation(prediction)
                label = StringUtils.remove_punctuation(label)
            score = 1.0 if prediction.strip().lower() == label.strip().lower() else 0.0
        else:
            score = 1.0 if task_result == ground_truth else 0.0
        return Score(MetricCategory.ACCURACY, 'exact_match', score, rater_version='V1')

    @staticmethod
    def jaccard_dict_similarity(task_result: DataBundle, ground_truth: DataBundle, case_sensitive: bool = True, fuzzy: bool = False, trace: Trace = Trace.OFF) -> Score:
        """Returns Jaccard similarity based on key-value pairs of the two dicts (Shallow method)."""
        if task_result.is_empty():
            return Score(MetricCategory.ACCURACY, 'jaccard_dict_similarity', 1.0 if ground_truth.is_empty() else 0.0)
        if ground_truth.is_empty():
            return Score(MetricCategory.ACCURACY, 'jaccard_dict_similarity', 1.0 if task_result.is_empty() else 0.0)
        if not task_result.is_json_dict():
            raise InvalidArgsError('task_result must be a JSON_DICT bundle')
        if not ground_truth.is_json_dict():
            raise InvalidArgsError('ground_truth must be a JSON_DICT bundle')
        if not task_result.is_single():
            raise InvalidArgsError('task_result must have a single data item')
        if not ground_truth.is_single():
            raise InvalidArgsError('ground_truth must have a single data item')
        dict_pr = task_result.first().value
        dict_gt = ground_truth.first().value
        score = DiffUtils.similarity_dicts(dict_pr, dict_gt, case_sensitive=case_sensitive, fuzzy=fuzzy, trace=trace)
        return Score(MetricCategory.ACCURACY, 'jaccard_dict_similarity', score, rater_version='V1')

    @staticmethod
    def list_of_dict_match(task_result: DataBundle, ground_truth: DataBundle, case_sensitive: bool = True, order_sensitive: bool = True, fuzzy: bool = False, trace: Trace = Trace.OFF) -> Score:
        if task_result.is_empty():
            return Score(MetricCategory.ACCURACY, 'jaccard_list_of_dict_similarity', 1.0 if ground_truth.is_empty() else 0.0)
        if ground_truth.is_empty():
            return Score(MetricCategory.ACCURACY, 'jaccard_list_of_dict_similarity', 1.0 if task_result.is_empty() else 0.0)
        if not task_result.is_json_array():
            raise InvalidArgsError('task_result must be a JSON_ARRAY bundle')
        if not ground_truth.is_json_array():
            raise InvalidArgsError('ground_truth must be a JSON_ARRAY bundle')
        if not task_result.is_single():
            raise InvalidArgsError('task_result must have a single data item')
        if not ground_truth.is_single():
            raise InvalidArgsError('ground_truth must have a single data item')
        list_pr = task_result.first().value
        list_gt = ground_truth.first().value
        score = DiffUtils.lists_of_dicts_match(list_pr, list_gt, case_sensitive=case_sensitive, fuzzy=fuzzy, order_sensitive=order_sensitive, trace=trace)
        return Score(MetricCategory.ACCURACY, 'list_of_dict_match', score, rater_version='V1')

    @staticmethod
    def jaccard_list_of_dict_similarity(id_field_name: str, task_result: DataBundle, ground_truth: DataBundle, case_sensitive: bool = True, fuzzy: bool = False, trace: Trace = Trace.OFF) -> Score:
        """Returns average Jaccard similarity of all list items matched by id_field_name."""
        if not id_field_name:
            raise InvalidArgsError('id_field_name must be set to the attrubute name used to match objects across lists.')
        if task_result.is_empty():
            return Score(MetricCategory.ACCURACY, 'jaccard_list_of_dict_similarity', 1.0 if ground_truth.is_empty() else 0.0)
        if ground_truth.is_empty():
            return Score(MetricCategory.ACCURACY, 'jaccard_list_of_dict_similarity', 1.0 if task_result.is_empty() else 0.0)
        if not task_result.is_json_array():
            raise InvalidArgsError('task_result must be a JSON_ARRAY bundle')
        if not ground_truth.is_json_array():
            raise InvalidArgsError('ground_truth must be a JSON_ARRAY bundle')
        if not task_result.is_single():
            raise InvalidArgsError('task_result must have a single data item')
        if not ground_truth.is_single():
            raise InvalidArgsError('ground_truth must have a single data item')
        list_pr = task_result.first().value
        list_gt = ground_truth.first().value
        score = DiffUtils.similarity_lists_of_dicts(id_field_name, list_pr, list_gt, case_sensitive=case_sensitive, fuzzy=fuzzy, trace=trace)
        return Score(MetricCategory.ACCURACY, 'jaccard_dict', score, rater_version='V1')

    @staticmethod
    def date_match(task_result: DataBundle, ground_truth: DataBundle, adjust_to_next_business_day: bool = False) -> Score:
        if task_result.is_empty():
            return Score(MetricCategory.ACCURACY, 'date_match', 1.0 if ground_truth.is_empty() else 0.0)
        if ground_truth.is_empty():
            return Score(MetricCategory.ACCURACY, 'date_match', 1.0 if task_result.is_empty() else 0.0)
        if not ground_truth.is_single_date():
            raise InvalidArgsError(f'date_match requires a single date as ground truth, received {ground_truth}')
        if not task_result.is_single_date():
            score = 0.0
        elif adjust_to_next_business_day:
            adjusted_result = DateUtils.next_business_date(task_result.to_date())
            adjusted_gt = DateUtils.next_business_date(ground_truth.to_date())
            score = 1.0 if adjusted_result == adjusted_gt else 0.0
        else:
            score = 1.0 if task_result == ground_truth else 0.0
        return Score(MetricCategory.ACCURACY, 'date_match', score, rater_version='V1')

    @staticmethod
    def json_array_match(task_result: DataBundle, ground_truth: DataBundle, ignore_words: list[str] | None = None, penalize_for_elements_not_in_gt: bool = True, trace: int = 0) -> Score:
        if task_result.is_empty():
            return Score(MetricCategory.ACCURACY, 'json_array_match', 1.0 if ground_truth.is_empty() else 0.0)
        if ground_truth.is_empty():
            return Score(MetricCategory.ACCURACY, 'json_array_match', 1.0 if task_result.is_empty() else 0.0)
        if not task_result.is_json_array():
            raise InvalidArgsError('task_result must be a JSON_ARRAY bundle')
        if not ground_truth.is_json_array():
            raise InvalidArgsError('ground_truth must be a JSON_ARRAY bundle')
        if not task_result.is_single():
            raise InvalidArgsError('task_result must have a single data item')
        if not ground_truth.is_single():
            raise InvalidArgsError('ground_truth must have a single data item')
        if ignore_words is None:
            ignore_words = []
        list_pr = task_result.first().value
        if not isinstance(list_pr, list):
            raise InvalidArgsError(f'The first item of task_result must be a list. Actual: {type(list_pr)}')
        list_pr = [StringUtils.remove_words(s.strip().lower(), ignore_words) for s in list_pr]
        list_pr.sort()
        list_gt = ground_truth.first().value
        if not isinstance(list_pr, list):
            raise InvalidArgsError(f'The first item of ground_truth must be a list. Actual: {type(list_pr)}')
        list_gt = [StringUtils.remove_words(s.strip().lower(), ignore_words) for s in list_gt]
        list_gt.sort()
        log(f'list_pr={list_pr}', trace)
        log(f'list_gt={list_gt}', trace)
        gt = set(list_gt)
        num_found = 0
        for value in list_pr:
            is_found = value in gt
            log(f'is_found={is_found}, value={value}', trace)
            if is_found:
                num_found += 1
        if penalize_for_elements_not_in_gt:
            score = Rater._list_score(list_gt, list_pr, num_found)
        else:
            score = num_found / len(list_gt)
        return Score(MetricCategory.ACCURACY, 'json_array_match', score, rater_version='V1')

    @staticmethod
    def json_dict_match(task_result: DataBundle, ground_truth: DataBundle, trace: int = 0) -> Score:
        if task_result.is_empty():
            return Score(MetricCategory.ACCURACY, 'json_dict_match', 1.0 if ground_truth.is_empty() else 0.0)
        if ground_truth.is_empty():
            return Score(MetricCategory.ACCURACY, 'json_dict_match', 1.0 if task_result.is_empty() else 0.0)
        if not task_result.is_json_dict():
            raise InvalidArgsError('task_result must be a JSON_DICT bundle')
        if not ground_truth.is_json_dict():
            raise InvalidArgsError('ground_truth must be a JSON_DICT bundle')
        if not task_result.is_single():
            raise InvalidArgsError('task_result must have a single data item')
        if not ground_truth.is_single():
            raise InvalidArgsError('ground_truth must have a single data item')

        dict_pr = task_result.first().value
        dict_gt = ground_truth.first().value
        keys_pr = [s for s in dict_pr.keys()]
        keys_pr.sort()
        log(f'keys_pr={keys_pr}', trace)
        log(json.dumps(dict_pr, indent=2), trace)
        keys_gt = [s for s in dict_gt.keys()]
        split_keys_gt = (
            {}
        )  # Allow each ground truth key to map to multiple extracted keys: 'Name1/Name2/Name3'
        for key in dict_gt.keys():
            if '/' in key:
                for k in key.split('/'):
                    split_keys_gt[k] = key
        keys_gt.sort()
        log(f'keys_gt={keys_gt}', trace)
        log(json.dumps(dict_gt, indent=2), trace)
        num_matches = 0
        for key in keys_pr:
            if key in keys_gt or key in split_keys_gt:
                val_ex = dict_pr[key]
                val_gt = (
                    dict_gt[key]
                    if key in dict_gt
                    else dict_gt[split_keys_gt[key]]
                )
                temp_val_gt = val_gt
                temp_val_ex = val_ex
                if isinstance(val_gt, int) and isinstance(val_ex, int) and val_ex == val_gt:
                    num_matches += 1
                elif isinstance(val_gt, float) and isinstance(val_ex, float) and val_ex == val_gt:
                    num_matches += 1
                elif isinstance(val_gt, bool) and isinstance(val_ex, bool) and val_ex == val_gt:
                    num_matches += 1
                elif isinstance(val_gt, str) and isinstance(val_ex, str):
                    temp_val_gt = str(val_gt).strip().lower()
                    temp_val_ex = str(val_ex).strip().lower()
                    if DateUtils.is_date(temp_val_gt) and DateUtils.is_date(temp_val_ex):
                        if DateUtils.parse_date(temp_val_gt).date() == DateUtils.parse_date(temp_val_ex).date():
                            num_matches += 1
                    elif temp_val_ex == temp_val_gt:
                        num_matches += 1
                else:
                    log(f'{key} does not match ground truth: {val_ex}|{val_gt}', trace)
        return Score(MetricCategory.ACCURACY, 'json_dict_match', Rater._list_score(keys_gt, keys_pr, num_matches), rater_version='V1')

    @staticmethod
    def _list_score(ground_truth: list, extracted: list, num_matches: int) -> float:
        if not len(ground_truth):
            raise InvalidArgsError('ground_truth must not be empty.')
        num_missing = len(ground_truth) - num_matches
        num_incorrect = len(extracted) - num_matches
        loss = num_missing + num_incorrect
        if loss >= len(ground_truth):
            return 0
        if loss == 0:
            return 1
        score = 1 - (loss / len(ground_truth))
        if score > 1:
            log(f'Houston we have a math problem: score={score}, loss={loss}, num_missing={num_missing}, num_incorrect={num_incorrect}, len(ground_truth)={len(ground_truth)}')
            return 1.0
        return score

    @staticmethod
    def initialize():
        if Rater._is_initialized:
            return
        with Rater._lock:
            nltk.download('wordnet')
            Rater._is_initialized = True

    @staticmethod
    def _null_check(prediction: str, label: str) -> tuple[bool, float]:
        if prediction:
            return False, 0.0
        else:
            return True, 0.0 if label else 1.0  # Score = 1 if label is also empty

    @staticmethod
    def _to_string(prediction: DataBundle, label: DataBundle) -> tuple[str, str]:
        if not prediction.is_empty() and not prediction.is_text():
            raise InvalidArgsError(f'This scoring method requires task result with type DataType.TEXT')
        if not label.is_empty() and not label.is_text():
            raise InvalidArgsError(f'This scoring method requires a ground truth label with type DataType.TEXT')
        return prediction.to_text(), label.to_text()
