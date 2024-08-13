import json
import nltk
import random
import threading
from typing import Optional
from collections.abc import Callable
from storage.base_file_store import BaseFileStore
from storage.local_storage import LocalStorage
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from rouge_score import rouge_scorer
from nltk import word_tokenize
from nltk.translate import bleu, meteor_score
from nltk.translate.bleu_score import SmoothingFunction
from google.cloud import spanner

from core.data_bundle import DataBundle
from core.interaction import Interaction
from core.score import Score
from core.metric_category import MetricCategory
from core.orm import ORM
from core.task import Task
from core.batch_task import BatchTask
from core.conversation import Conversation
from core.batch import Batch

from engine.engine import Engine
from engine.scaler import Scaler
from engine.batch_execution import BatchExecution

from utils.sampling_method import SamplingMethod
from utils.numeric_sequence_stats import NumericSequenceStats
from utils.exceptions import InvalidArgsError, InvalidContext
from utils.strings import StringUtils
from utils.diff import DiffUtils
from utils.dates import DateUtils
from utils.logger import Trace, log
from utils.timer import Timer

from config.defaults import Default

class Dataset():
    """Represents evaluation datasets structured as list[dict]."""
    def __init__(self,
        records: list[dict],
        ground_truth_loader: Callable[[BaseFileStore, str, dict], DataBundle],
        context_loader: Optional[Callable[[BaseFileStore, str, dict], DataBundle]] = None,  # Fetch context data for a single-input task.
        batch_context_loader: Optional[Callable[[BaseFileStore, str, dict, Trace], list[DataBundle]]] = None,  # Fetch context data for a batch task.
        metadata_loader: Optional[Callable[[str, dict], dict]] = None,  # Optional metadata tracked in Spanner like input_location -> URI for the PDF document behind the text chunks.
        result_loader: Optional[Callable[[str, dict], DataBundle]] = None,  # This is not needed for normal evals, but can help score results coming from other sources by skipping task execution.
        storage: BaseFileStore | None = None, # If specified, will be passed to context_loader and ground_truth_loader.
        task_id: str = '',  # Passed to context_loader and ground_truth_loader to help identify the needed data type.
        trace: Trace = Trace.OFF
    ):
        if not ground_truth_loader:
            raise ValueError('ground_truth_loader must be provided.')
        if not context_loader and not batch_context_loader:
            raise ValueError('Either context_loader or batch_context_loader must be provided.')
        if context_loader and batch_context_loader:
            raise ValueError('Only one of context_loader or batch_context_loader can be provided.')
        self.context_loader = context_loader
        self.batch_context_loader = batch_context_loader
        self.ground_truth_loader = ground_truth_loader
        self.metadata_loader = metadata_loader
        self.result_loader = result_loader
        self.records = records
        self.storage = storage or LocalStorage()
        self.task_id = task_id
        self.trace = trace

    def __iter__(self):
        for item in self.records:
            ground_truth = self.ground_truth_loader(self.storage, self.task_id, item)
            metadata = self.metadata_loader(self.task_id, item) if self.metadata_loader else {}
            cached_result = self.result_loader(self.task_id, item) if self.result_loader else None
            if self.context_loader:
                yield self.context_loader(self.storage, self.task_id, item), ground_truth, metadata, cached_result
            elif self.batch_context_loader:
                yield self.batch_context_loader(self.storage, self.task_id, item, self.trace), ground_truth, metadata, cached_result
            else:
                raise InvalidArgsError('Dataset record does not have a context loader.')

    def __len__(self):
        return len(self.records)

    def get_sample( self, sampling_method: SamplingMethod, sample_size: int ) -> 'Dataset':
        """Creates a new Dataset with the requested sample of records."""
        if sampling_method != SamplingMethod.NONE:
            log(f'apply_sampling: sampling_method={sampling_method}, sample_size={sample_size}')
        if sampling_method == SamplingMethod.NONE:
            return self
        if sample_size < 1:
            raise InvalidArgsError(f"sample_size must be >= 1, received {sample_size}")
        if len(self.records) < sample_size:
            log(f'warning: apply_sampling called with sample_size={sample_size} while dataset has only {len(self.records)} rows.')
            sample_size = len(self.records)
        sample = self.records.copy()  # Avoid modifying the original dataset so that one evaluation can run multiple tasks on the same dataset.
        match sampling_method:
            case SamplingMethod.RANDOM:
                log(f'Using a seeded random number generator to help with caching and reproducable evals.')
                random.seed(42)
                random.shuffle( sample )
                sample = sample[:sample_size]
            case SamplingMethod.HEAD:
                sample = sample[:sample_size]
            case SamplingMethod.TAIL:
                sample = sample[-sample_size:]
            case _:
                raise InvalidArgsError(f"Unknown sampling method {sampling_method}")
        return Dataset(
            storage=self.storage,
            task_id=self.task_id,
            records=sample,
            ground_truth_loader=self.ground_truth_loader,
            context_loader=self.context_loader,
            batch_context_loader=self.batch_context_loader,
            metadata_loader=self.metadata_loader,
            result_loader=self.result_loader
        )

@dataclass
class EvalRun():
    """This class maps 1:1 to DB table 'EvalRuns'. All IO is handled by Proton ORM."""
    id: str
    eval_name: str
    eval_version: str
    model_name: str
    model_version: str
    dataset_location: str
    dataset_version: str
    duration: float = 0.0
    num_task_runs: int = 0
    mean_score: float = 0.0
    median_score: float = 0.0
    num_inferences: int = 0
    num_input_tokens: int = 0
    num_output_tokens: int = 0
    num_cached_inferences: int = 0
    namespace: str = 'default'
    insert_date: datetime | None = None

class SamplingMethod(Enum):
    NONE = 0
    RANDOM = 1
    HEAD = 2
    TAIL = 3

@dataclass
class TaskExecution():
    """This class maps 1:1 to DB table 'TaskExecutions'. All IO is handled by Proton ORM"""
    id: str
    eval_name: str
    eval_version: str
    eval_run_id: str
    task_name: str
    task_version: str
    task_run_id: str
    model_name: str
    model_version: str
    dataset_location: str
    dataset_version: str
    duration: float
    num_inferences: int
    mean_inference_latency: float
    median_inference_latency: float
    num_cached_inferences: int
    num_cache_reads: int
    mean_cache_read_latency: float
    median_cache_read_latency: float
    num_cache_writes: int
    mean_cache_write_latency: float
    median_cache_write_latency: float
    mean_prompt_length: float
    mean_prediction_length: float
    input_text: str
    input_location: str
    output_text: str
    output_location: str
    output_context: str
    last_inference_output: str
    ground_truth: str
    main_metric_name: str
    main_metric_score: float
    namespace: str = 'default'
    insert_date: datetime | None = None

@dataclass
class TaskRun():
    """Data class for tracking TaskRuns in Spanner via Proton ORM."""
    id: str
    eval_name: str
    eval_version: str
    eval_run_id: str
    task_name: str
    task_version: str
    model_name: str
    model_version: str
    dataset_location: str
    dataset_version: str
    num_executions: int = 0
    mean_execution_latency: float = 0.0
    median_execution_latency: float = 0.0
    min_execution_latency: float = 0.0
    max_execution_latency: float = 0.0
    num_inferences: int = 0
    mean_inference_latency: float = 0.0
    median_inference_latency: float = 0.0
    num_cached_inferences: int = 0
    num_cache_reads: int = 0
    mean_cache_read_latency: float = 0.0
    median_cache_read_latency: float = 0.0
    num_cache_writes: int = 0
    mean_cache_write_latency: float = 0.0
    median_cache_write_latency: float = 0.0
    mean_prompt_length: float = 0.0
    mean_prediction_length: float = 0.0
    duration: float = 0.0
    mean_score: float = 0.0
    median_score: float = 0.0
    namespace: str = 'default'
    insert_date: datetime | None = None

# This dataclass maps 1:1 to DB table 'TaskScores'. All I/O is handled by Proton ORM.
@dataclass
class TaskScore():
    id: str
    task_execution_id: str
    task_run_id: str
    task_name: str
    task_version: str
    eval_run_id: str
    rater_name: str
    rater_version: str
    metric_category: str
    metric_name: str
    is_main_metric: bool
    score: float
    namespace: str = 'default'
    insert_date: datetime | None = None

@dataclass
class PrimitiveExecution():
    """This class maps 1:1 to DB table 'PrimitiveExecutions'. All I/O is handled by Proton ORM."""
    id: str
    namespace: str
    primitive_type: str
    primitive_id: str
    output_type: str
    eval_name: str
    eval_version: str
    eval_run_id: str
    task_name: str
    task_version: str
    task_run_id: str
    task_execution_id: str
    model_name: str
    model_version: str
    dataset_location: str
    dataset_version: str
    duration: float
    num_inferences: int
    mean_inference_latency: float
    median_inference_latency: float
    num_cached_inferences: int
    num_cache_reads: int
    mean_cache_read_latency: float
    median_cache_read_latency: float
    num_cache_writes: int
    mean_cache_write_latency: float
    median_cache_write_latency: float
    mean_prompt_length: float
    mean_prediction_length: float
    input_text: str
    input_location: str
    output_text: str
    output_location: str
    insert_date: datetime | None = None

# This dataclass maps 1:1 to DB table 'Inferences'. All I/O is handled by Proton ORM.
@dataclass
class Inference():
    id: str
    namespace: str
    eval_run_id: str
    task_run_id: str
    task_execution_id: str
    primitive_execution_id: str
    model_name: str
    model_version: str
    duration: float
    is_cached: bool
    is_multimodal: bool
    input_text: str
    input_location: str
    num_input_tokens: int
    output_text: str
    output_location: str
    num_output_tokens: int
    insert_date: datetime | None = None

class TaskExecutionStats():
    def __init__(self, interactions: list[Interaction], score: float, task_duration_seconds: float):
        self.scores = NumericSequenceStats()
        self.scores.add_value(score)
        self.durations = NumericSequenceStats()
        self.durations.add_value(task_duration_seconds)
        self.num_inferences = NumericSequenceStats()
        self.num_inferences.add_value(len(interactions))
        self.inference_latencies = NumericSequenceStats()
        self.cached_inference_latencies = NumericSequenceStats()
        self.cache_read_latencies = NumericSequenceStats()
        self.cache_write_latencies = NumericSequenceStats()
        self.prompt_lengths = NumericSequenceStats()
        self.output_lengths = NumericSequenceStats()
        for interaction in interactions:
            if interaction.flags.cache_hit:
                self.cached_inference_latencies.add_value(interaction.timers.cache_read_latency.seconds)
            else:
                self.inference_latencies.add_value(interaction.timers.inference_latency.seconds)
            self.cache_read_latencies.add_value(interaction.timers.cache_read_latency.seconds)
            self.cache_write_latencies.add_value(interaction.timers.cache_write_latency.seconds)
            self.prompt_lengths.add_value(len(interaction.prompt.to_text()))
            self.output_lengths.add_value(len(interaction.output.to_text()))

    @staticmethod
    def aggregate(runs: list['TaskExecutionStats']) -> 'TaskExecutionStats':
        agg = TaskExecutionStats([], 0.0, 0.0)
        # Reset the 3 fields that get auto-assigned elements
        agg.scores.values = []
        agg.durations.values = []
        agg.num_inferences.values = []
        for run in runs:
            agg.scores.add_values(run.scores.values)
            agg.durations.add_values(run.durations.values)
            agg.num_inferences.add_values(run.num_inferences.values)
            agg.inference_latencies.add_values(run.inference_latencies.values)
            agg.cached_inference_latencies.add_values(run.cached_inference_latencies.values)
            agg.cache_read_latencies.add_values(run.cache_read_latencies.values)
            agg.cache_write_latencies.add_values(run.cache_write_latencies.values)
            agg.prompt_lengths.add_values(run.prompt_lengths.values)
            agg.output_lengths.add_values(run.output_lengths.values)
        return agg
    
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

class EvalTracker:
    def __init__(self, project_id: str, spanner_instance_id: str, database_id: str, namespace: str = 'DEFAULT', trace: Trace = Trace.OFF):
        dataclasses = [EvalRun, Inference, PrimitiveExecution, TaskRun, TaskExecution, TaskScore]
        self.orm = ORM(project_id, spanner_instance_id, database_id, dataclasses)
        self.namespace = namespace
        self.eval_run: EvalRun
        self.task_run: TaskRun
        self.trace = trace

    def start_eval_run(
        self,
        eval_name: str,
        eval_version: str,
        model_name: str,
        model_version: str,
        dataset_location: str,
        dataset_version: str,
    ):
        params = (eval_name, eval_version, model_name, model_version, dataset_location, dataset_version)
        if None in params:
            raise InvalidArgsError('All arguments are required.')
        if hasattr(self, 'eval_run') and self.eval_run:
            raise InvalidContext('EvaluationRun already started.')
        self.eval_run = EvalRun(
            id=StringUtils.generate_database_id(),
            namespace=self.namespace,
            eval_name=eval_name,
            eval_version=eval_version,
            model_name=model_name,
            model_version=model_version,
            dataset_location=dataset_location,
            dataset_version=dataset_version
        )
        self.orm.write('EvalRuns', self.eval_run)
        log(f'Started EvalRun #{self.eval_run.id} {self.eval_run.eval_name}:{self.eval_run.eval_version}')
        return self.eval_run

    def finalize_eval_run(self, scaler: Scaler, duration: float, num_task_runs: int, mean_score: float, median_score: float, inference_stats: str = ''):
        self.eval_run.duration = duration
        self.eval_run.num_task_runs = num_task_runs
        self.eval_run.mean_score = mean_score
        self.eval_run.median_score = median_score
        self.eval_run.num_inferences = scaler.num_inferences
        self.eval_run.num_cached_inferences = scaler.num_cached_inferences
        self.eval_run.num_input_tokens = scaler.num_input_tokens
        self.eval_run.num_output_tokens = scaler.num_output_tokens
        self.orm.write('EvalRuns', self.eval_run)
        log(f'EvalRun {self.eval_run.id} took {duration:.1f} seconds, score={mean_score*100:.2f}% {inference_stats}')
        return self.eval_run

    def start_task_run(self, task_name: str, task_version: str, dataset_location: str, dataset_version: str):
        if None in (task_name, task_version, dataset_location, dataset_version):
            raise InvalidArgsError('All arguments are required: task_name, task_version')
        if not self.eval_run:
            raise InvalidArgsError('EvalRun must be started before TaskRun')
        self.task_run = TaskRun(
            id=StringUtils.generate_database_id(),
            namespace=self.namespace,
            eval_name=self.eval_run.eval_name,
            eval_version=self.eval_run.eval_version,
            eval_run_id=self.eval_run.id,
            task_name=task_name,
            task_version=task_version,
            model_name=self.eval_run.model_name,
            model_version=self.eval_run.model_version,
            dataset_location=dataset_location,
            dataset_version=dataset_version,
        )
        self.orm.write('TaskRuns', self.task_run)
        log(f'Started TaskRun #{self.task_run.id} for {self.task_run.task_name}:{self.task_run.task_version}')
        return self.task_run

    def finalize_task_run(
        self,
        duration: float,
        num_executions: int,
        mean_execution_latency: float,
        median_execution_latency: float,
        min_execution_latency: float,
        max_execution_latency: float,
        num_inferences: int,
        mean_inference_latency: float,
        median_inference_latency: float,
        num_cached_inferences: int,
        num_cache_reads: int,
        mean_cache_read_latency: float,
        median_cache_read_latency: float,
        num_cache_writes: int,
        mean_cache_write_latency: float,
        median_cache_write_latency: float,
        mean_prompt_length: float,
        mean_prediction_length: float,
        mean_score: float,
        median_score: float,
    ):
        self.task_run.duration = duration
        self.task_run.num_executions = num_executions
        self.task_run.mean_execution_latency = mean_execution_latency
        self.task_run.median_execution_latency = median_execution_latency
        self.task_run.min_execution_latency = min_execution_latency
        self.task_run.max_execution_latency = max_execution_latency
        self.task_run.num_inferences = num_inferences
        self.task_run.mean_inference_latency = mean_inference_latency
        self.task_run.median_inference_latency = median_inference_latency
        self.task_run.num_cached_inferences = num_cached_inferences
        self.task_run.num_cache_reads = num_cache_reads
        self.task_run.mean_cache_read_latency = mean_cache_read_latency
        self.task_run.median_cache_read_latency = median_cache_read_latency
        self.task_run.num_cache_writes = num_cache_writes
        self.task_run.mean_cache_write_latency = mean_cache_write_latency
        self.task_run.median_cache_write_latency = median_cache_write_latency
        self.task_run.mean_prompt_length = mean_prompt_length
        self.task_run.mean_prediction_length = mean_prediction_length
        self.task_run.mean_score = mean_score
        self.task_run.median_score = median_score
        self.orm.write('TaskRuns', self.task_run)
        log(f'TaskRun {self.task_run.id} ({self.task_run.task_name}) took {duration:.1f} seconds, score={mean_score*100:.2f}%')
        return self.task_run

    def create_task_execution(
        self,
        duration: float,
        num_inferences: int,
        mean_inference_latency: float,
        median_inference_latency: float,
        num_cached_inferences: int,
        num_cache_reads: int,
        mean_cache_read_latency: float,
        median_cache_read_latency: float,
        num_cache_writes: int,
        mean_cache_write_latency: float,
        median_cache_write_latency: float,
        mean_prompt_length: float,
        mean_prediction_length: float,
        input_text: str,
        input_location: str,
        output_text: str,
        output_location: str,
        output_context: str,
        last_inference_output: str,
        ground_truth: str,
        main_metric_name: str,
        main_metric_score: float,
    ) -> TaskExecution:
        if not self.eval_run:
            raise InvalidArgsError('EvalRun must be started before TaskExecution')
        if not self.task_run:
            raise InvalidArgsError('TaskRun must be started before TaskExecution')
        execution = TaskExecution(
            id=StringUtils.generate_database_id(),
            eval_name=self.eval_run.eval_name,
            eval_version=self.eval_run.eval_version,
            eval_run_id=self.eval_run.id,
            task_name=self.task_run.task_name,
            task_version=self.task_run.task_version,
            task_run_id=self.task_run.id,
            model_name=self.task_run.model_name,
            model_version=self.task_run.model_version,
            dataset_location=self.task_run.dataset_location,
            dataset_version=self.task_run.dataset_version,
            duration=duration,
            num_inferences=num_inferences,
            mean_inference_latency=mean_inference_latency,
            median_inference_latency=median_inference_latency,
            num_cached_inferences=num_cached_inferences,
            num_cache_reads=num_cache_reads,
            mean_cache_read_latency=mean_cache_read_latency,
            median_cache_read_latency=median_cache_read_latency,
            num_cache_writes=num_cache_writes,
            mean_cache_write_latency=mean_cache_write_latency,
            median_cache_write_latency=median_cache_write_latency,
            mean_prompt_length=mean_prompt_length,
            mean_prediction_length=mean_prediction_length,
            input_text=input_text,
            input_location=input_location,
            output_text=output_text,
            output_location=output_location,
            output_context=output_context,
            last_inference_output = last_inference_output,
            ground_truth=ground_truth,
            main_metric_name=main_metric_name,
            main_metric_score=float(main_metric_score),
            namespace=self.namespace
        )
        self.orm.write('TaskExecutions', execution)
        return execution

    def create_task_score(
        self,
        task_execution_id: str,
        rater_name: str,
        rater_version: str,
        metric_category: str,
        metric_name: str,
        is_main_metric: bool,
        score: float,
    ) -> TaskScore:
        task_score = TaskScore(
            id=StringUtils.generate_database_id(),
            task_execution_id = task_execution_id,
            task_run_id=self.task_run.id,
            task_name=self.task_run.task_name,
            task_version=self.task_run.task_version,
            eval_run_id=self.eval_run.id,
            rater_name=rater_name,
            rater_version=rater_version,
            metric_category=metric_category,
            metric_name=metric_name,
            is_main_metric=is_main_metric,
            score=float(score),
            namespace=self.namespace
        )
        self.orm.write('TaskScores', task_score)
        return task_score

    def create_primitive_execution(
        self,
        task_execution_id: str,
        primitive_type: str,
        primitive_id: str,
        output_type: str,
        duration: float,
        num_inferences: int,
        mean_inference_latency: float,
        median_inference_latency: float,
        num_cached_inferences: int,
        num_cache_reads: int,
        mean_cache_read_latency: float,
        median_cache_read_latency: float,
        num_cache_writes: int,
        mean_cache_write_latency: float,
        median_cache_write_latency: float,
        mean_prompt_length: float,
        mean_prediction_length: float,
        output_text: str,
        output_location: str,
        input_text: str = '',
        input_location: str = ''
    ) -> PrimitiveExecution:
        if not self.eval_run:
            raise InvalidArgsError('EvalRun must be started before PrimitiveExecution')
        if not self.task_run:
            raise InvalidArgsError('TaskRun must be started before PrimitiveExecution')
        execution = PrimitiveExecution(
            id=StringUtils.generate_database_id(),
            namespace=self.namespace,
            primitive_type=primitive_type,
            primitive_id=primitive_id,
            output_type=output_type,
            eval_name=self.eval_run.eval_name,
            eval_version=self.eval_run.eval_version,
            eval_run_id=self.eval_run.id,
            task_name=self.task_run.task_name,
            task_version=self.task_run.task_version,
            task_run_id=self.task_run.id,
            task_execution_id=task_execution_id,
            model_name=self.task_run.model_name,
            model_version=self.task_run.model_version,
            dataset_location=self.task_run.dataset_location,
            dataset_version=self.task_run.dataset_version,
            duration=duration,
            num_inferences=num_inferences,
            mean_inference_latency=mean_inference_latency,
            median_inference_latency=median_inference_latency,
            num_cached_inferences=num_cached_inferences,
            num_cache_reads=num_cache_reads,
            mean_cache_read_latency=mean_cache_read_latency,
            median_cache_read_latency=median_cache_read_latency,
            num_cache_writes=num_cache_writes,
            mean_cache_write_latency=mean_cache_write_latency,
            median_cache_write_latency=median_cache_write_latency,
            mean_prompt_length=mean_prompt_length,
            mean_prediction_length=mean_prediction_length,
            input_text=input_text,
            input_location=input_location,
            output_text=output_text,
            output_location=output_location
        )
        self.orm.write('PrimitiveExecutions', execution)
        return execution

    def create_inference(
        self,
        task_execution_id: str,
        primitive_execution_id: str,
        model_name: str,
        model_version: str,
        duration: float,
        is_cached: bool,
        is_multimodal: bool,
        input_text: str,
        input_location: str,
        num_input_tokens: int,
        output_text: str,
        output_location: str,
        num_output_tokens: int
    ) -> Inference:
        if not self.eval_run:
            raise InvalidArgsError('EvalRun must be started before tracking Inferences')
        if not self.task_run:
            raise InvalidArgsError('TaskRun must be started before tracking Inferences')
        inference = Inference(
            id=StringUtils.generate_database_id(),
            namespace=self.namespace,
            eval_run_id = self.eval_run.id,
            task_run_id = self.task_run.id,
            task_execution_id = task_execution_id,
            primitive_execution_id = primitive_execution_id,
            model_name = model_name,
            model_version = model_version,
            duration = duration,
            is_cached = is_cached,
            is_multimodal = is_multimodal,
            input_text = input_text,
            input_location = input_location,
            num_input_tokens = num_input_tokens,
            output_text = output_text,
            output_location = output_location,
            num_output_tokens = num_output_tokens,
        )
        self.orm.write('Inferences', inference)
        return inference

class TrackingMode(Enum):
    NONE=0
    FULL=1
    RANDOM_SAMPLE_SINGLE=2
    RANDOM_SAMPLE_DOZEN=3

class Evaluator():
    """Orchestrates evaluation runs."""

    def __init__(
        self,
        engine: Engine,
        spanner_project_id: str | None,
        spanner_instance_id: str | None,
        database_id: str | None,
        namespace: str = 'DEFAULT',
        trace: Trace = Trace.OFF
    ):
        if not spanner_project_id:
            raise InvalidArgsError('spanner_project_id is required.')
        if not spanner_instance_id:
            raise InvalidArgsError('spanner_instance_id is required.')
        if not database_id:
            raise InvalidArgsError('database_id is required.')
        self.engine = engine
        self.tracker = EvalTracker(spanner_project_id, spanner_instance_id, database_id, namespace)
        self.trace = trace

    def evaluate(
            self,
            tasks: Task | BatchTask | list[Task] | list[BatchTask],
            datasets: Dataset | list[Dataset],
            eval_name: str = '',
            eval_version: str = '',
            model_name: str = '',
            model_version: str = '',
            dataset_location: str = '',
            dataset_version: str = '',
            sampling_method: SamplingMethod = SamplingMethod.NONE,
            sample_size: int = 0,
            inference_tracking: TrackingMode = TrackingMode.NONE,  # Full tracking increases latency and Spanner costs.
            use_short_docs: bool = False,  # Passed to each task.run
            trace: Trace = Trace.OFF
        ) -> EvalRun:
        """The lists of Tasks and Dataset must match by index. All other parameters are optional."""
        if not tasks:
            raise InvalidArgsError('Tasks list is empty.')
        if not datasets:
            raise InvalidArgsError('Datasets list is empty.')
        self.trace = trace
        self.tracker.trace = trace
        if isinstance(tasks, Task):
            tasks = [tasks]
        elif isinstance(tasks, BatchTask):
            tasks = [tasks]
        if isinstance(datasets, Dataset):
            datasets = [datasets]
        if len(tasks) != len(datasets):
            raise InvalidArgsError('Tasks and Datasets lists must match by index.')
        if not eval_name:
            eval_name = f'{len(tasks)}_tasks' if len(tasks) > 1 else tasks[0].task_id
        if not eval_version:
            eval_version = datetime.strftime(datetime.now(), '%b%d_%H:%M:%S')
        if not model_name:
            model_name, version = self.engine.default_model().parse_model_name_version()
            if not model_version:
                model_version = version
        first_dataset_length = len(datasets[0])
        comment = f'up to {sample_size} inputs' if sample_size else f'{first_dataset_length} inputs'
        log(f'Evaluating {len(tasks)} tasks on {comment} using {self.engine.default_model().model_name}.')
        self.inference_tracking = inference_tracking
        self.trace = trace
        self.tracker.start_eval_run(eval_name, eval_version, model_name, model_version, dataset_location, dataset_version)
        duration = Timer()
        task_run_stats = []
        for task, dataset in zip(tasks, datasets):
            stats = self.track_task_run(task, dataset, dataset_location, dataset_version, sampling_method, sample_size, use_short_docs)
            task_run_stats.extend(stats)
        agg = TaskExecutionStats.aggregate(task_run_stats)
        scaler = self.engine.scaler
        inference_stats = f', {scaler.num_inferences} inferences, {scaler.num_input_tokens + scaler.num_output_tokens} tokens' if self.trace > 0 else ''
        evaluation_duration = duration.seconds
        self.tracker.finalize_eval_run(scaler, evaluation_duration, len(tasks), agg.scores.mean(), agg.scores.median(), inference_stats)
        self.display_eval_summary(agg, eval_name, eval_version, len(tasks), sample_size, first_dataset_length, evaluation_duration, inference_stats, agg.prompt_lengths.mean())
        return self.tracker.eval_run

    def track_task_run(self, task: Task | BatchTask, dataset: Dataset, dataset_location, dataset_version, sampling_method: SamplingMethod = SamplingMethod.NONE, sample_size: int = 0, use_short_docs: bool = False) -> list[TaskExecutionStats]:
        self.engine.flags.shortdocs_enabled = use_short_docs
        dataset = dataset.get_sample(sampling_method, sample_size)
        self.tracker.start_task_run(task.task_id, task.task_version, dataset_location, dataset_version or 'V1.0')
        duration = Timer()
        if isinstance(task, BatchTask):
            task_run_stats = self.run_batch_task_on_dataset(task, dataset)
        else:
            task_run_stats = self.run_single_input_task_on_dataset(task, dataset)
        agg = TaskExecutionStats.aggregate(task_run_stats)
        self.tracker.finalize_task_run(
            duration = duration.seconds,
            num_executions = len(dataset),
            mean_execution_latency = agg.durations.mean(),
            median_execution_latency = agg.durations.median(),
            min_execution_latency = agg.durations.min(),
            max_execution_latency = agg.durations.max(),
            num_inferences = agg.num_inferences.count(),
            mean_inference_latency = agg.inference_latencies.mean(),
            median_inference_latency = agg.inference_latencies.median(),
            num_cached_inferences = agg.cached_inference_latencies.count(),
            num_cache_reads = agg.cache_read_latencies.count(),
            mean_cache_read_latency = agg.cache_read_latencies.mean(),
            median_cache_read_latency = agg.cache_read_latencies.median(),
            num_cache_writes = agg.cache_write_latencies.count(),
            mean_cache_write_latency = agg.cache_write_latencies.mean(),
            median_cache_write_latency = agg.cache_write_latencies.median(),
            mean_prompt_length = agg.prompt_lengths.mean(),
            mean_prediction_length = agg.output_lengths.mean(),
            mean_score = agg.scores.mean(),
            median_score = agg.scores.median()
        )
        return task_run_stats

    def run_single_input_task_on_dataset(self, task: Task, dataset: Dataset) -> list[TaskExecutionStats]:
        if not len(dataset):
            raise InvalidArgsError('Dataset is empty.')
        task_run_stats: list[TaskExecutionStats] = []
        log(f'Evaluating task {task.task_id} on {len(dataset)} ground truth records.')
        for bundle, ground_truth, metadata, cached_result in dataset:
            if isinstance(bundle, list) and len(bundle) == 1 and isinstance(bundle[0], DataBundle):
                bundle = bundle[0]
            if not isinstance(bundle, DataBundle):
                raise InvalidArgsError(f'Evaluation dataset for single input tasks must yield DataBundle (received {type(bundle)})')
            interaction = Interaction(Conversation.from_data_bundle(bundle), data_source=bundle.data_source)
            stats = self.execute_task(task, interaction, ground_truth, bundle.to_text(), metadata, cached_result)
            task_run_stats.append(stats)
        return task_run_stats

    def run_batch_task_on_dataset(self, task: BatchTask, dataset: Dataset) -> list[TaskExecutionStats]:
        if not len(dataset):
            raise InvalidArgsError('Dataset is empty.')
        task_run_stats: list[TaskExecutionStats] = []
        log(f'Evaluating batch task {task.task_id} on {len(dataset)} ground truth records.')
        for bundles, ground_truth, metadata, cached_result in dataset:
            if not isinstance(bundles, list):
                raise InvalidArgsError('Evaluation dataset for batch tasks must yield a list[DataBundle]')
            batch = Batch.from_data_bundles(bundles)
            context_for_tracing = batch.combine_interaction_contexts(defrag_text=True).to_text()
            stats = self.execute_task(task, batch, ground_truth, context_for_tracing, metadata, cached_result)
            task_run_stats.append(stats)
        return task_run_stats

    def execute_task(self, task: BatchTask | Task, context: Batch | Interaction, ground_truth: DataBundle, input_as_text: str, metadata: dict, cached_result: DataBundle | None) -> TaskExecutionStats:
        timer = Timer()
        data_source = f'({context.data_source.name})' if context.data_source and context.data_source.name else ''
        log(f'{task.task_id}{data_source}')
        if self.inference_tracking != TrackingMode.NONE:
            self.engine.enable_token_stats()  # Disable by default because of latency overhead in Bison
        self.engine.start_tracking()
        self.engine.scaler.start_inference_tracking()
        if cached_result:
            log(f'evaluator is skipping task execution for {task.task_id} because this dataset provided a cached result.')
            result = cached_result
        else:
            result = task.run(self.engine, context) # type: ignore
        timer.stop()
        interactions = self.engine.scaler.stop_inference_tracking()
        primitive_executions = self.engine.stop_tracking()
        scoring_result = task.rate(result, ground_truth)  # The actual scoring happens here
        main_score = scoring_result if isinstance(scoring_result, Score) else scoring_result[0]
        all_scores = [main_score] if isinstance(scoring_result, Score) else scoring_result
        if len(str(result)) > 160 and self.trace != Trace.ON:
            log(f'SCORE={main_score.score:.1f}\nRESULT={StringUtils.truncate(str(result), 160)}\nLABEL= {StringUtils.truncate(str(ground_truth), 160)}')
        elif len(str(result)) > 60:
            log(f'SCORE={main_score.score:.1f}\nresult={result}\nlabel= {ground_truth}')
        else:
            log(f'SCORE={main_score.score:.1f} result={result} label={ground_truth}')
        ts = TaskExecutionStats(interactions, main_score.score, timer.seconds)
        task_execution = self.track_task_execution(ground_truth, input_as_text, metadata, timer.seconds, result, interactions, primitive_executions, main_score, ts)
        for score in all_scores:
            self.tracker.create_task_score(
                task_execution_id = task_execution.id,
                rater_name = score.rater_name,
                rater_version = score.rater_version,
                metric_category = str(score.metric_category),
                metric_name = score.metric_name,
                is_main_metric = True,
                score = score.score
            )
        return ts

    def track_task_execution(self, ground_truth: DataBundle, input_as_text: str, metadata: dict, duration_seconds: float, task_result: DataBundle, interactions: list[Interaction], primitive_executions: list[BatchExecution], score: Score, stats: TaskExecutionStats) -> TaskExecution:
        task_execution = self.tracker.create_task_execution(
            duration = duration_seconds,
            num_inferences = len(interactions),
            mean_inference_latency = stats.inference_latencies.mean(),
            median_inference_latency = stats.inference_latencies.median(),
            num_cached_inferences = stats.cached_inference_latencies.count(),
            num_cache_reads = stats.cache_read_latencies.count(),
            mean_cache_read_latency = stats.cache_read_latencies.mean(),
            median_cache_read_latency = stats.cache_read_latencies.median(),
            num_cache_writes = stats.cache_write_latencies.count(),
            mean_cache_write_latency = stats.cache_write_latencies.mean(),
            median_cache_write_latency = stats.cache_write_latencies.median(),
            mean_prompt_length = stats.prompt_lengths.mean(),
            mean_prediction_length = stats.output_lengths.mean(),
            input_text = input_as_text,
            input_location = metadata['input_location'] if 'input_location' in metadata else '',
            output_text = task_result.to_text(),
            output_location = metadata['output_location'] if 'output_location' in metadata else '',
            output_context = interactions[-1].prompt.to_text() if interactions else '', # Last executed prompt for investigations and SFT
            last_inference_output = interactions[-1].output.to_text() if interactions else '',
            ground_truth = ground_truth.to_text(),
            main_metric_name = score.metric_name,
            main_metric_score = score.score
        )
        self.track_primitive_executions(metadata, primitive_executions, task_execution)
        return task_execution


    def track_primitive_executions(self, metadata: dict, primitive_executions: list[BatchExecution], task_execution: TaskExecution):
        for p in primitive_executions:
            stats = TaskExecutionStats(p.batch.interactions, 0, 0)
            primitive_execution = self.tracker.create_primitive_execution(
                task_execution_id=task_execution.id,
                primitive_type=p.primitive.__class__.__name__,
                primitive_id=p.primitive.id,
                output_type=p.primitive.output_type.name,
                duration=p.duration_seconds or 0.0,
                num_inferences=len(p.batch),
                mean_inference_latency = stats.inference_latencies.mean(),
                median_inference_latency = stats.inference_latencies.median(),
                num_cached_inferences = stats.cached_inference_latencies.count(),
                num_cache_reads = stats.cache_read_latencies.count(),
                mean_cache_read_latency = stats.cache_read_latencies.mean(),
                median_cache_read_latency = stats.cache_read_latencies.median(),
                num_cache_writes = stats.cache_write_latencies.count(),
                mean_cache_write_latency = stats.cache_write_latencies.mean(),
                median_cache_write_latency = stats.cache_write_latencies.median(),
                mean_prompt_length = stats.prompt_lengths.mean(),
                mean_prediction_length = stats.output_lengths.mean(),
                output_text = str(p.batch.non_empty()),
                output_location = ''
            )
            if self.inference_tracking != TrackingMode.NONE:
                interactions = p.batch.interactions
                if self.inference_tracking == TrackingMode.RANDOM_SAMPLE_SINGLE:
                    interactions = [interactions[random.randint(0, len(interactions) - 1)]]
                elif self.inference_tracking == TrackingMode.RANDOM_SAMPLE_DOZEN:
                    interactions = random.sample(interactions, min(12, len(interactions)))
                else:
                    log(f'Evaluator is saving {len(interactions)} inferences to Spanner')
                self.track_inferences(metadata, interactions, task_execution, primitive_execution)

    def track_inferences(self, metadata: dict, interactions: list[Interaction], task_execution: TaskExecution, primitive_execution: PrimitiveExecution):
        for i in interactions:
            model_name, model_version = i.selected_model.parse_model_name_version() if i.selected_model else ('', '')
            self.tracker.create_inference(
                task_execution_id=task_execution.id,
                primitive_execution_id=primitive_execution.id,
                model_name=model_name,
                model_version=model_version,
                duration = i.timers.inference_latency.seconds,
                is_cached = i.flags.cache_hit,
                is_multimodal = not i.is_output_empty() and not i.output.is_text(),
                input_text = i.prompt.to_text(),
                input_location = metadata['input_location'] if 'input_location' in metadata else '',
                num_input_tokens = i.inference_metadata.num_input_tokens,
                output_text = i.raw_model_output_text or i.output.to_text(),
                output_location = metadata['output_location'] if 'output_location' in metadata else '',
                num_output_tokens = i.inference_metadata.num_output_tokens
            )

    def display_eval_summary(self, agg: TaskExecutionStats, eval_name: str, eval_version: str, num_of_tasks: int, sample_size: int, dataset_length: int, duration: float, inference_stats: str = '', mean_prompt_length_chars: float = 0.0):
        model = self.engine.default_model()
        log('='*50)
        log('Summary:')
        log(f' - Eval name: {eval_name}')
        log(f' - Eval version: {eval_version}')
        log(f' - Model name: {model.model_name}')
        log(f' - {model.model_config}')
        log(f' - Number of tasks: {num_of_tasks}')
        log(f' - Number of inputs: {sample_size if sample_size else dataset_length} (avg={mean_prompt_length_chars} chars)')
        log(f' - Number of inferences: {int(agg.num_inferences.sum())}')
        if self.trace > Trace.OFF:
            log(f' - Inference Stats: {inference_stats}')
        log(f' - Mean score: {agg.scores.mean()*100:.2f}%')
        log(f' - Median score: {agg.scores.median()*100:.2f}%')
        log(f' - Duration: {duration:.1f} seconds')
        log('='*50)

class EvalSchema():
    def __init__(self, project_id: str, spanner_instance_id: str, spanner_database_id: str):
        self.project_id = project_id
        self.spanner_instance_id = spanner_instance_id
        self.spanner_database_id = spanner_database_id
        self.client = spanner.Client(project_id)
        self.instance = self.client.instance(spanner_instance_id)
        self.database = self.instance.database(spanner_database_id)

    def create_database(self):
        """Creates a new Spanner database for Evaluations."""
        self.database = self.instance.database(self.spanner_database_id)
        operation = self.database.create()
        log(f'Adding database {self.spanner_database_id} to Spanner instance {self.spanner_instance_id} in project {self.project_id}')
        operation.result(120)
        log('Evaluation DB is ready.')

    def create_table_eval_runs(self):
        log('Creating table EvalRuns...')
        operation = self.database.update_ddl(["""
        CREATE TABLE EvalRuns (
            id STRING(MAX) NOT NULL,
            namespace STRING(MAX) NOT NULL,
            insert_date TIMESTAMP NOT NULL OPTIONS(allow_commit_timestamp=true),
            eval_name STRING(MAX) NOT NULL,
            eval_version STRING(MAX),
            model_name STRING(MAX),
            model_version STRING(MAX),
            dataset_location STRING(MAX),
            dataset_version STRING(MAX),
            duration FLOAT64,
            num_task_runs INT64,
            mean_score FLOAT64,
            median_score FLOAT64,
            num_inferences INT64,
            num_input_tokens INT64,
            num_output_tokens INT64,
            num_cached_inferences INT64
        )
        PRIMARY KEY (id)
        """])
        operation.result(Default.SPANNER_TIMEOUT)
        log('EvalRuns table created.')

    def create_table_task_runs(self):
        log('Creating table TaskRuns...')
        operation = self.database.update_ddl(["""
        CREATE TABLE TaskRuns (
            id STRING(MAX) NOT NULL,
            namespace STRING(MAX) NOT NULL,
            insert_date TIMESTAMP NOT NULL OPTIONS(allow_commit_timestamp=true),
            eval_name STRING(MAX) NOT NULL,
            eval_version STRING(MAX),
            eval_run_id STRING(MAX) NOT NULL,
            task_name STRING(MAX) NOT NULL,
            task_version STRING(MAX),
            model_name STRING(MAX),
            model_version STRING(MAX),
            dataset_location STRING(MAX),
            dataset_version STRING(MAX),
            duration FLOAT64,
            num_executions INT64,
            mean_execution_latency FLOAT64,
            median_execution_latency FLOAT64,
            min_execution_latency FLOAT64,
            max_execution_latency FLOAT64,
            num_inferences INT64,
            mean_inference_latency FLOAT64,
            median_inference_latency FLOAT64,
            num_cached_inferences INT64,
            num_cache_reads INT64,
            mean_cache_read_latency FLOAT64,
            median_cache_read_latency FLOAT64,
            num_cache_writes INT64,
            mean_cache_write_latency FLOAT64,
            median_cache_write_latency FLOAT64,
            mean_prompt_length FLOAT64,
            mean_prediction_length FLOAT64,
            mean_score FLOAT64,
            median_score FLOAT64
        )
        PRIMARY KEY (id)
        """])
        operation.result(Default.SPANNER_TIMEOUT)
        log('TaskRuns table created.')

    def create_table_task_executions(self):
        log('Creating table TaskExecutions...')
        operation = self.database.update_ddl(["""
        CREATE TABLE TaskExecutions(
            id STRING(MAX) NOT NULL,
            namespace STRING(MAX) NOT NULL,
            insert_date TIMESTAMP NOT NULL OPTIONS(allow_commit_timestamp=true),
            eval_name STRING(MAX) NOT NULL,
            eval_version STRING(MAX),
            eval_run_id STRING(MAX) NOT NULL,
            task_name STRING(MAX) NOT NULL,
            task_version STRING(MAX),
            task_run_id STRING(MAX) NOT NULL,
            model_name STRING(MAX),
            model_version STRING(MAX),
            dataset_location STRING(MAX),
            dataset_version STRING(MAX),
            duration FLOAT64,
            num_inferences INT64,
            mean_inference_latency FLOAT64,
            median_inference_latency FLOAT64,
            num_cached_inferences INT64,
            num_cache_reads INT64,
            mean_cache_read_latency FLOAT64,
            median_cache_read_latency FLOAT64,
            num_cache_writes INT64,
            mean_cache_write_latency FLOAT64,
            median_cache_write_latency FLOAT64,
            mean_prompt_length FLOAT64,
            mean_prediction_length FLOAT64,
            input_text STRING(MAX),
            input_location STRING(MAX),
            output_text STRING(MAX),
            output_location STRING(MAX),
            output_context STRING(MAX),
            last_inference_output STRING(MAX),
            ground_truth STRING(MAX),
            main_metric_name STRING(MAX),
            main_metric_score FLOAT64
        )
        PRIMARY KEY (id)
        """])
        operation.result(Default.SPANNER_TIMEOUT)
        log('TaskExecutions table created.')

    def create_table_primitive_executions(self):
        log('Creating table PrimitiveExecutions...')
        operation = self.database.update_ddl(["""
        CREATE TABLE PrimitiveExecutions(
            id STRING(MAX) NOT NULL,
            namespace STRING(MAX) NOT NULL,
            insert_date TIMESTAMP NOT NULL OPTIONS(allow_commit_timestamp=true),
            primitive_type STRING(MAX) NOT NULL,
            primitive_id STRING(MAX) NOT NULL,
            output_type STRING(MAX) NOT NULL,
            eval_name STRING(MAX) NOT NULL,
            eval_version STRING(MAX),
            eval_run_id STRING(MAX) NOT NULL,
            task_name STRING(MAX) NOT NULL,
            task_version STRING(MAX),
            task_run_id STRING(MAX) NOT NULL,
            task_execution_id STRING(MAX) NOT NULL,
            model_name STRING(MAX),
            model_version STRING(MAX),
            dataset_location STRING(MAX),
            dataset_version STRING(MAX),
            duration FLOAT64,
            num_inferences INT64,
            mean_inference_latency FLOAT64,
            median_inference_latency FLOAT64,
            num_cached_inferences INT64,
            num_cache_reads INT64,
            mean_cache_read_latency FLOAT64,
            median_cache_read_latency FLOAT64,
            num_cache_writes INT64,
            mean_cache_write_latency FLOAT64,
            median_cache_write_latency FLOAT64,
            mean_prompt_length FLOAT64,
            mean_prediction_length FLOAT64,
            input_text STRING(MAX),
            input_location STRING(MAX),
            output_text STRING(MAX),
            output_location STRING(MAX),
            output_context STRING(MAX),
            ground_truth STRING(MAX),
            main_metric_name STRING(MAX),
            main_metric_score FLOAT64
        )
        PRIMARY KEY (id)
        """])
        operation.result(Default.SPANNER_TIMEOUT)
        log('PrimitiveExecutions table created.')

    def create_table_inferences(self):
        log('Creating table Inferences...')
        operation = self.database.update_ddl(["""
        CREATE TABLE Inferences(
            id STRING(MAX) NOT NULL,
            namespace STRING(MAX) NOT NULL,
            insert_date TIMESTAMP NOT NULL OPTIONS(allow_commit_timestamp=true),
            eval_run_id STRING(MAX) NOT NULL,
            task_run_id STRING(MAX) NOT NULL,
            task_execution_id STRING(MAX) NOT NULL,
            primitive_execution_id STRING(MAX) NOT NULL,
            model_name STRING(MAX) NOT NULL,
            model_version STRING(MAX),
            duration FLOAT64,
            is_cached BOOL NOT NULL,
            is_multimodal BOOL NOT NULL,
            input_text STRING(MAX),
            input_location STRING(MAX),
            num_input_tokens INT64,
            output_text STRING(MAX),
            output_location STRING(MAX),
            num_output_tokens INT64
        )
        PRIMARY KEY (id)
        """])
        operation.result(Default.SPANNER_TIMEOUT)
        log('Inferences table created.')

    def create_table_scores(self):
        log('Creating table Scores...')
        operation = self.database.update_ddl(["""
        CREATE TABLE TaskScores(
            id STRING(MAX) NOT NULL,
            namespace STRING(MAX) NOT NULL,
            insert_date TIMESTAMP NOT NULL OPTIONS(allow_commit_timestamp=true),
            task_execution_id STRING(MAX) NOT NULL,
            task_run_id STRING(MAX) NOT NULL,
            task_name STRING(MAX) NOT NULL,
            task_version STRING(MAX),
            eval_run_id STRING(MAX) NOT NULL,
            rater_name STRING(MAX) NOT NULL,
            rater_version STRING(MAX),
            metric_category STRING(MAX),
            metric_name STRING(MAX) NOT NULL,
            is_main_metric BOOL NOT NULL,
            score FLOAT64 NOT NULL
        )
        PRIMARY KEY (id)
        """])
        operation.result(Default.SPANNER_TIMEOUT)
        log('Scores table created.')

    def create_indexes(self):
        log('Creating Spanner indexes...')
        operation = self.database.update_ddl([
            'CREATE UNIQUE INDEX EvalRuns_NamespaceNameVersion ON EvalRuns (namespace, eval_name, eval_version)',
            'CREATE INDEX TaskRuns_EvalRunId ON TaskRuns (eval_run_id)',
            'CREATE INDEX TaskExecutions_EvalRunId ON TaskExecutions (eval_run_id)',
            'CREATE INDEX TaskExecutions_TaskRunId ON TaskExecutions (task_run_id)',
            'CREATE INDEX PrimitiveExecutions_TaskExecutionId ON PrimitiveExecutions (task_execution_id)',
            'CREATE INDEX Inferences_PrimitiveExecutionId ON Inferences (primitive_execution_id)',
            'CREATE INDEX TaskScores_TaskExecutionId ON TaskScores (task_execution_id)'
        ])
        operation.result(Default.SPANNER_TIMEOUT)
        log('Indexes are ready.')

    def create_schema(self):
        self.create_table_eval_runs()
        self.create_table_task_runs()
        self.create_table_task_executions()
        self.create_table_primitive_executions()
        self.create_table_inferences()
        self.create_table_scores()
        self.create_indexes()
        log('Successfully created all tables and indexes.')

    def drop_schema(self):
        log('Deleting all Spanner indexes and tables for Evaluations')
        operation = self.database.update_ddl([
            # 'DROP INDEX EvalRuns_NamespaceNameVersion',
            # 'DROP INDEX TaskExecutions_EvalRunId',
            # 'DROP INDEX TaskExecutions_TaskRunId',
            # 'DROP INDEX TaskRuns_EvalRunId',
            # 'DROP INDEX TaskScores_TaskExecutionId',
            # 'DROP INDEX PrimitiveExecutions_TaskExecutionId',
            # 'DROP INDEX Inferences_PrimitiveExecutionId',
            # 'DROP TABLE EvalRuns',
            # 'DROP TABLE TaskRuns',
            # 'DROP TABLE TaskExecutions',
            # 'DROP TABLE PrimitiveExecutions',
            # 'DROP TABLE Inferences',
            # 'DROP TABLE TaskScores'
        ])
        operation.result(Default.SPANNER_TIMEOUT)
        log('Deleted all tables and indexes.')

    def add_columns(self):
        log('Adding new columns...')
        operation = self.database.update_ddl([
            'ALTER TABLE TaskExecutions ADD COLUMN last_inference_output STRING(MAX)',
            # 'ALTER TABLE EvalRuns ADD COLUMN num_inferences INT64',
            # 'ALTER TABLE EvalRuns ADD COLUMN num_input_tokens INT64',
            # 'ALTER TABLE EvalRuns ADD COLUMN num_output_tokens INT64',
            # 'ALTER TABLE EvalRuns ADD COLUMN num_cached_inferences INT64'
        ])
        operation.result(Default.SPANNER_TIMEOUT)
        log('Finished adding columns.')

class EvalResults():
    """Read only API for fetching evaluation results from Spanner."""

    def __init__(self, project_id: str, spanner_instance_id: str, spanner_database_id: str):
        dataclasses = [EvalRun, TaskRun, TaskExecution, TaskScore, PrimitiveExecution, Inference]
        self.orm = ORM(project_id, spanner_instance_id, spanner_database_id, dataclasses)
        spanner_instance = spanner.Client(project_id).instance(spanner_instance_id)
        self.database = spanner_instance.database(spanner_database_id)

    def load_task_executions(self, task_run_id: str | None = None, eval_run_id: str | None = None) -> list[TaskExecution]:
        if not task_run_id and not eval_run_id:
            raise InvalidArgsError('Either task_run_id or task_run_ids must be specified')
        if task_run_id and eval_run_id:
            raise InvalidArgsError('Only one of task_run_id or task_run_ids can be specified')
        with self.database.snapshot() as snapshot:
            if task_run_id:
                sql = 'SELECT id FROM TaskExecutions WHERE task_run_id = @param'
            else:
                sql = 'SELECT id FROM TaskExecutions WHERE eval_run_id = @param'
            log(f'Loading task executions...')
            cursor = snapshot.execute_sql(
                sql,
                params={'param': task_run_id or eval_run_id},
                param_types={'param': spanner.param_types.STRING}
            )
            results = []
            for row in cursor:
                results.append(self.orm.read('TaskExecutions', row[0]))
            log(f'Loaded {len(results)} task executions')
            return results

    def load_inferences(self, task_execution_id: str) -> list[Inference]:
        if not task_execution_id:
            raise InvalidArgsError('task_execution_id must be specified')
        with self.database.snapshot() as snapshot:
            cursor = snapshot.execute_sql(
                'SELECT id FROM Inferences WHERE task_execution_id = @param',
                params={'param': task_execution_id},
                param_types={'param': spanner.param_types.STRING}
            )
            results = []
            for row in cursor:
                results.append(self.orm.read('Inferences', row[0]))
            log(f'Loaded {len(results)} inferences for task execution {task_execution_id}')
            return results
