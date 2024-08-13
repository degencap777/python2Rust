import statistics
from typing import Any
from datetime import datetime
from utils.dates import DateUtils
from utils.logger import Trace, log
from utils.exceptions import InvalidArgsError

class DiffUtils():

    @staticmethod
    def _variable_to_str(variable: Any, case_sensitive: bool = True, fuzzy: bool = False) -> str:
        if isinstance(variable, float):
            variable = f'{round(variable)}' if variable.is_integer() else f'{variable:.3f}'
        if isinstance(variable, str):
            if not case_sensitive:
                variable = variable.lower()
            if fuzzy:
                variable = variable.strip()
                if DateUtils.is_date( variable ):  # Normalize date format
                    variable = DateUtils.format_date(DateUtils.parse_date(variable))
        return str(variable)

    @staticmethod
    def _is_array_of_basic_types(array: list) -> bool:
        if not isinstance(array, list):
            return False
        for item in array:
            if not isinstance(item, (int, float, str, datetime, bool, complex)):
                return False
        return True

    @staticmethod
    def _array_to_str(array: list, case_sensitive: bool = True, fuzzy: bool = False) -> str:
        if not DiffUtils._is_array_of_basic_types(array):
            raise InvalidArgsError(f'array must be a lit of basic Python data types: {array}')
        strings = [DiffUtils._variable_to_str(x, case_sensitive, fuzzy) for x in array]
        csv = ','.join(strings)
        return f'[{csv}]'

    @staticmethod
    def _dict_to_sorted_kvset_shallow(kv_dict: dict, case_sensitive: bool = True, fuzzy: bool = False) -> set[str]:
        if not isinstance(kv_dict, dict):
            raise InvalidArgsError(f'kv_dict must be a dict, not {type(kv_dict)}')
        kv_list = []
        for key, value in kv_dict.items():
            key = DiffUtils._variable_to_str(key, case_sensitive, fuzzy)
            if isinstance(value, list):
                value = DiffUtils._array_to_str(value, case_sensitive, fuzzy)
            else:
                value = DiffUtils._variable_to_str(value, case_sensitive, fuzzy)
            kv_list.append(f'{key}={value}')
        return set(sorted(kv_list))

    @staticmethod
    def _dict_to_sorted_kv_lines_shallow(kv_dict: dict, case_sensitive: bool = True, fuzzy: bool = False) -> str:
        kvs = list(DiffUtils._dict_to_sorted_kvset_shallow(kv_dict, case_sensitive, fuzzy))
        return "\n".join(sorted(kvs))

    @staticmethod
    def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
        """Computes Jaccard similarity between two sets of strings"""
        if not isinstance(set_a, set):
            raise InvalidArgsError(f'set_a must be a set of strings, not {type(set_a)}')
        if not isinstance(set_b, set):
            raise InvalidArgsError(f'set_b must be a set of strings, not {type(set_b)}')
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union

    @staticmethod
    def similarity_dicts(obj_a: dict, obj_b: dict, case_sensitive: bool = True, fuzzy: bool = False, trace: Trace = Trace.OFF) -> float:
        """Returns Jaccard similarity based on key-value pairs of the two dicts (Shallow method)."""
        set_a = DiffUtils._dict_to_sorted_kvset_shallow(obj_a, case_sensitive, fuzzy)
        set_b = DiffUtils._dict_to_sorted_kvset_shallow(obj_b, case_sensitive, fuzzy)
        jaccard_similarity = DiffUtils.jaccard_similarity(set_a, set_b)
        if jaccard_similarity < 1 and trace > 0:
            log(f'Jaccard similarity={jaccard_similarity:.2f}:')
            log(set_a)
            log(set_b)
        return jaccard_similarity

    @staticmethod
    def similarity_lists_of_dicts(id_field_name: str, list_a: list[dict], list_b: list[dict], case_sensitive: bool = True, fuzzy: bool = False, trace: Trace = Trace.OFF) -> float:
        """Returns average Jaccard similarity of all list items matched by id_field_name."""
        if not id_field_name:
            raise InvalidArgsError(f'id_field_name is required and must point to a dict key that exists in every list item')
        if not isinstance(list_a, list):
            raise InvalidArgsError(f'list_a must be a list of dicts, not {type(list_a)}')
        if not isinstance(list_b, list):
            raise InvalidArgsError(f'list_b must be a list of dicts, not {type(list_b)}')
        for item in list_a:
            if not isinstance(item, dict):
                raise InvalidArgsError(f'list_a includes an item that is not a dict: {type(item)}')
            if not id_field_name in item or not item[id_field_name]:
                raise InvalidArgsError(f'list_a includes an item without ID attribute "{id_field_name}": {item}')
        a_by_id = {obj[id_field_name]: obj for obj in list_a}
        for item in list_b:
            if not isinstance(item, dict):
                raise InvalidArgsError(f'list_b includes an item that is not a dict: {type(item)}')
            if not id_field_name in item or not item[id_field_name]:
                raise InvalidArgsError(f'list_b includes an item without ID attribute "{id_field_name}": {item}')
        b_by_id = {obj[id_field_name]: obj for obj in list_b}
        id_similarity = DiffUtils.jaccard_similarity(set(a_by_id.keys()), set(b_by_id.keys()))
        if id_similarity < 1 and trace > 0:
            log(f'Detected ID mismatch between lists:')
            log(set(sorted(a_by_id.keys())))
            log(set(sorted(b_by_id.keys())))
        obj_similarities = []
        for key in a_by_id.keys():
            if key in b_by_id:
                similarity = DiffUtils.similarity_dicts(a_by_id[key], b_by_id[key], case_sensitive, fuzzy)
                if similarity < 1 and trace > 0:
                    log(f'Dict key {key} similarity = {similarity:.1f}')
                    log(DiffUtils._dict_to_sorted_kvset_shallow(a_by_id[key], case_sensitive, fuzzy))
                    log(DiffUtils._dict_to_sorted_kvset_shallow(b_by_id[key], case_sensitive, fuzzy))
                obj_similarities.append(similarity)
        # Reduce similarity score for both mismatched list items and differences in attributes for matching objects
        return id_similarity * statistics.fmean(obj_similarities) if len(obj_similarities) else 0.0

    @staticmethod
    def lists_of_dicts_match(list_a: list[dict], list_b: list[dict], case_sensitive: bool = True, fuzzy: bool = False, order_sensitive: bool = False, trace: Trace = Trace.OFF) -> bool:
        """Returns True if the lists are identical, with options to ignore case and list order."""
        if not isinstance(list_a, list):
            raise InvalidArgsError(f'list_a must be a list of dicts, not {type(list_a)}')
        if not isinstance(list_b, list):
            raise InvalidArgsError(f'list_b must be a list of dicts, not {type(list_b)}')
        for item in list_a:
            if not isinstance(item, dict):
                raise InvalidArgsError(f'list_a includes an item that is not a dict: {type(item)}')
        for item in list_b:
            if not isinstance(item, dict):
                raise InvalidArgsError(f'list_b includes an item that is not a dict: {type(item)}')
        if len(list_a) != len(list_b):
            return False
        a = [DiffUtils._dict_to_sorted_kv_lines_shallow(item, case_sensitive, fuzzy) for item in list_a]
        b = [DiffUtils._dict_to_sorted_kv_lines_shallow(item, case_sensitive, fuzzy) for item in list_b]
        if not order_sensitive:
            a = sorted(a)
            b = sorted(b)
        for a, b in zip(a, b):
            if a != b:
                return False
        return True