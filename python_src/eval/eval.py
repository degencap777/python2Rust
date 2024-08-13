import random
from typing import Optional
from collections.abc import Callable
from storage.base_file_store import BaseFileStore
from storage.local_storage import LocalStorage
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from utils.sampling_method import SamplingMethod
from utils.data_bundle import DataBundle
from utils.exceptions import InvalidArgsError
from utils.logger import Trace, log

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

    