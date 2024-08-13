
from enum import Enum
from typing import Optional
from collections.abc import Callable
from utils.sampling_method import SamplingMethod
from utils.data_bundle import DataBundle
from utils.exceptions import InvalidArgsError

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
