from datetime import datetime
from dataclasses import dataclass

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
