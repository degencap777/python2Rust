from datetime import datetime
from dataclasses import dataclass

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
