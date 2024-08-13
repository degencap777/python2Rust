from datetime import datetime
from dataclasses import dataclass

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
