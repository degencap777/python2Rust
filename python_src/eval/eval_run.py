from datetime import datetime
from dataclasses import dataclass

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
