from datetime import datetime
from dataclasses import dataclass

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
