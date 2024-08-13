from datetime import datetime
from dataclasses import dataclass

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
