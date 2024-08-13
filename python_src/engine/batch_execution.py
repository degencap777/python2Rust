from datetime import datetime
from dataclasses import dataclass
from core.batch import Batch
from primitives.base_primitive import BasePrimitive

@dataclass
class BatchExecution():
    primitive: BasePrimitive
    batch: Batch
    start_time: datetime
    duration_seconds: float | None = None
