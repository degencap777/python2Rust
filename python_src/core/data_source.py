from enum import Enum
from dataclasses import dataclass

class DataSourceType(Enum):
    DOCUMENT = 1
    IMAGE = 2
    DIRECTORY = 3

@dataclass
class DataSource():
    """Information about the source of data in Batches and Interactions."""
    source_type: DataSourceType = DataSourceType.DOCUMENT
    name: str = ''
    location: str = ''
    version: str = ''

    def __repr__(self) -> str:
        v = self.version or ''
        return f'{self.name}.{v}'