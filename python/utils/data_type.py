from enum import Enum

class DataType(Enum):
    """Multimodal inputs and outputs are expected to use these data types."""

    UNDEFINED = 1   # triggers automatic type detection
    TEXT = 2        # str
    INT = 3
    FLOAT = 4
    BOOL = 5
    CHAR = 6
    DATE = 7
    JSON_ARRAY = 8  # TODO: should we rename it to LIST_OF_STRINGS?
    JSON_DICT = 9   # TODO: should we rename it to DICT?
    IMAGE = 10      # PIL.Image
    MULTISELECT_ANSWERS = 11  # list of answer option labels like ['B', 'F']
    PDF = 12
    QUESTIONSET_ANSWERS = 13  # list of answers
    DATETIME = 14

    def __repr__(self):
        return repr(self.name)

    def to_string(self) -> str:
        return self.name

    @classmethod
    def from_string(cls, string: str) -> 'DataType':
        return cls[string]
  