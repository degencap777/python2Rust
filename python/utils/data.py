from typing import Any
from data_type import DataType
from datetime import datetime
from exceptions import InvalidArgsError
from PIL import Image as PI

def remove_line_breaks(text: str) -> str:
    return ' '.join(text.splitlines())

def truncate(text: str, max_length: int, ellipsis: str = '...', no_linebreaks: bool = False) -> str:
    """Truncates a string to a given length."""
    if no_linebreaks:
        text = remove_line_breaks(text)
    if len(text) > max_length:
        return text[:max_length - len(ellipsis)] + ellipsis
    else:
        return text
    
class Data():
    """Unit of Input or Output data for multimodal inference."""

    def __init__(self, value: Any, data_type: DataType = DataType.UNDEFINED, id: str = ''):
        self.value = value
        self.id = id
        if data_type == DataType.UNDEFINED:
            self.data_type = Data.detect_data_type( value )
        else:
            self.data_type = data_type

    def __repr__(self):
        if self.data_type == DataType.IMAGE:
            width, height = self.value.size
            return f'{self.data_type.name}: [{width}x{height}]'
        elif self.data_type == DataType.PDF:
            return f'PDF: [{len(self.value)}]'
        elif self.data_type == DataType.TEXT:
            if not self.value:
                return f'{self.data_type.name}[0]: None'
            truncated_to_one_line = truncate(self.value, 100).replace('\n', ' ')
            return f'{self.data_type.name}[{len(self.value)}]: {truncated_to_one_line}'
        else:
            return f'{self.data_type.name}: {self.value}'
        
    def detect_data_type(value: Any) -> DataType:
        if isinstance(value, bool):  # bool is subclass of int so this must come first
            return DataType.BOOL
        if isinstance(value, int):
            return DataType.INT
        if isinstance(value, str):
            return DataType.TEXT
        if isinstance(value, float):
            return DataType.FLOAT
        if isinstance(value, datetime):
            return DataType.DATE
        if isinstance(value, PI.Image):
            return DataType.IMAGE
        if isinstance(value, list):
            if all([isinstance(item, str) and len(item) == 1 for item in value]):
                return DataType.MULTISELECT_ANSWERS
            else:
                return DataType.JSON_ARRAY
        if isinstance(value, dict):
            return DataType.JSON_DICT
        if isinstance(value, bytes):
            return DataType.PDF
        raise InvalidArgsError(f'Unknown data type: {type(value)}.')
