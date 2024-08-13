from PIL import Image as PI
from core.data import Data
from core.error_code import ErrorCode
from core.data_source import DataSource
from core.data_type import DataType
from core.inference_metadata import InferenceMetadata
from utils.exceptions import InvalidArgsError

class DataBundle():
    """Ordered list of texts, images, videos intended for multimodal inference input and output."""

    def __init__(
        self,
        items: list[Data],
        error_code: ErrorCode = ErrorCode.UNDEFINED,
        data_source: DataSource | None = None
    ):
        self.items = items
        self.error_code = error_code
        self.inference_metadata: InferenceMetadata
        self.data_source = data_source         # optional metadata about the source of data for this bundle

    def __repr__(self):
        if self.is_empty():
            return 'EMPTY'
        if len(self.items) == 1:
            return f'Bundle[{self.items[0]}]'
        else:
            return f'Bundle[{len(self.items)} items starting with {self.items[0]}]'
    
    def is_empty(self) -> bool:
        if not self.items:
            return True
        for item in self.items:
            if item.value:
                return False
        return True

    @staticmethod
    def from_float(value: float) -> 'DataBundle':
        if not isinstance(value, (float, int)):
            raise InvalidArgsError(f'Expected a float, got: {type(value)}')
        return DataBundle([Data(float(value), DataType.FLOAT)])
    
    @staticmethod
    def from_image(image: PI.Image) -> 'DataBundle':
        if not isinstance(image, PI.Image):
            raise InvalidArgsError(f'Expected a PI.Image, got: {type(image)}')
        return DataBundle([Data(image)])
