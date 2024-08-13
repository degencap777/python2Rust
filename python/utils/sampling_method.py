from enum import Enum

class SamplingMethod(Enum):
    NONE = 0
    RANDOM = 1
    HEAD = 2
    TAIL = 3