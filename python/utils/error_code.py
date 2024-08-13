from enum import Enum

class ErrorCode(Enum):
    """Represents errors that Proton can workaround without throwing an exception."""

    UNDEFINED = 1
    RESPONSE_BLOCKED = 2  # Model prediction blocked by filters, can retry with a different model
