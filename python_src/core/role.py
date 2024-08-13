from enum import Enum

class Role(Enum):
    """Roles for multiturn (conversational) inference."""

    USER = 1
    MODEL = 2
    SYSTEM = 3
    FUNCTION = 4

    def __repr__(self):
        return repr(self.name)

    def to_string(self) -> str:
        return self.name

    @classmethod
    def from_string(cls, string: str) -> 'Role':
        return cls[string]