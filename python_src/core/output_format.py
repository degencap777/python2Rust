from enum import Enum

class Format(Enum):
    """All primitives are expected to return one of these formats."""

    DEFAULT = 1              # Use full model output
    ANSWER_ON_LAST_LINE = 2  # Used to parse output of Chain Of Thought prompts

    def __repr__(self):
        return repr(self.name)
