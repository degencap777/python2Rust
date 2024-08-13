from abc import ABCMeta, abstractmethod
from typing import Any
from core.data_type import DataType
from core.output_format import Format
from core.interaction import Interaction
from utils.logger import log


class BasePrimitive(object, metaclass=ABCMeta):
    """Abstract base class for all Primitives."""
    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @property
    @abstractmethod
    def output_type(self) -> DataType:
        pass

    @property
    @abstractmethod
    def output_format(self) -> Format:
        pass

    @abstractmethod
    def build_prompt(self, interaction: Interaction, **kwargs: Any) -> None:
        pass

    def inject_prompt_parameters(self, prompt: str, **kwargs: Any) -> str:
        """Expecting string keys matching curly braces in the prompt string."""
        for key, value in kwargs.items():
            prompt = prompt.replace('{%s}' % key, value)
        return prompt

    def is_output_json(self) -> bool:
        return self.output_type in [DataType.JSON_ARRAY, DataType.JSON_DICT]

    def trace_prompt(self, prompt: str, trace: int) -> None:
        log(f'''
---------PROMPT ({len(prompt)} chars)-----------
{prompt}
---------PROMPT_END-----------''', trace)