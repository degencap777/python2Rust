from typing import Any
from core.conversation import Conversation
from core.data import Data
from core.data_bundle import DataBundle
from core.data_type import DataType
from core.output_format import Format
from core.interaction import Interaction
from primitives.base_primitive import BasePrimitive
from utils.exceptions import InvalidArgsError
from utils.logger import log

# Alternative name ideas: Manual, Unstructured, Custom, Template, Prompt,
class Instructions(BasePrimitive):
    """Generate prompt by adding the instructions to the end of text context."""
    next_id = 1

    def __init__(
            self,
            instructions: str,
            preamble: str = '',
            output_data_type: DataType = DataType.TEXT,
            output_format: Format = Format.DEFAULT,
            custom_id: str | None = None,
            trace: int = 0
        ):
        """Configures the Instructions primitive.

        Args:
        - instructions: plain text, required
        - preamble: optional system instructions rendered on the very top of the prompt.
        - output_data_type: enables post-processing
        - output_format: can be used to simplify parsing CoT prompts
        - custom_id: optional id for tracking in the database
        - trace: set to Trace.ON or Trace.VERBOSE
        """
        if not instructions:
            raise InvalidArgsError('instructions are required.')
        if not custom_id:
            custom_id = f'Instructions_{Instructions.next_id}'
            Instructions.next_id += 1
        self.custom_id = custom_id
        self.instructions = instructions
        self.preamble = preamble
        self.output_data_type = output_data_type
        self.format = output_format
        self.trace = trace

    @property
    def id(self) -> str:
        return self.custom_id

    @property
    def output_type(self) -> DataType:
        return self.output_data_type

    @property
    def output_format(self) -> Format:
        return self.format

    def build_prompt(self, interaction: Interaction, **kwargs: Any) -> None:
        ctx = interaction.context
        if ctx.is_empty() or ctx.is_single_text():
            context = '' if ctx.is_empty() else f'{ctx.to_text()}\n\n'
            prompt = f'{context}{self.instructions}'
            prompt = self.inject_prompt_parameters(prompt, **kwargs)
            if self.preamble:
                prompt = f'{self.preamble}\n\n{prompt}'
            self.trace_prompt(prompt, self.trace)
            interaction.prompt = Conversation.from_text(prompt)
        elif ctx.is_single_image() or ctx.is_single_pdf():
            if self.preamble:
                prompt = DataBundle.from_text(self.preamble)
            else:
                prompt = DataBundle.empty()
            prompt.append(ctx.first_data_bundle().first())
            prompt.append(Data(self.inject_prompt_parameters(self.instructions, **kwargs), DataType.TEXT))
            if self.trace > 0:
                log(f'---------PROMPT ({len(prompt)} parts)-----------')
                for part in prompt:
                    log(part)
                log(f'---------PROMPT_END-----------')
            interaction.prompt = Conversation.from_data_bundle(prompt)
        else:
            raise InvalidArgsError('Instructions requires a single text, image, or pdf input context.')