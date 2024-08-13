from typing import Any
from core.conversation import Conversation
from core.data import Data
from core.data_type import DataType
from core.data_bundle import DataBundle
from core.few_shot_example import FewShotExample
from core.output_format import Format
from core.interaction import Interaction
from primitives.base_primitive import BasePrimitive
from utils.exceptions import InvalidArgsError
from utils.logger import log, Trace

class FewShot(BasePrimitive):
    """Construct few-shot prompts based on text and multimodal inputs."""
    next_id = 1

    def __init__(self,
            examples: list[tuple[Any, ...]],
            preamble: str = '',     # Preamble is injected on top of the prompt (above the list of examples)
            instructions: str = '', # Instructions are injected below the list of examples
            input_label: str='',
            output_label: str='',
            render_labels_as_tags: bool=True,  # Wraps examples in Html tags to help the model understand the structure of long prompts.
            output_data_type: DataType = DataType.TEXT,
            output_format: Format = Format.DEFAULT,
            custom_id: str | None=None,
            trace: Trace = Trace.OFF
        ):
        """Construct a few-shot prompt from examples specified as a list of tuples. Each tuple must have two elements such as an image and a string."""
        if not custom_id:
            custom_id = f'FewShot_{FewShot.next_id}'
            FewShot.next_id += 1
        self.custom_id = custom_id
        self.output_data_type = output_data_type
        self.format = output_format
        self.trace = trace
        self.preamble = preamble
        self.instructions = instructions
        self.input_label = input_label
        self.output_label = output_label
        self.render_labels_as_tags = render_labels_as_tags
        self.examples: list[FewShotExample] = []
        for example in examples:
            if len(example) < 2 or len(example) > 3:
                raise InvalidArgsError(f'FewShot examples must be tuples of two or three elements, got {len(example)}')
            if len(example) == 2:
                input, output = example
                self.examples.append(FewShotExample(Data(input), Data(output)))
            else:
                input, output, source = example
                self.examples.append(FewShotExample(Data(input), Data(output), source))
        if not self.examples:
            raise InvalidArgsError('FewShot requires at least one example.')

    @property
    def id(self) -> str:
        return self.custom_id

    @property
    def output_type(self) -> DataType:
        return self.output_data_type

    @property
    def output_format(self) -> Format:
        return self.format

    def clone(self) -> 'FewShot':
        return FewShot(
            examples = [(example.input.value, example.output.value) for example in self.examples],
            preamble = self.preamble,
            instructions = self.instructions,
            input_label = self.input_label,
            output_label = self.output_label,
            render_labels_as_tags = self.render_labels_as_tags,
            output_data_type = self.output_data_type,
            output_format = self.output_format,
            custom_id = self.custom_id,
            trace = self.trace
        )

    def build_prompt(self, interaction: Interaction, **kwargs: Any) -> None:
        if interaction.context.is_empty():
            raise InvalidArgsError('FewShot called with empty input.')
        if not len(self.examples):
            raise InvalidArgsError('FewShot does not have any examples.')
        if not interaction.context.is_single_turn():
            raise InvalidArgsError('FewShot called with a multi-turn conversation in context which is not supported yet.')
        prompt = DataBundle.empty()
        if self.preamble:
            preamble_tag_open = '<INSTRUCTIONS>\n' if self.render_labels_as_tags else ''
            preamble_tag_close = '\n</INSTRUCTIONS>\n' if self.render_labels_as_tags else ''
            prompt.append(Data(preamble_tag_open + self.preamble + preamble_tag_close + '\n', DataType.TEXT))
        if self.render_labels_as_tags:
            prompt.append(Data('\n<EXAMPLES>\n', DataType.TEXT))

        for example in self.examples:
            match example.input.data_type:
                case DataType.TEXT:
                    tag_open, tag_close = self.build_input_tags()
                    prompt.append(Data(tag_open + self.inject_prompt_parameters(example.input.value, **kwargs) + tag_close + '\n', DataType.TEXT))
                case DataType.IMAGE:
                    prompt.append(example.input)
                case DataType.PDF:
                    prompt.append(example.input)
                case _:
                    raise InvalidArgsError(f'FewShot configured with data type which is not supported yet: {example.input.data_type}')
            match example.output.data_type:
                case DataType.TEXT:
                    tag_open, tag_close = self.build_output_tags()
                    prompt.append(Data(tag_open + self.inject_prompt_parameters(example.output.value, **kwargs) + tag_close + '\n', DataType.TEXT))
                case DataType.IMAGE:
                    prompt.append(example.output)
                case DataType.PDF:
                    prompt.append(example.output)
                case _:
                    raise InvalidArgsError(f'FewShot configured with data type which is not supported yet: {example.output.data_type}')

        if self.render_labels_as_tags:
            prompt.append(Data('\n</EXAMPLES>\n', DataType.TEXT))
        if self.instructions:
            prompt.append(Data(f'\n{self.instructions}\n', DataType.TEXT))
        data = interaction.context.first_data_bundle().clone()
        if self.input_label and data.first().data_type == DataType.TEXT:
            tag_open, tag_close = self.build_input_tags()
            data.first().value = tag_open + data.first().value + tag_close
        prompt.import_bundle(data)
        if self.output_label:
            tag_open, _ = self.build_output_tags()
            prompt.append(Data(tag_open, DataType.TEXT))
        if self.trace > 0:
            log(f'FewShot: {prompt}')
        interaction.prompt = Conversation.from_data_bundle(prompt)

    def build_input_tags(self):
        tag_open = ''
        tag_close = ''
        if self.input_label:
            tag_open = f'\n<{self.input_label.strip()}>\n' if self.render_labels_as_tags else self.input_label + ' '
            tag_close = f'\n</{self.input_label.strip()}>' if self.render_labels_as_tags else ''
        return tag_open,tag_close

    def build_output_tags(self):
        tag_open = ''
        tag_close = ''
        if self.output_label:
            tag_open = f'\n<{self.output_label.strip()}>\n' if self.render_labels_as_tags else self.output_label + ' '
            tag_close = f'\n</{self.output_label.strip()}>' if self.render_labels_as_tags else ''
        return tag_open,tag_close

    def format_model_output(self, prediction: str) -> str:
        if not self.output_label:
            return prediction
        if self.render_labels_as_tags:
            tag_open, tag_close = self.build_output_tags()
            if prediction.startswith(tag_open):
                prediction = prediction[len(tag_open):].strip()
            if prediction.endswith(tag_close):
                prediction = prediction[:-len(tag_close)].strip()
        else:
            if prediction.lower().startswith(self.output_label.lower()):
                prediction = prediction[len(self.output_label):].strip()
        return prediction
