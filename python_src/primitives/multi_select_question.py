from typing import Any

from utils.exceptions import InvalidArgsError, BadPredictionError
from core.conversation import Conversation
from core.data_type import DataType
from core.output_format import Format
from core.interaction import Interaction
from primitives.base_primitive import BasePrimitive


class MultiSelectQuestion(BasePrimitive):
    """Asks a multi-select question with an option to verify positive answers."""
    next_id = 1

    def __init__(
        self,
        question: str,
        answer_options: list[str],
        verify_answers: bool = False,
        output_format: Format = Format.DEFAULT,
        custom_id: str | None = None,
        preamble: str = '',
        rules: list[str] | None = None,
        answer_tags: list[Any] | None = None,
        suppress_default_rules: bool = False,
        trace: int = 0
    ):
        """Configures MultiSelectQuestion primitive.

        Args:
        - question: plain text question, required.
        - answer_options: list of possible answers without labels - the engine will generate them.
        - answer_tags: any additional data associated with answers. This has no impact on inference and intended to simplify mapping answers to any other entities.
        - verify_answers: runs an additional MultipleChoice question to confirm each positive answer from MultiSelect.
        - output_data_type: enables post-processing
        - output_format: can be used to simplify parsing CoT prompts
        - custom_id: optional id for tracking in the database
        - preamble: optional system instructions rendered on top of the prompt.
        - rules: optional rules rendered on the bottom of the prompt but above the question.
        - suppress_default_rules: removes instructions related to NOT_FOUND_TAG.
        - trace: set to Trace.ON or Trace.VERBOSE
        """
        if not question:
            raise InvalidArgsError('question is required')
        if not answer_options:
            raise InvalidArgsError('answer_options are required')
        if answer_tags and not isinstance(answer_tags, list):
            raise InvalidArgsError('answer_tags must be a list')
        if answer_tags and len(answer_tags) != len(answer_options):
            raise InvalidArgsError('answer_tags must be a list of the same length as answer_options.')
        if not custom_id:
            custom_id = f'MultiSelectQuestion_{MultiSelectQuestion.next_id}'
            MultiSelectQuestion.next_id += 1
        self.custom_id = custom_id
        self.question = question
        self.preamble = preamble
        self.answer_options = answer_options
        self.answer_tags = answer_tags
        self.rules = [r for r in rules or [] if r is not None]
        self.suppress_default_rules = suppress_default_rules
        self.verify_answers = verify_answers
        self.output_data_type = DataType.MULTISELECT_ANSWERS
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
        if interaction.context.is_empty():
            raise InvalidArgsError('Question called with empty input')
        if not interaction.context.is_single_text():
            raise InvalidArgsError('Question requires a single text input')
        prompt = self.fill_template(
            interaction.context.to_text(),
            self.question,
            self.build_rules(),
            self.build_answer_options(),
        )
        prompt = self.inject_prompt_parameters(prompt, **kwargs)
        if self.preamble:
            prompt = f'{self.preamble}\n\n{prompt}'
        self.trace_prompt(prompt, self.trace)
        interaction.prompt = Conversation.from_text(prompt)

    def build_rules(self):
        items = self.answer_options + ['None of the above.']
        letters = [f'{chr(65+i)}' for i in range(len(items))]
        default_rules = [
            'Answer this question using only the information from the CONTEXT above.',
            f'The ANSWER must be a comma separated list of one or more of the following letters that correspond to the given answer choices: {", ".join(letters)}.'
        ]
        custom_rules = (
            default_rules + self.rules
            if not self.suppress_default_rules
            else self.rules
        )
        numbered_items = [f'{i+1}. {rule}' for i, rule in enumerate(custom_rules)]
        return '\n'.join(numbered_items)

    def build_answer_options(self) -> str:
        items = self.answer_options + ['None of the above.']
        answer_options = [
            f'{chr(65+i)}. {answer_option}' for i, answer_option in enumerate(items)
        ]
        return '\n'.join(answer_options)

    def get_answer_by_label(self, label: str) -> str:
        index = self.get_answer_index(label)
        if not self.answer_options or len(self.answer_options) <= index:
            raise BadPredictionError(
                f'Label {label} does not match any of the {len(self.answer_options)+1} answer options.'
            )
        return self.answer_options[index]

    def get_answer_index(self, label: str) -> int:
        if len(label) != 1:
            raise BadPredictionError('label must be a single character')
        return ord(label[0]) - 65

    def is_empty_answer_label(self, label: str) -> bool:
        return self.get_answer_index(label) == len(self.answer_options)

    def is_empty_answer(self, comma_separated_letters: str) -> bool:
        letters = comma_separated_letters.split(',')
        if len(letters) != 1:
            return False
        return self.is_empty_answer_label(letters[0].strip())

    def fill_template(
        self, context: str, question: str, rules: str, answers: str
    ) -> str:
        return """Answer the multi-select QUESTION below using only the information from the following CONTEXT.
<CONTEXT>
{0}
</CONTEXT>

<INSTRUCTIONS>
Use the following rules to answer the multi-select QUESTION below:
{1}
</INSTRUCTIONS>

QUESTION: {2}
{3}
ANSWER:""".format(
            context, rules, question, answers
        )
