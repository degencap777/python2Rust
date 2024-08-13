from typing import Any
from utils.exceptions import InvalidArgsError, BadPredictionError
from core.data_type import DataType
from core.output_format import Format
from core.conversation import Conversation
from core.interaction import Interaction
from primitives.base_primitive import BasePrimitive

class MultipleChoiceQuestion(BasePrimitive):
    """Answers a multiple choice question and returns a label like 'A', 'B', 'C'."""
    next_id = 1

    def __init__(
            self,
            question: str,
            answer_options: list[str],
            output_format: Format = Format.DEFAULT,
            custom_id: str | None = None,
            preamble: str = '',
            rules: list[str] | None=None,
            trace: int = 0
        ):
        """Configures MultipleChoiceQuestion primitive.

        Args:
        - question: plain text, required.
        - answer_options: list of possible answers without labels - the engine will generate them.
        - output_data_type: enables post-processing
        - custom_id: optional id for tracking in the database
        - preamble: optional system instructions rendered on top of the prompt.
        - rules: optional rules rendered on the bottom of the prompt but above the question.
        - trace: set to Trace.ON or Trace.VERBOSE
        """

        if not question:
            raise InvalidArgsError('question is required')
        if not answer_options:
            raise InvalidArgsError('answer_options are required')
        if not custom_id:
            custom_id = f'MultipleChoiceQuestion_{MultipleChoiceQuestion.next_id}'
            MultipleChoiceQuestion.next_id += 1
        self.custom_id = custom_id
        self.preamble = preamble
        if not question:
            raise InvalidArgsError('question is empty')
        if not answer_options:
            raise InvalidArgsError('answer_options are empty')
        self.question = question
        self.answer_options = answer_options
        self.rules = [r for r in rules or [] if r is not None]
        self.output_data_type = DataType.CHAR
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
        if not interaction.context.is_empty():
            if not interaction.context.is_single_text():
                raise InvalidArgsError('This question only works with Text input.')
        prompt = self.fill_template( interaction.context.to_text(), self.question, self.build_rules(interaction), self.build_answer_options())
        prompt = self.inject_prompt_parameters(prompt, **kwargs)
        if self.preamble:
            prompt = f'{self.preamble}\n\n{prompt}'
        self.trace_prompt(prompt, self.trace)
        interaction.prompt = Conversation.from_text(prompt)

    def build_rules(self, interaction: Interaction):
        items = []
        if not interaction.context.is_empty():
            items.append('You must answer this question using only the information from the CONTEXT above.')
            items.append('Answer with one letter corresponding to the correct answer option.')
        if self.rules:
            items += self.rules
        numbered_items = [f'{i+1}. {rule}' for i, rule in enumerate(items)]
        return '\n'.join(numbered_items)

    def build_answer_options(self) -> str:
        items = self.answer_options
        answer_options = [f'{chr(65+i)}. {answer_option}' for i, answer_option in enumerate(items)]
        return '\n'.join(answer_options)

    def list_answer_labels(self) -> str:
        labels = [f'({chr(65+i)})' for i in range(len(self.answer_options))]
        return ', '.join(labels[:-1]) + ', or ' + labels[-1]

    def get_answer_by_label( self, label: str ) -> str:
        index = self.get_answer_index( label )
        if not self.answer_options or len(self.answer_options) <= index:
            raise BadPredictionError(f'Label {label} does not match any of the {len(self.answer_options)} answer options.')
        return self.answer_options[index]

    def get_answer_index( self, label: str ) -> int:
        if len(label) != 1:
            raise BadPredictionError('label must be a single character')
        return ord( label[0] ) - 65

    def fill_template(self, context: str, question: str, rules: str, answers: str) -> str:
        if not context:
            prompt = f'Answer the multiple choice QUESTION below with {self.list_answer_labels()}.\n'
            if rules:
                prompt += f'Use the following rules to answer the multiple choice QUESTION below:\n{rules}\n'
            prompt += f'QUESTION: {question}\n{answers}\nANSWER:'
            return prompt
        else:
            return """Answer the multiple choice QUESTION below using only the information from the following CONTEXT.
<CONTEXT>
{0}
</CONTEXT>

Use the following rules to answer the multiple choice QUESTION below:
{1}
QUESTION: {2}
{3}
ANSWER:""".format(context, rules, question, answers)