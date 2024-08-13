from typing import Any
from config.defaults import Default
from core.conversation import Conversation
from core.data import Data
from core.data_bundle import DataBundle
from core.data_type import DataType
from core.output_format import Format
from core.interaction import Interaction
from primitives.base_primitive import BasePrimitive
from utils.exceptions import InvalidArgsError
from utils.logger import Trace, log

class QSQuestion():
    def __init__(self, question_id:str, question: str, output_data_type: DataType = DataType.TEXT) -> None:
        '''Defines a question for the QuestionSet primitive.

        Args:
        - question: plain text question.
        - output_data_type: enables post-processing.
        '''
        self.question_id = question_id
        self.question = question
        self.output_data_type = output_data_type

class QSResult():
    def __init__(self, question_id:str, answer: DataBundle):
        self.question_id: str = question_id
        self.answer: DataBundle = answer
        self.section_tag: str = Default.SECTION_NOT_FOUND_TAG
        self.section_text: str = ''
        self.page_number: int = -1

    def set_section(self, section_tag: str, section_text: str) -> None:
        self.section_tag = section_tag
        section_text = section_text.strip()
        section_text = section_text.removeprefix(f'<{section_tag}>')
        section_text = section_text.removesuffix(f'</{section_tag}>')
        self.section_text = section_text.strip()

    def __repr__(self) -> str:
        if self.page_number >= 0:
            return f'QSResult({self.question_id}): {self.answer} found on page #{self.page_number}, snippet:\n{self.section_text}'
        else:
            return f'QSResult({self.question_id}): {self.answer} PAGE NOT FOUND, snippet[\n{self.section_text}]'

class QSExample():
    def __init__(self, question: str, answer: str) -> None:
        '''Defines a few-shot example for the QuestionSet primitive.

        Args:
        - question: plain text question.
        - answer: plain text answer.
        '''
        self.question = question
        self.answer = answer

class QuestionSet(BasePrimitive):
    """Combines multiple zero-shot questions into a single text prompt.

    This Primitive uses a table format with a column for each question.
    This allows injecting question-specific definitions and preambles.
    """
    next_id = 1

    @staticmethod
    def section_not_found_tag() -> str:
        '''This tag is returned for questions that don't have a matching Section.'''
        return Default.SECTION_NOT_FOUND_TAG

    def __init__(
            self,
            questions: list[QSQuestion],
            examples: list[QSExample] = [],
            preamble: str = '',
            custom_id: str | None = None,
            trace: int = 0
        ):
        """Configures the QuestionSet primitive.

        Args:
        - questions: list of ZeroShot objects.
        - preamble: optional system instructions rendered on the very top of the prompt.
        - custom_id: optional id for tracking in the database
        - trace: set to Trace.ON or Trace.VERBOSE
        """
        if not questions:
            raise InvalidArgsError('questions are required.')
        if not custom_id:
            custom_id = f'Instructions_{QuestionSet.next_id}'
            QuestionSet.next_id += 1
        self.questions = questions
        self.examples = examples
        self.custom_id = custom_id
        self.preamble = preamble
        self.trace = trace
        self.output_data_type = DataType.QUESTIONSET_ANSWERS

    @property
    def id(self) -> str:
        return self.custom_id

    @property
    def output_type(self) -> DataType:
        return self.output_data_type

    @property
    def output_format(self) -> Format:
        return Format.DEFAULT

    def parse_results(self, interaction: Interaction, trace: Trace = Trace.OFF) -> list[QSResult]:
        if not isinstance(interaction, Interaction):
            raise InvalidArgsError(f'parse_results called with {type(interaction)} instead of Interaction')
        bundle = interaction.output
        if bundle.is_empty():
            log(f'QuestionSet got empty results: {interaction.output}')
            return []
        if bundle.first().data_type != DataType.JSON_ARRAY:
            raise InvalidArgsError(f'parse_results requires Interaction with output DataType.JSON_ARRAY, called with: {bundle.first().data_type}')
        answer_list = bundle.first().value
        if len(answer_list) != len(self.questions):
            raise InvalidArgsError(f'{len(answer_list)} answers in interaction output does not match the {len(self.questions)} questions in this QuestionSet.')
        results = []
        for question, answer in zip(self.questions, answer_list):
            log(f'QSResult {question.question_id} -> {answer}')
            tag_name = str(answer.value).strip().lstrip('<').lstrip('&lt;').rstrip('>').rstrip('&gt;').strip()
            results.append(QSResult(question.question_id, DataBundle([Data(tag_name, DataType.TEXT)])))
        return results

    def index_results(self, interaction: Interaction, trace: Trace = Trace.OFF) -> dict[str, QSResult]:
        results = self.parse_results(interaction, trace)
        return {r.question_id: r for r in results}

    def _build_question_table(self) -> str:
        question_headers: list[str] = ['']
        questions: list[str] = ['QUESTIONS:']
        answers: list[str] = ['ANSWERS:']
        column_number = 0
        for i, q in enumerate(self.examples):
            column_number = column_number + 1
            question_header = f'**Question #{column_number}**'
            #max_len = max(len(question_header), len(q.question), len(answer_header), len(q.answer))
            question_headers.append(question_header )  # .rjust(max_len, ' ')
            questions.append(q.question)
            answers.append(q.answer)
        for i, q in enumerate(self.questions):
            column_number = column_number + 1
            question_header = f'**Question #{column_number}**'
            # max_len = max(len(question_header), len(q.question), len(answer_header))
            question_headers.append(question_header)
            questions.append(q.question)
        separators = [':---:' for i in range(len(questions))]
        rows = []
        rows.append(f'| {" | ".join(question_headers)} |')
        rows.append(f'| {" | ".join(separators)} |')
        rows.append(f'| {" | ".join(questions)} |')
        rows.append(f'| {" | ".join(answers)} |')
        return '\n'.join(rows)

    def build_prompt(self, interaction: Interaction, **kwargs: Any) -> None:
        ctx = interaction.context
        preamble = self.preamble
        if not preamble:
            if ctx.is_empty():
                preamble = '''You are a detail oriented AI assistant designed to provide concise and accurate answers to the given questions.
    You are given a Markdown Table with a row of questions. Your goal is to answer each of these questions and write the answer to the "ANSWERS" row on the bottom of the same markdown table.
    '''
            else:
                preamble = '''You are a detail oriented AI assistant specializing in reading documents and answering questions about these documents.
Your answers must be concise, accurate, and strictly based on information from the following DOCUMENT:
'''
        if ctx.is_empty():
            instructions = '''
<INSTRUCTIONS>
1. Answer all questions found in the "QUESTIONS:" row in the Markdown Table below, and write the answers to the "ANSWERS:" row of the same markdown table.
2. Do not add any additional comments, headers, or labels to your answers.
</INSTRUCTIONS>
'''
        else:
            instructions = f'''
<INSTRUCTIONS>
1. Answer all questions found in the "QUESTIONS:" row in the Markdown Table below, and write the answers to the "ANSWERS:" row of the same markdown table.
2. Use only the information from the DOCUMENT above to answer these questions. If the document does not provide enough information to answer some of the questions, write the code "{Default.SECTION_NOT_FOUND_TAG}" to the answer cell.
3. Do not add any additional comments, headers, or labels to your answers.
</INSTRUCTIONS>
'''
        instructions = instructions + '\n**Markdown Table with QUESTIONS and ANSWERS**\n'
        question_table = self.inject_prompt_parameters(self._build_question_table(), **kwargs)
        if ctx.is_empty() or ctx.is_single_text():
            context = '' if ctx.is_empty() else f'\n<DOCUMENT>\n{ctx.to_text()}\n</DOCUMENT>'
            prompt = f'{preamble}\n{context}\n{instructions}\n{question_table}'
            self.trace_prompt(prompt, self.trace)
            interaction.prompt = Conversation.from_text(prompt)
        elif ctx.is_single_image() or ctx.is_single_pdf():
            prompt = DataBundle.from_text(preamble)
            prompt.append(ctx.first_data_bundle().first())
            prompt.append(Data(instructions, DataType.TEXT))
            prompt.append(Data(question_table, DataType.TEXT))
            if self.trace > 0:
                log(f'---------PROMPT ({len(prompt)} parts)-----------')
                for part in prompt:
                    log(part)
                log(f'---------PROMPT_END-----------')
            interaction.prompt = Conversation.from_data_bundle(prompt)
        else:
            raise InvalidArgsError('QuestionSet requires a single text, image, or pdf input context.')