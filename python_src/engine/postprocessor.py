import json
from enum import IntEnum
from config.defaults import Default
from core.conversation import Conversation
from core.data_type import DataType
from core.data import Data
from core.interaction import Interaction
from engine.scaler import Scaler
from models.base_model import BaseModel
from primitives.base_primitive import BasePrimitive
from primitives.multi_select_question import MultiSelectQuestion
from primitives.multiple_choice_question import MultipleChoiceQuestion
from primitives.instructions import Instructions
from primitives.few_shot import FewShot
from primitives.question_set import QuestionSet
from utils.dates import DateUtils
from utils.numerals import NumberUtils
from utils.strings import StringUtils
from utils.exceptions import InvalidArgsError, BadPredictionError
from utils.logger import log


class DataFormatErrorPolicy(IntEnum):
    """Error handling options for Proton's Postprocessor."""
    RAISE_EXCEPTION = 0
    RETURN_EMPTY_RESULT = 1

class Postprocessor():
    def __init__(self, scaler: Scaler, error_policy: DataFormatErrorPolicy = DataFormatErrorPolicy.RAISE_EXCEPTION):
        self.scaler = scaler
        self.error_policy = error_policy

    def convert_text(self, model: BaseModel, text: str, data_type: DataType, primitive: BasePrimitive, trace: int = 0) -> Data:
        if not model:
            raise InvalidArgsError('Model is required')
        if not text:
            raise InvalidArgsError('text is required')
        if not primitive:
            raise InvalidArgsError('primitive is required')
        text = text.strip()
        if isinstance(primitive, FewShot):
            text = primitive.format_model_output(text)

        match data_type:
            case DataType.TEXT:
                return Data( text, DataType.TEXT )

            case DataType.CHAR:  # This option is used for multiple choice questions
                if isinstance(primitive, MultipleChoiceQuestion):
                    return self.parse_multiple_choice_answers(primitive, text)
                return Data( text[0], DataType.CHAR )

            case DataType.BOOL:
                pred = text[0].strip().lower()
                value = pred.startswith('true') or pred.startswith('yes') or pred.startswith('y') or pred.startswith('t')
                return Data( value, DataType.BOOL )

            case DataType.DATE:
                if StringUtils.is_date(text):
                    return Data( DateUtils.parse_date(text), DataType.DATE )
                text = self.extract_date(model, text)
                if StringUtils.is_date(text):
                    return Data( DateUtils.parse_date(text), DataType.DATE )
                return Data.none()

            case DataType.DATETIME:
                if StringUtils.is_date(text):
                    return Data( DateUtils.parse_date(text), DataType.DATETIME )
                text = self.extract_date(model, text)
                if StringUtils.is_date(text):
                    return Data( DateUtils.parse_date(text), DataType.DATETIME )
                return Data.none()

            case DataType.INT:
                text = text.replace('%', '').strip().rstrip('.')
                if StringUtils.is_numeric(text):
                    return Data( round(NumberUtils.parse_float( text )), DataType.INT )
                text = self.extract_number(model, text)
                if StringUtils.is_numeric(text):
                    return Data( round(NumberUtils.parse_float( text )), DataType.INT )
                return Data.none()

            case DataType.FLOAT:
                text = text.replace('%', '').strip().rstrip('.')
                if StringUtils.is_numeric(text):
                    return Data( NumberUtils.parse_float( text ), DataType.FLOAT )
                text = self.extract_number(model, text)
                if StringUtils.is_numeric(text):
                    return Data( NumberUtils.parse_float( text ), DataType.FLOAT )
                return Data.none()

            case DataType.JSON_ARRAY:
                if '[' not in text:
                    log(f'format_inference_output failed to parse JSON Array: {text}')
                    return Data.none()
                text = text.strip("'")  # Sometimes LLM wraps JSON into '''
                if text[0] != '[':  # Drop any commentary added by LLM
                    text = '[' + text.split('[', 1)[1]
                clean_json = StringUtils.fix_json_format(text)
                if clean_json == '[[]]':
                    clean_json = '[]'
                clean_json = clean_json.strip().lstrip('{').rstrip('}')
                if not clean_json.endswith(']'):
                    clean_json = clean_json + ']'
                json_object = f'{{"arr": {clean_json}}}'
                try:
                    obj = json.loads(json_object)
                    return Data(obj['arr'], DataType.JSON_ARRAY)
                except json.decoder.JSONDecodeError:
                    if self.error_policy == DataFormatErrorPolicy.RAISE_EXCEPTION:
                        raise BadPredictionError(f'Invalid JSON array in prediction {text}=>{clean_json}')
                    else:
                        log(f'Postprocessor failed to parse JSON Array and returns None due to error_policy={self.error_policy}: {text}')
                        return Data.none()

            case DataType.JSON_DICT:
                clean_json = StringUtils.fix_json_format(text)
                if not clean_json.startswith('{'):
                    if '{' in clean_json:
                        clean_json = clean_json.split('{', 1)[1]
                    clean_json = '{' + clean_json
                if not clean_json.endswith('}'):
                    clean_json = clean_json + '}'
                try:
                    obj = json.loads(clean_json)
                    return Data(obj, DataType.JSON_DICT)
                except json.decoder.JSONDecodeError as e:
                    if self.error_policy == DataFormatErrorPolicy.RAISE_EXCEPTION:
                        raise BadPredictionError(f'Invalid JSON Dict in prediction: {text}\nautoformatted to\n{clean_json}')
                    else:
                        log(f'Postprocessor failed to parse JSON Dict and returns None due to error_policy={self.error_policy}: {text}')
                        return Data.none()

            case DataType.MULTISELECT_ANSWERS:
                return self.parse_multiselect_answers(primitive, text)

            case DataType.QUESTIONSET_ANSWERS:
                if not isinstance(primitive, QuestionSet):
                    raise InvalidArgsError(f'DataType QUESTIONSET_ANSWERS is only supported for primitive QUESTIONSET_ANSWERS')
                return self.parse_question_set_answers(model, text, primitive, trace)

            case _:
                raise InvalidArgsError(f'Unknown output type: {data_type}')

    def parse_question_set_answers(self, model: BaseModel, text: str, primitive: QuestionSet, trace: int = 0) -> Data:
        '''Parse a markdown table row, convert each ell to requested data type.'''
        log(f'in parse_question_set_answers with text: {text}')
        text = text.strip().splitlines()[-1].strip().lstrip('|').rstrip('|')
        cells = text.split('|')
        log(f'parse_question_set_answers has {len(primitive.questions)} questions and {len(cells)} answers: {cells}', trace)
        answers = []
        for i, cell in enumerate(cells):
            cell = cell.strip().lstrip('*').rstrip('*').strip()  # Some models add markdown bold
            cell = cell.lstrip('<').rstrip('>').strip()  # Some models return the full XML section tag
            if i < len(primitive.questions):
                if not cell:
                    log(f'parse_question_set_answers is skipping empty answer for question #{i}: {primitive.questions[i]}')
                    answers.append(None)
                else:
                    question = primitive.questions[i]
                    answers.append(self.convert_text(model, text=cell, data_type=question.output_data_type, primitive=primitive, trace=trace))
            else:
                log(f'Skipping invalid QuestionSet answer #{i} which does not have a corresponding question: {cell}')
        return Data(answers, DataType.JSON_ARRAY)

    def strip_cot_last_line_prefix(self, text: str) -> str:
        text = text.strip().rstrip('.').lstrip('#').strip().lstrip('*').rstrip('*').strip() # ignore markdown bold/header
        prefixes = ['ANSWER', 'THEREFORE', 'THE ANSWER IS', 'THE CORRECT ANSWER IS']
        text = StringUtils.remove_prefixes(text, prefixes, case_sensitive=False)
        text = text.lstrip(',').strip().lstrip(':').strip()
        text = StringUtils.remove_prefixes(text, prefixes, case_sensitive=False)
        text = text.lstrip(',').strip().lstrip(':').strip()
        text = text.strip().lstrip('#').strip().lstrip('*').rstrip('*').strip()
        return text

    def parse_multiple_choice_answers(self, primitive: BasePrimitive, text: str) -> Data:
        text = self.strip_cot_last_line_prefix(text)
        text = text.lstrip('(').rstrip(')').strip()
        return Data( text[0], DataType.CHAR )

    def parse_multiselect_answers(self, primitive: BasePrimitive, text: str) -> Data:
        if not isinstance(primitive, MultiSelectQuestion):
            raise InvalidArgsError(f'parse_multiselect_answers expects a MultiSelectQuestion, got {primitive.id}')
        text = self.strip_cot_last_line_prefix(text.upper())
        text = text.lstrip('(').rstrip(')').strip()
        labels = [l.strip() for l in text.split(',') if len(l.strip()) > 0]
        letters = []
        allowed_letters = [chr(65 + i) for i in range(len(primitive.answer_options) + 1)]
        for l in labels:
            letter = l.split('.')[0].strip().lstrip('*').rstrip('*').lstrip('(').rstrip(')').strip() # Remove the answer label text if present
            if len(letter) == 1:
                letters.append(letter.upper())
        selected_letters = []
        for letter in letters:
            if letter in allowed_letters:
                if letter != allowed_letters[-1]:  # Remove the injected 'None of the above' answer from output so that the interaction can be Empty
                    selected_letters.append(letter)
            else:
                if self.error_policy == DataFormatErrorPolicy.RAISE_EXCEPTION:
                    raise BadPredictionError(f'Invalid answer label in {primitive.id}: {text} (expected: {allowed_letters})')
                else:
                    log(f'Postprocessor failed to parse multiselect answer label and returning empty result due to error_policy={self.error_policy}: {text} (expected: {allowed_letters})')
                    return Data.none()

        if len(selected_letters):
            return Data(selected_letters, DataType.MULTISELECT_ANSWERS)
        else:
            return Data.none()


    def extract_date(self, model: BaseModel, text_with_date: str) -> str:
        interaction = Interaction(Conversation.from_text(text_with_date), model)
        primitive = Instructions('Extract the date from the text above and format it as "Month DAY, YEAR"', custom_id='proton_date_extractor')
        primitive.build_prompt(interaction)
        self.scaler.run_single(interaction)
        prediction = interaction.output.to_text().strip()
        if StringUtils.is_date(prediction):
            return prediction
        else:
            log(f'Postprocessor failed to extract date from: {text_with_date[:100]}')
            return Default.NOT_FOUND_TAG

    def extract_number(self, model: BaseModel, text_with_number: str) -> str:
        if len(text_with_number) < 10:
            text_with_number = text_with_number.replace(',', '')
            text_with_number = f'The number is {text_with_number}'
        interaction = Interaction(Conversation.from_text(text_with_number), model)
        primitive = Instructions('<INSTRUCTIONS>Extract the numeric value from the text above and return the number without any additional comments.</INSTRUCTIONS>\nANSWER: ', custom_id='proton_number_extractor')
        primitive.build_prompt(interaction)
        self.scaler.run_single(interaction)
        prediction = interaction.output.to_text().strip()
        if StringUtils.is_numeric(prediction):
            return prediction
        else:
            log(f'Postprocessor failed to extract number from: {text_with_number[:100]}')
            return Default.NOT_FOUND_TAG