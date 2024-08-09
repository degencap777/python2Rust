import unittest
from unidecode import unidecode
from nltk.tokenize.punkt import PunktSentenceTokenizer
import os
from urllib.parse import urlparse
from pathlib import Path
from typing import Any
from enum import Enum
from datetime import datetime
from PIL import Image as PI
from dataclasses import dataclass

class Default:
    TEMPERATURE = 0
    TEMPERATURE_GEMINI = 0.36    # 0.36 is based on a Bayesian optimization for GeminiPro on document extraction tasks
    TOP_K = 40
    TOP_P = 0.97
    TOP_P_GEMINI = 1.0
    TOP_P_OPENAI = None         # official recommendation is to use either temperature or top_p
    MAX_OUTPUT_TOKENS = 2048    # tokens
    MAX_OUTPUT_TOKENS_GEMINI15 = 8192
    MAX_OUTPUT_TOKENS_ANTHROPIC = 4096
    MAX_OUTPUT_TOKENS_LLAMA_70B = 2048
    CHUNK_SIZE = 10           # characters
    CHUNK_OVERLAP = 0           # characters
    CHUNKER_MAX_SENTENCE = 1000 # characters
    FILE_ENCODING = 'utf-8' # used to be 'ISO-8859-1'
    GCP_LOCATION = 'us-central1'
    SPANNER_INSTANCE = 'proton'
    SPANNER_TIMEOUT = 300       # seconds, the slowest operation is index creation
    SPANNER_CACHE_DB = 'inference_cache'
    SPANNER_EVAL_DB = 'evaluations'
    SPANNER_WORKER_DB = 'worker'
    MODEL_PALM = 'text-bison-32k@002'
    MODEL_GECKO = 'textembedding-gecko@003'
    MODEL_GEMINI_TEXT = 'gemini-1.5-flash-preview-0514'
    MODEL_GEMINI_MULTIMODAL = 'gemini-1.5-flash-preview-0514'
    MODEL_OPENAI_TEXT = 'gpt-4-turbo-preview'
    MODEL_CLAUDE3_HAIKU = 'claude-3-haiku@20240307'
    MODEL_CLAUDE3_SONNET = 'claude-3-sonnet@20240229'
    MODEL_CLAUDE3_OPUS = 'claude-3-opus@20240229'
    MODEL_LLAMA_70B = 'llama-3-70b@001'
    MODEL_LLAMA_70B_IT = 'llama-3-70b-it@001'
    MODEL_SHORTDOC_PRIMARY = 'gemini-1.0-pro-001'
    MODEL_SHORTDOC_FALLBACK = 'gemini-1.5-flash-preview-0514'
    MISTRAL_MODEL = 'Mistral-7B-IT-01'
    GEMMA_MODEL = 'gemma-7b-it'
    MAX_INFERENCE_THREADS = 20
    INFERENCE_THREAD_TIMEOUT_SECONDS = 60*4  # Larger timeout to accommodate potential wait due to enter the thread pool
    INFERENCE_THREAD_MAX_TIMEOUT_RETRIES = 3
    NOT_FOUND_TAG = 'NOT_FOUND'
    SECTION_NOT_FOUND_TAG = 'NO_SECTION'  # Used in QuestionSet primitive to avoid NOT_FOUND_TAG which causes the entire inference to be filtered out.
    PROJECT_ID = os.getenv('PROJECT_ID', '')
    SEARCH_DATASTORE = os.getenv('SEARCH_DATASTORE', '')
    BUCKET_NAME = os.getenv('BUCKET_NAME', '')
    VS_LOCATION = 'global'      # vertex search
    VS_DEFAULT_CONFIG = 'default_config'
    VS_MAX_RESULTS = 10         # maximum = 100
    VS_NUM_SUMMARY_SOURCES = 5  # maximum = 5
    VS_NUM_EXTRACTIVE_ANSWERS = 1   # per document, maximum = 5
    VS_NUM_EXTRACTIVE_SEGMENTS = 5  # per document, maximum = 10
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_ORG_ID = os.getenv('OPENAI_ORG_ID')
    WORKER_DEFAULT_NAMESPACE = 'DEFAULT'
    WORKER_MAX_TASK_RETRIES = 3
    WORKER_POLLING_INTERVAL_SECONDS = 1.0
    API_PAGE_SIZE = 100
    SHORTDOC_PREAMBLE = 'You are a detail oriented AI assistant responsible for reading documents and answering questions about them.'
    SHORTDOC_CHUNK_SIZE = 3000

class DataType(Enum):
    """Multimodal inputs and outputs are expected to use these data types."""

    UNDEFINED = 1   # triggers automatic type detection
    TEXT = 2        # str
    INT = 3
    FLOAT = 4
    BOOL = 5
    CHAR = 6
    DATE = 7
    JSON_ARRAY = 8  # TODO: should we rename it to LIST_OF_STRINGS?
    JSON_DICT = 9   # TODO: should we rename it to DICT?
    IMAGE = 10      # PIL.Image
    MULTISELECT_ANSWERS = 11  # list of answer option labels like ['B', 'F']
    PDF = 12
    QUESTIONSET_ANSWERS = 13  # list of answers
    DATETIME = 14

    def __repr__(self):
        return repr(self.name)

    def to_string(self) -> str:
        return self.name

    @classmethod
    def from_string(cls, string: str) -> 'DataType':
        return cls[string]

def remove_line_breaks(text: str) -> str:
    return ' '.join(text.splitlines())

def truncate(text: str, max_length: int, ellipsis: str = '...', no_linebreaks: bool = False) -> str:
    """Truncates a string to a given length."""
    if no_linebreaks:
        text = remove_line_breaks(text)
    if len(text) > max_length:
        return text[:max_length - len(ellipsis)] + ellipsis
    else:
        return text

@dataclass
class InferenceMetadata():
    num_input_tokens: int = 0
    num_output_tokens: int = 0

    def __repr__(self) -> str:
        return f'num_input_tokens={self.num_input_tokens}, num_output_tokens={self.num_output_tokens}'

class ProtonException(Exception):
    def __init__(
        self,
        message: str | None = None,
    ):
        super(ProtonException, self).__init__(message)
  
class InvalidArgsError(ProtonException):
    """A proton method was called with invalid arguments."""

class ErrorCode(Enum):
    """Represents errors that Proton can workaround without throwing an exception."""

    UNDEFINED = 1
    RESPONSE_BLOCKED = 2  # Model prediction blocked by filters, can retry with a different model
    
class Data():
    """Unit of Input or Output data for multimodal inference."""

    def __init__(self, value: Any, data_type: DataType = DataType.UNDEFINED, id: str = ''):
        self.value = value
        self.id = id
        if data_type == DataType.UNDEFINED:
            self.data_type = Data.detect_data_type( value )
        else:
            self.data_type = data_type

    def __repr__(self):
        if self.data_type == DataType.IMAGE:
            width, height = self.value.size
            return f'{self.data_type.name}: [{width}x{height}]'
        elif self.data_type == DataType.PDF:
            return f'PDF: [{len(self.value)}]'
        elif self.data_type == DataType.TEXT:
            if not self.value:
                return f'{self.data_type.name}[0]: None'
            truncated_to_one_line = truncate(self.value, 100).replace('\n', ' ')
            return f'{self.data_type.name}[{len(self.value)}]: {truncated_to_one_line}'
        else:
            return f'{self.data_type.name}: {self.value}'
        
    def detect_data_type(value: Any) -> DataType:
        if isinstance(value, bool):  # bool is subclass of int so this must come first
            return DataType.BOOL
        if isinstance(value, int):
            return DataType.INT
        if isinstance(value, str):
            return DataType.TEXT
        if isinstance(value, float):
            return DataType.FLOAT
        if isinstance(value, datetime):
            return DataType.DATE
        if isinstance(value, PI.Image):
            return DataType.IMAGE
        if isinstance(value, list):
            if all([isinstance(item, str) and len(item) == 1 for item in value]):
                return DataType.MULTISELECT_ANSWERS
            else:
                return DataType.JSON_ARRAY
        if isinstance(value, dict):
            return DataType.JSON_DICT
        if isinstance(value, bytes):
            return DataType.PDF
        raise InvalidArgsError(f'Unknown data type: {type(value)}.')

class DataSourceType(Enum):
    DOCUMENT = 1
    IMAGE = 2
    DIRECTORY = 3
    
@dataclass
class DataSource():
    """Information about the source of data in Batches and Interactions."""
    source_type: DataSourceType = DataSourceType.DOCUMENT
    name: str = ''
    location: str = ''
    version: str = ''

    def __repr__(self) -> str:
        v = self.version or ''
        return f'{self.name}.{v}'

class DataBundle():
    """Ordered list of texts, images, videos intended for multimodal inference input and output."""

    def __init__(
        self,
        items: list[Data],
        error_code: ErrorCode = ErrorCode.UNDEFINED,
        data_source: DataSource | None = None
    ):
        self.items = items
        self.error_code = error_code
        self.inference_metadata: InferenceMetadata
        self.data_source = data_source         # optional metadata about the source of data for this bundle

    def __repr__(self):
        if self.is_empty():
            return 'EMPTY'
        if len(self.items) == 1:
            return f'Bundle[{self.items[0]}]'
        else:
            return f'Bundle[{len(self.items)} items starting with {self.items[0]}]'
    
    def is_empty(self) -> bool:
        if not self.items:
            return True
        for item in self.items:
            if item.value:
                return False
        return True
      
class TestStringMethods(unittest.TestCase):
    def fix_line_breaks(self, text: str) -> str:
        text = '\n'.join(text.splitlines())
        text = text.replace('\r', '')
        return text
    
    def fix_apostrophes(self, text: str) -> str:
        if not text:
            return text
        text = unidecode(text)
        return text.replace('\\u2019', "'")
    
    def normalize_special_characters(self, text: str) -> str:
        if not text:
            return text
        text = self.fix_line_breaks(text)
        return self.fix_apostrophes(text)
      
    def read_text_file(self, path_to_text_file: str, normalize_special_characters_flag: bool = True) -> str:
        with open(path_to_text_file, 'r') as file:
            text = file.read()
            if normalize_special_characters_flag:
                text = self.normalize_special_characters(text)
            return text
        
    def split_into_sentences(self, text: str) -> list[str]:
        '''Uses NLTK span_tokenize in order to preserve all white space and line breaks found between sentences.'''
        if not text:
            return []
        punkt = PunktSentenceTokenizer()
        spans = punkt.span_tokenize(text=text, realign_boundaries=True)
        sentences = []
        if not spans:
            return sentences
        previous_span = None
        for span in spans:
            if previous_span:
                sentences.append(text[previous_span[0]: span[0]])
            previous_span = span
        if previous_span:
            sentences.append(text[previous_span[0]:])
        return sentences
    
    def create_nonoverlapping_chunks(self, text: str, chunk_length_characters: int = Default.CHUNK_SIZE, trace: int = 0):
        '''Split text into sentences, while preserving all original white space and line breaks.'''
        sentences = self.split_into_sentences(text)
        chunks = []
        chunk = []
        chunk_size = 0
        max_chunk = 0
        for sentence in sentences:
            if len(sentence) > Default.CHUNKER_MAX_SENTENCE:
                print(f'Warning: long sentence ({len(sentence)} chars)', trace)
            chunk.append(sentence)
            chunk_size += len(sentence)
            if chunk_size >= chunk_length_characters:
                combined = ''.join(chunk)
                if len(combined) > max_chunk:
                    max_chunk = len(combined)
                chunks.append(combined)
                chunk = []
                chunk_size = 0
        if len(chunk):
            chunks.append(' '.join(chunk).strip())
        print(f'Created {len(chunks)} nonoverlapping chunks from {len(text)} characters, longest chunk={max_chunk}', trace)
        return chunks
        
    def _merge_chunks(self, chunks: list[str], start_index: int, num_chunks: int) -> str:
        group = []
        for i in range(num_chunks):
            if start_index + i < len(chunks):
                group.append(chunks[start_index + i])
        return '\n'.join(group)
    
    def parse_filename_with_extension_from_uri(self, uri: str) -> str:
        """Parses the file name from a URI."""
        parts = urlparse(uri)
        path = Path(parts.path)
        return path.stem + path.suffix

    def get_text_chunks(self, document_uri: str, document_text: str, chunk_length_characters: int = 1000, overlap_factor: int = 0, normalize_special_characters_flag: bool = True):
        file_name = self.parse_filename_with_extension_from_uri(document_uri)
        source = DataSource(DataSourceType.DOCUMENT, file_name, location=document_uri)
        if overlap_factor < 2:
            overlap_factor = 0
        if normalize_special_characters_flag:
            document_text = self.normalize_special_characters(document_text)
        if chunk_length_characters <= 0:
            return [DataBundle([Data(document_text, id='1')], data_source=source)]
        if overlap_factor == 0:
            chunk_strings = self.create_nonoverlapping_chunks(document_text, chunk_length_characters)
        else:
            small_chunks = self.create_nonoverlapping_chunks(document_text, round(chunk_length_characters / overlap_factor))
            chunk_strings = []
            for i in range(0, len(small_chunks), overlap_factor - 1):
                chunk_strings.append(self._merge_chunks(small_chunks, i, overlap_factor))
            print(f'Merged {len(small_chunks)} nonoverlapping chunks into {len(chunk_strings)} chunks with 1/{overlap_factor} overlap.')
        chunks = []
        for i, chunk_string in enumerate(chunk_strings):
            chunk = DataBundle([Data(chunk_string, id=str(i+1))], data_source=source)
            chunks.append(chunk)
        print(f'Created {len(chunks)} chunks for {document_uri}[{len(document_text)}] ({chunk_length_characters} chars, overlap={overlap_factor})')
        return chunks
    
    """Test Functions"""
    # def test_fix_line_breaks(self):
    #     goal = "First line.\nSecond line.\nThird line."
    #     temp = "First line.\nSecond line.\rThird line."
    #     result = self.fix_line_breaks(temp)
    #     self.assertEqual(result, goal)
        
    # def test_fix_apostrophes(self):
    #     goal = "Here's an example text with fancy apostrophes: It's time to test this. Don't worry about it! She said, It's great!"
    #     temp = goal.replace("'", "\\u2019")  # For testing purposes
    #     result = self.fix_apostrophes(temp)
    #     self.assertEqual(result, goal)
    
    # def test_normalize_special_characters(self):
    #     goal = "First 'line'.\nSecond 'line'.\nThird 'line'."
    #     temp = "First \\u2019line\\u2019.\nSecond \\u2019line\\u2019.\rThird \\u2019line\\u2019."
    #     result = self.normalize_special_characters(temp)
    #     self.assertEqual(result, goal)
    
    # def test_read_text_file(self):
    #     path = "books/book1.txt"
    #     goal = "First 'line'.\nSecond 'line'.\nThird 'line'."
    #     result = self.read_text_file(path)
    #     self.assertEqual(result, goal)
    
    # def test_split_into_sentences(self):
    #     goal = ["First 'line'.\n", "Second 'line'.\n", "Third 'line'."]
    #     temp = "First 'line'.\nSecond 'line'.\nThird 'line'."
    #     result = self.split_into_sentences(temp)
    #     self.assertEqual(result, goal)
    
    # def test_create_nonoverlapping_chunks(self):
    #     goal = ["First 'line'.\n", "Second 'line'.\n", "Third 'line'."]
    #     temp = "First 'line'.\nSecond 'line'.\nThird 'line'."
    #     result = self.create_nonoverlapping_chunks(temp, 10)
    #     self.assertEqual(result, goal)
    
    # def test_merge_chunks(self):
    #     goal = ["First 'line'.\nSecond 'line'.", "Second 'line'.\nThird 'line'.", "Third 'line'."]
    #     temp = ["First 'line'.", "Second 'line'.", "Third 'line'."]
    #     chunk_strings = []
    #     overlap_factor = 2
    #     for i in range(0, len(temp), overlap_factor - 1):
    #         chunk_strings.append(self._merge_chunks(temp, i, overlap_factor))
    #     self.assertEqual(chunk_strings, goal)
    
    # def test_parse_filename_with_extension_from_uri(self):
    #     goal = "book1.txt"
    #     temp = "book/book1.txt"
    #     result = self.parse_filename_with_extension_from_uri(temp)
    #     self.assertEqual(result, goal)
    
    def test_get_text_chunks(self):
        goal = "[Bundle[TEXT[14]: First 'line'. ], Bundle[TEXT[15]: Second 'line'. ], Bundle[TEXT[13]: Third 'line'.]]"
        temp = "First 'line'.\nSecond 'line'.\nThird 'line'."
        result = self.get_text_chunks('book1.txt', temp, 5)
        self.assertEqual(str(result), goal)

if __name__ == '__main__':
    unittest.main()