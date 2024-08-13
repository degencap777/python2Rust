from abc import ABCMeta, abstractmethod
from core.data_bundle import DataBundle
from core.conversation import Conversation
from models.model_config import ModelConfig
from utils.exceptions import InvalidArgsError

class BaseModel(object, metaclass=ABCMeta):
    """Abstract base class for all models."""
    model_config: ModelConfig

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def max_input_tokens(self) -> int:
        pass

    @property
    @abstractmethod
    def config(self) -> ModelConfig:
        pass

    @abstractmethod
    def predict(self, prompt: Conversation, config: ModelConfig) -> DataBundle:
        pass

    @abstractmethod
    def get_token_count(self, prompt: Conversation) -> int:
        pass

    def parse_model_name_version(self):
        model_name_with_version = self.model_name
        if '@' in model_name_with_version:
            model_version = model_name_with_version.split('@')[1]
            model_name = model_name_with_version.split('@')[0]
        elif 'gemini-' in model_name_with_version.lower():
            model_version = '-'.join(model_name_with_version.split('-')[1:])
            model_name = model_name_with_version.split('-')[0]
        elif model_name_with_version.lower().startswith('gpt-'):
            model_version = '-'.join(model_name_with_version.split('-')[1:])
            model_name = model_name_with_version.split('-')[0]
        elif 'gemma-' in model_name_with_version.lower():
            model_version = '-'.join(model_name_with_version.split('-')[1:])
            model_name = model_name_with_version.split('-')[0]
        else:
            raise InvalidArgsError(f'Model version cannot be parsed from model name: {model_name_with_version}')
        return model_name, model_version

    def split_text_into_chunks(self, text: str, max_chunk_length_tokens: int, percent_threshold: float = 2.0) -> list[str]:
        '''Fast way to split a long string into chunks with given maximum token length.'''
        if not text:
            return [text]
        if max_chunk_length_tokens < 1:
            raise InvalidArgsError('max_chunk_length_tokens must be a positive integer number')
        chunks = []
        while True:
            chunk = self.substring_with_token_length(text, max_chunk_length_tokens, percent_threshold)
            chunks.append(chunk)
            if len(chunk) >= len(text):
                break
            text = text[len(chunk):]
        return chunks

    def substring_with_token_length(self, text: str, max_tokens: int, percent_threshold: float = 2.0) -> str:
        '''Cut the string at max_tokens or within percent_threshold below max_tokens using binary search.'''
        if max_tokens < 1:
            raise InvalidArgsError(f'max_tokens must be a positive integer number')
        if percent_threshold < 0 or percent_threshold > 50:
            raise InvalidArgsError(f'percent_threshold must be between 0 and 50.')
        if not text or len(text) < max_tokens:
            return text
        total_tokens = self.get_token_count(Conversation.from_text(text))
        if total_tokens <= max_tokens:
            return text
        cutoff = round(max_tokens * (100-percent_threshold) / 100)
        start_index = max_tokens  # no need to probe below num_tokens because token >= character
        end_index = len(text) - 1
        while start_index <= end_index:
            mid_index = (start_index + end_index) // 2
            trimmed_text = text[:mid_index]
            num_tokens = self.get_token_count(Conversation.from_text(trimmed_text))
            if num_tokens >= cutoff and num_tokens <= max_tokens:
                last_index = trimmed_text.rfind('. ') + 1
                if last_index < len(trimmed_text) - 100:
                    last_index = trimmed_text.rfind(' ')
                if last_index < len(trimmed_text) - 100:
                    last_index = trimmed_text.rfind('\t')
                if last_index < len(trimmed_text) - 100:
                    last_index = trimmed_text.rfind('\n')
                if last_index < len(trimmed_text) - 100:
                    last_index = len(trimmed_text) - 1
                return trimmed_text[:last_index]
            elif num_tokens < cutoff:
                start_index = mid_index + 1
            else:
                end_index = mid_index - 1
        return text