from unidecode import unidecode
from urllib.parse import urlparse
from pathlib import Path
from nltk.tokenize.punkt import PunktSentenceTokenizer
from python_src.utils.data_source import DataSource, DataSourceType
from python_src.utils.data_bundle import DataBundle
from python_src.utils.data import Data
from python_src.utils.defaults import Default
  
class TextChunker:
    def __init__(self) -> None:
        pass

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

    def read_text_file(self, path_to_text_file: str, normalize_special_characters_flag: bool = True) -> str:
        with open(path_to_text_file, 'r') as file:
            text = file.read()
            if normalize_special_characters_flag:
                text = self.normalize_special_characters(text)
            return text
            
    def normalize_special_characters(self, text: str) -> str:
            if not text:
                return text
            text = self.fix_line_breaks(text)
            return self.fix_apostrophes(text)

    def fix_line_breaks(self, text: str) -> str:
            text = '\n'.join(text.splitlines())
            text = text.replace('\r', '')
            return text

    def fix_apostrophes(self, text: str) -> str:
            if not text:
                return text
            text = unidecode(text)
            return text.replace('\\u2019', "'")
    
    def parse_filename_with_extension_from_uri(self, uri: str) -> str:
        """Parses the file name from a URI."""
        parts = urlparse(uri)
        path = Path(parts.path)
        return path.stem + path.suffix
         
if __name__ == '__main__':
    document = 'Santa claus is coming to town. ' * 10
    
    text = TextChunker().read_text_file('../../books/book1.txt')
    chunks = TextChunker().get_text_chunks('book1.txt', text, 5)
    for chunk in chunks:
        print('=' * 100)
        print(chunk)