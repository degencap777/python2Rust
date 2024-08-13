import traceback
from typing import Optional
import regex as re
from proton.config.defaults import Default
from proton.core.conversation import Conversation
from proton.core.data import Data
from proton.core.data_bundle import DataBundle
from proton.core.data_source import DataSource, DataSourceType
from proton.core.interaction import Interaction
from proton.core.batch import Batch
from proton.dataloader.data_types.base_document import PageSet
from proton.dataloader.engines.factory import DataLoaderEngineFactory
from proton.dataloader.extractors.base_extractor import BaseExtractor
from proton.dataloader.utils.aspose_engine_support import is_aspose_supported
from proton.utils.doc_parsing import extract_file_basename
from proton.utils.exceptions import DataLoaderExtractionTaskError
if is_aspose_supported(): # Aspose is not supported in Mac/Darwin
    from proton.dataloader.extractors.aspose_text_extractor import AsposeTextExtractor
from proton.dataloader.orchestration import DataLoaderOrchestrator
from proton.storage.gcs import GCS
from proton.storage.local_storage import LocalStorage
from proton.utils.numerals import NumberUtils
from proton.utils.strings import StringUtils
from proton.utils.sentence_splitter import SentenceSplitter
from proton.utils.logger import Trace, log

class TextChunker:
    def __init__(self):
        self.cache = None
        self.next_bullet_id = 1

    def _merge_chunks(self, chunks: list[str], start_index: int, num_chunks: int) -> str:
        group = []
        for i in range(num_chunks):
            if start_index + i < len(chunks):
                group.append(chunks[start_index + i])
        return '\n'.join(group)

    def create_nonoverlapping_chunks(self, text: str, chunk_length_characters: int = Default.CHUNK_SIZE, trace: int = 0):
        '''Split text into sentences, while preserving all original white space and line breaks.'''
        sentences = SentenceSplitter.split_into_sentences(text)
        chunks = []
        chunk = []
        chunk_size = 0
        max_chunk = 0
        for sentence in sentences:
            if len(sentence) > Default.CHUNKER_MAX_SENTENCE:
                log(f'Warning: long sentence ({len(sentence)} chars)', trace)
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
        log(f'Created {len(chunks)} nonoverlapping chunks from {len(text)} characters, longest chunk={max_chunk}', trace)
        return chunks

    # Inject unique IDs into all alphanumeric list item bullets: (A) -> (A.001), (B) -> (B.002), etc.
    def inject_list_bullet_ids(self, text: str):
        upper = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        lower = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        numeric = [str(i) for i in range(1, 31)]
        roman = [NumberUtils.int_to_roman(i) for i in range(1, 31)]
        target_labels = [f'({label})' for label in upper + lower + numeric + roman]
        lines = text.split('\n')
        lines_injected = []
        for line in lines:
            line = line.strip()
            if line.startswith('-'):  # TODO: consider removing page numbers via PDF parser
                line = re.sub(r'^\-[0-9]*\-\s', ' ', line, count=1).strip()
            if line.startswith('('):
                index = line.find(')')
                if index > 0:
                    label = line[: index + 1].replace(' ', '')
                    if label in target_labels:
                        line = (
                            label.replace(')', f'.{str(self.next_bullet_id).zfill(3)})')
                            + line[len(label) :]
                        )
                        self.next_bullet_id += 1
            lines_injected.append(line)
        return '\n'.join(lines_injected)

    def get_batch(self, document_uri: str, document_text: str, chunk_length_characters: int = 1000, overlap_factor: int = 0, normalize_special_characters: bool = True, trace: Trace = Trace.OFF) -> Batch:
        """Splits the document into chunks and wraps them into a Batch of interactions with Text inputs."""
        file_name = StringUtils.parse_filename_from_uri(document_uri)
        source = DataSource(DataSourceType.DOCUMENT, file_name.split('.')[0], location=document_uri)
        chunks = self.get_text_chunks(document_uri, document_text, chunk_length_characters, overlap_factor, normalize_special_characters, trace)
        interactions = [Interaction(Conversation.from_data_bundle(chunk), data_source=source) for chunk in chunks]
        return Batch(interactions, data_source=source)

    # Overlap is specified as 1/2, 1/3, 1/N and implemented by combining N non-overlapping chunks
    def get_text_chunks(self, document_uri: str, document_text: str, chunk_length_characters: int = 1000, overlap_factor: int = 0, normalize_special_characters: bool = True, trace: Trace = Trace.OFF) -> list[DataBundle]:
        """Splits the document into chunks with optional overlap.
        Args:
            document_uri: used only as a cache key
            document_text: full document text
            chunk_length_characters: minimum chunk length (actual chunks can be longer as the algorithm avoids splitting senteces.)
            overlap_factor: specifies overlap as a fraction of chunk size: 5 represents 20% overlap, 1 represents no overlap.
            normalize_special_characters: normalizes the style of line breaks and apostrophes
        """
        file_name = StringUtils.parse_filename_with_extension_from_uri(document_uri)
        source = DataSource(DataSourceType.DOCUMENT, file_name, location=document_uri)
        if overlap_factor < 2:
            overlap_factor = 0
        if normalize_special_characters:
            document_text = StringUtils.normalize_special_characters(document_text)
        if chunk_length_characters <= 0:
            return [DataBundle([Data(document_text, id='1')], data_source=source)]
        if self.cache:
            cached: list[Data] = self.cache.get(document_uri, chunk_length_characters, overlap_factor)
            if cached:
                log(f'Using {len(cached)} cached [{chunk_length_characters}] chunks for {document_uri}')
                return cached
        if overlap_factor == 0:
            chunk_strings = self.create_nonoverlapping_chunks(document_text, chunk_length_characters)
        else:
            small_chunks = self.create_nonoverlapping_chunks(document_text, round(chunk_length_characters / overlap_factor))
            chunk_strings = []
            for i in range(0, len(small_chunks), overlap_factor - 1):
                chunk_strings.append(self._merge_chunks(small_chunks, i, overlap_factor))
            log(f'Merged {len(small_chunks)} nonoverlapping chunks into {len(chunk_strings)} chunks with 1/{overlap_factor} overlap.')
        chunks = []
        for i, chunk_string in enumerate(chunk_strings):
            chunk = DataBundle([Data(chunk_string, id=str(i+1))], data_source=source)
            chunks.append(chunk)
        if self.cache:
            self.cache.set(document_uri, chunk_length_characters, overlap_factor, chunks)
        log(f'Created {len(chunks)} chunks for {document_uri}[{len(document_text)}] ({chunk_length_characters} chars, overlap={overlap_factor})', trace)
        return chunks

    def wrap_pages_into_section_tags(self, pages: list[str]) -> dict[str, str]:
        sections_by_id: dict[str, str] = {}
        for i, page in enumerate(pages):
            id = f'SECTION-{i+1}'
            sections_by_id[id] = f'<{id}>\n{page}\n</{id}>'
        return sections_by_id

    def split_into_sections(self, document_uri: str, document_text: str, chunk_length_characters: int = 1000, trace: Trace = Trace.OFF) -> dict[str, str]:
        '''Inserts tags like <SECTION-123>...</SECTION-123> so that LLM can identify relevant chunks.
        Args:
            document_uri: used only as a cache key
            document_text: full document text
            chunk_length_characters: minimum chunk length (actual chunks can be longer as the algorithm avoids splitting senteces.)
            overlap_factor: specifies overlap as a fraction of chunk size: 5 represents 20% overlap, 1 represents no overlap.
            normalize_special_characters: normalizes the style of line breaks and apostrophes
        '''
        chunks = self.get_text_chunks(
            document_uri=document_uri,
            document_text=document_text,
            chunk_length_characters=chunk_length_characters,
            overlap_factor=0,
            normalize_special_characters=False,
            trace=trace)
        log(f'in split_into_sections with {len(document_text)} characters, {len(chunks)} chunks.', trace)
        sections_by_id: dict[str, str] = {}
        for i, chunk in enumerate(chunks):
            id = f'SECTION-{i+1}'
            sections_by_id[id] = f'<{id}>\n{chunk.to_text()}\n</{id}>'
        return sections_by_id

    def get_pages(self, document_uri: str, extractor: Optional[BaseExtractor] = None, trace: int = Trace.OFF) -> list[tuple[int, str]]:
        """Get page text from a PDF document URI, with optional extraction and tracing."""

        if not extractor:
            extractor = DataLoaderEngineFactory().get_extractor(extractor_type='text')
        log(f'Extracting pages from {document_uri} using {extractor.__class__.__name__}', trace)
        loader = DataLoaderOrchestrator.init(initial_extractors={'text': extractor}, trace=trace)
        results = loader.extract_data_from_source(document_uri, trace=trace)
        try:
            document_key = extract_file_basename(GCS.get_blob_name_from_uri(document_uri))
            result = results[document_key]['text'][0].result

            # Type Check and Handling:
            if isinstance(result, PageSet):
                return result.to_list_with_page_nums()
            elif result is None:  # Handle None case separately
                raise DataLoaderExtractionTaskError(f'No pages extracted for document: {document_uri}')
            else:
                raise DataLoaderExtractionTaskError(f'Extracted result is not a PageSet. Type: {type(result)}')
        except Exception as e:
            err_trace = traceback.format_exc()
            err_type = type(e).__name__
            raise DataLoaderExtractionTaskError(f'Unable to extract pages for document: {document_uri}, {err_type}: {err_trace}') from e  # Chain the original exception
        finally:
            loader.cleanup()


if __name__ == '__main__':
    document = 'Santa claus is coming to town. ' * 10
    batch = TextChunker().split_into_sections('test.txt', document, chunk_length_characters=100, trace=Trace.ON)
    print(batch)

    text = LocalStorage().read_text_file('examples/books/book1.txt')
    chunks = TextChunker().get_text_chunks('book1.txt', text, 500)
    for chunk in chunks[:3]:
        print('=' * 100)
        print(chunk)

    # file_path = 'gs://test_proton/deal_docs/pdf/VENT_45.pdf'
    # extractor = DataLoaderEngineFactory().get_extractor(extractor_type='text')
    # pages = TextChunker().get_pages(file_path, extractor=extractor, trace=Trace.ON)
    # print(pages[:1])