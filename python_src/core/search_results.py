from dataclasses import dataclass
from core.data_bundle import DataBundle
from core.interaction import Interaction
from utils.strings import StringUtils
from utils.exceptions import InvalidArgsError
from utils.logger import log, Trace

@dataclass
class DocumentExtract():
    page: int  # set to -1 whenever search returns empty pageNumber
    text: str
    def __repr__(self):
        return f'[{len(self.text)} chars, P{self.page}]: {StringUtils.truncate(self.text, 90, no_linebreaks=True)}'

class DocumentResult():
    """Represents one document in search results."""
    def __init__(self, id: str, rank: int, link: str):
        self.id = id
        self.rank = rank
        self.link = link
        self.snippets: list[str] = []
        self.segments: list[DocumentExtract] = []
        self.answers: list[DocumentExtract] = []

    def __repr__(self):
        if self.link:
            return f'{self.rank}: {self.link}'
        elif self.segments:
            return f'{self.rank}: {len(self.segments)} segments'
        elif self.snippets:
            return f'{self.rank}: {len(self.snippets)} snippets'
        else:
            return f'{self.rank}: empty result'

class SearchResults():
    """Container for search results."""
    def __init__(self, query: str, summary: str='', documents: list[DocumentResult] | None = None):
        self.query = query
        self.summary = summary
        self.documents = documents or []

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return self.documents[index]

    def __iter__(self):
        return iter(self.documents)

    def __repr__(self):
        return f'{len(self.documents)} results for query: {self.query}'

    def is_empty(self) -> bool:
        return len(self.documents) == 0

    def append(self, doc: DocumentResult):
        if not isinstance(doc, DocumentResult):
            raise InvalidArgsError(f'Only DocumentResult objects can be appended to SearchResults. Received: {type(doc)}.')
        self.documents.append(doc)

    def to_link_bundle(self) -> DataBundle:
        """Creates a DataBundle with all document links found in search results."""
        links = [doc.link for doc in self.documents]
        return DataBundle.from_list_of_strings(links)

    def combine(self) -> DataBundle:
        """Creates a DataBundle with all relevant summaries, answers, and segments found in search results."""
        texts = []
        if self.summary:
            texts.append(self.summary)
        for doc in self.documents:
            texts.append('\n\n'.join([a.text for a in doc.answers]))
            texts.append('\n\n'.join([s.text for s in doc.segments]))
        return DataBundle.from_text('\n\n'.join(texts))

    def to_interaction(self) -> Interaction:
        return Interaction.from_data_bundle(self.combine())

    def trace(self, trace: Trace = Trace.OFF) -> None:
        log(f'QUERY: {self.query} (trace={trace})')
        if self.summary:
            content = self.summary if trace == Trace.VERBOSE else StringUtils.truncate(self.summary, 100)
            log(f'Summary: {content}')
        for result in self.documents:
            log(f"{result.rank}: {result.link}")
            if result.snippets:
                log('  Snippets:')
                for snippet in result.snippets:
                    content = snippet if trace == Trace.VERBOSE else StringUtils.truncate(snippet, 100)
                    log(f'    {content}')
            if result.answers:
                log('  Answers:')
                for answer in result.answers:
                    content = answer.text if trace == Trace.VERBOSE else StringUtils.truncate(str(answer), 100)
                    log(f'    {content}')
            if result.segments:
                log('  Segments:')
                for segment in result.segments:
                    if trace == Trace.VERBOSE:
                        page_number = f'FROM PAGE {segment.page}:' if segment.page > 0 else ''
                        log(f'SEGMENT {page_number}---------------------------------------------------------\n{segment.text}')
                    else:
                        log(f'    {StringUtils.truncate(str(segment), 100)}')
