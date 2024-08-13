import google.cloud.discoveryengine_v1alpha as discoveryengine
import json
from google.cloud.discoveryengine_v1alpha import SearchRequest
from google.cloud.discoveryengine_v1alpha.services.search_service.pagers import SearchPager
from google.cloud import aiplatform
from config.defaults import Default
from core.data_source import DataSource
from core.search_results  import SearchResults, DocumentResult, DocumentExtract
from utils.exceptions import InvalidApiResponseError
from utils.exceptions import InvalidArgsError
from utils.strings import StringUtils
from utils.logger import Trace, log


class Search:
    """Retrieve information from documents located in the given Vertex AI Search data store."""

    def __init__(self,
        data_store_id: str,
        project_id: str = Default.PROJECT_ID,
        vertex_search_location: str | None = None,
        serving_config_id: str | None = None,
        trace: Trace = Trace.OFF
    ):
        if not project_id:
            raise InvalidArgsError('project_id is required.')
        if not data_store_id:
            raise InvalidArgsError('data_store_id is required.')
        log(f'Initializing Search with data_store_id={data_store_id} in project={project_id}', trace)
        aiplatform.init(project=project_id)
        self.client = discoveryengine.SearchServiceClient()
        self.serving_config = self.client.serving_config_path(
            project=project_id,
            location=vertex_search_location or Default.VS_LOCATION,
            data_store=data_store_id,
            serving_config=serving_config_id or Default.VS_DEFAULT_CONFIG
        )

    def search(self,
            query: str,
            filter: str = '',   # works only with structured data stores
            scope: DataSource | None = None, # adds filter = uri: ANY("scope.location")
            max_results: int = Default.VS_MAX_RESULTS,
            snippets: bool = True,
            query_expansion: bool = False,
            max_answers_per_result: int = Default.VS_NUM_EXTRACTIVE_ANSWERS,
            max_segments_per_result: int = Default.VS_NUM_EXTRACTIVE_SEGMENTS,
            trace: Trace = Trace.OFF
        ) -> SearchResults:
        """Returns a list of documents matching the given query, including snippets, extractive answers, and summary."""
        if filter and scope:
            raise InvalidArgsError(f'Parameters filter and scope are mutually exclusive.')
        if scope and scope.location:
            filter = f'uri: ANY("{scope.location}")'
        log(f'Searching with filter={filter}, max_results={max_results}, query={query}', trace)
        request = SearchRequest(
            query = query,
            filter = filter,
            page_size = max_results,
            serving_config = self.serving_config,
            content_search_spec = SearchRequest.ContentSearchSpec(
                snippet_spec = SearchRequest.ContentSearchSpec.SnippetSpec(
                    return_snippet = snippets
                ),
                extractive_content_spec = SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                    max_extractive_answer_count = max_answers_per_result,
                    max_extractive_segment_count = max_segments_per_result,
                    return_extractive_segment_score = True,
                    num_previous_segments = 1,
                    num_next_segments = 1
                ),
                summary_spec = SearchRequest.ContentSearchSpec.SummarySpec(
                    summary_result_count = Default.VS_NUM_SUMMARY_SOURCES,
                    include_citations = True,
                    ignore_adversarial_query = False,
                    ignore_non_summary_seeking_query = False
                )
            ),
            query_expansion_spec = SearchRequest.QueryExpansionSpec(
                condition = SearchRequest.QueryExpansionSpec.Condition.AUTO if query_expansion else SearchRequest.QueryExpansionSpec.Condition.DISABLED
            ),
            spell_correction_spec = SearchRequest.SpellCorrectionSpec(
                mode = SearchRequest.SpellCorrectionSpec.Mode.SUGGESTION_ONLY
            )
        )
        response = self.client.search(request)
        results = self._parse_results(query, response, trace)
        return results

    # TODO: switch from sequential to concurrent queries.
    def search_batch(
        self,
        queries: list[str],
        filter: str = '',   # works only with structured data stores
        scope: DataSource | None = None, # adds filter = uri: ANY("scope.location")
        max_results: int = Default.VS_MAX_RESULTS,
        snippets: bool = True,
        query_expansion: bool = False,
        max_answers_per_result: int = Default.VS_NUM_EXTRACTIVE_ANSWERS,
        max_segments_per_result: int = Default.VS_NUM_EXTRACTIVE_SEGMENTS,
        trace: Trace = Trace.OFF
    ) -> SearchResults:
        """Runs multiple search queries and combines their results."""
        all_queries = []
        all_summaries = []
        all_documents = []
        for query in queries:
            if not query:
                raise InvalidArgsError(f'Query cannot be empty.')
            results = self.search(query, filter, scope, max_results, snippets, query_expansion, max_answers_per_result, max_segments_per_result, trace)
            all_queries.append(query)
            all_summaries.append(results.summary)
            all_documents.extend(results.documents)
        return SearchResults('\n'.join(all_queries), '\n'.join(all_summaries), all_documents)

    def _parse_results(self, query: str, response: SearchPager, trace: Trace = Trace.OFF) -> SearchResults:
        if trace == Trace.VERBOSE:
            log(f'Attribution token: {response.attribution_token}')
        results = SearchResults(query, response.summary.summary_text or '')
        for rank, item in enumerate(response.results):
            document = dict(item.document.derived_struct_data)
            if not 'link' in document:
                raise InvalidApiResponseError(f'Search result is missing expected field link: {document}')
            doc = DocumentResult(item.document.id, rank+1, document['link'])
            if 'snippets' in document:
                doc.snippets = [dict(s)['snippet'] for s in document['snippets'] if s['snippet_status'] == 'SUCCESS']
            if 'extractive_answers' in document:
                answers = [dict(answer) for answer in document['extractive_answers']]
                for a in answers:
                    page_number = int(a['pageNumber']) if 'pageNumber' in a else -1
                    doc.answers.append(DocumentExtract(page_number, a['content']))
            if 'extractive_segments' in document:
                segments = [dict(segment) for segment in document['extractive_segments']]
                for s in segments:
                    page_number = int(s['pageNumber']) if 'pageNumber' in s else -1
                    doc.segments.append(DocumentExtract(page_number, s['content']))
            results.append(doc)
        if trace > 0:
            results.trace(trace)
        return results

    @staticmethod
    def generate_vertex_metadata_file(document_uris: list[str], mime_type: str = 'application/pdf') -> str:
        """Returns JSONL required by Vertex Search Datastores for unstructured documents with metadata.

        This type of data stores enables search queries with optional 'filter' parameter that can reduce the scope to a single document by name.
        """
        lines = []
        for i, uri in enumerate(document_uris):
            line = {
                'id':  f'doc-{i+1}',
                'structData': {
                    'filename': StringUtils.parse_filename_from_uri(uri),
                    'uri': uri
                },
                'content': {
                    'mimeType': mime_type,
                    'uri': uri
                }
            }
            lines.append(json.dumps(line))
        return '\n'.join(lines)
