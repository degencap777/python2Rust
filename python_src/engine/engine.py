from dataclasses import dataclass
import json
import random
from datetime import datetime

from models.base_model import BaseModel
from core.batch import Batch
from core.conversation import Conversation
from core.data import Data
from core.data_source import DataSource
from core.data_type import DataType
from core.output_format import Format
from core.data_bundle import DataBundle
from core.interaction import Interaction
from config.defaults import Default
from core.search_results import SearchResults
from engine.batch_execution import BatchExecution
from engine.postprocessor import Postprocessor, DataFormatErrorPolicy
from engine.scaler import Scaler
from engine.search import Search
from engine.text_similarity import TextSimilarity
from loader.text_chunker import TextChunker
from primitives.base_primitive import BasePrimitive
from primitives.multiple_choice_question import MultipleChoiceQuestion
from primitives.multi_select_question import MultiSelectQuestion
from primitives.instructions import Instructions
from storage.base_file_store import BaseFileStore
from utils.exceptions import InvalidArgsError, BadPredictionError
from utils.timer import Timer
from utils.strings import StringUtils
from utils.logger import Trace, log

@dataclass
class EngineFlags():
    tracking_enabled: bool = False
    shortdocs_enabled: bool = False

class Engine:
    """Inference orchestration engine responsible for running Tasks and Primitives on Batches of Interactions."""

    def __init__(self,
            project_id: str,
            storage: BaseFileStore,
            scaler: Scaler,
            models: list[BaseModel],
            search_engine: Search | None = None,
            default_model_lb_pool: dict[BaseModel, float] | None = None,  # optional pool of default models initialized in different projects for load balancing
            postprocessor_error_policy: DataFormatErrorPolicy = DataFormatErrorPolicy.RAISE_EXCEPTION,
            trace: Trace = Trace.OFF
        ):
        if not models:
            raise InvalidArgsError('At least one model is required.')
        if not project_id:
            raise InvalidArgsError('project_id must not be empty')
        self.project_id = project_id
        self.storage = storage
        self.models = models
        self.default_model_lb_pool = default_model_lb_pool
        self.embeddings_model = None
        self._text_similarity = None
        self.scaler = scaler
        self._search_engine = search_engine
        self.postprocessor = Postprocessor(scaler, postprocessor_error_policy)
        self.history: list[BatchExecution] = []
        self.flags = EngineFlags()
        self.default_trace = trace
        log(f'Proton engine is ready', trace)

    def start_tracking(self):
        self.history = []
        self.flags.tracking_enabled = True

    def enable_token_stats(self):
        for model in self.models:
            if isinstance(model, Palm):
                model.enable_token_stats()

    def stop_tracking(self) -> list[BatchExecution]:
        self.flags.tracking_enabled = False
        return self.history

    def __repr__(self) -> str:
        return f'Proton is running with default model: {self.default_model().model_name}'

    def default_model(self) -> BaseModel:
        """Returns the default model and throws exception if it is not configured."""
        if not self.models:
            raise InvalidArgsError('No models are configured.')
        return self.models[0]

    def set_embeddings_model(self, model: BaseModel):
        self.embeddings_model = model
        self._text_similarity = TextSimilarity(self.scaler, self.embeddings_model)

    def disable_cache(self):
        self.scaler.inference_cache = None

    def disable_read_cache(self):
        """This amounts to overwriting previously cached inferences. This feature is needed for internal Proton experimentation, and not intended for end users."""
        if self.scaler.inference_cache:
            self.scaler.inference_cache._overwrite_cache = True

    def run( self, primitive: BasePrimitive, context: Interaction | None = None, force_model: BaseModel | None = None, trace: Trace = Trace.OFF) -> Interaction:
        """Run the Primitive on a single input context."""
        if context:
            if not isinstance(context, Interaction):
                raise InvalidArgsError(f'This method requires an Interaction object, but received {type(context)}.')
        results = self.run_batch(primitive, Batch([context or Interaction.empty()]), force_model, trace)
        return results.first()

    def _load_balanced_models(self) -> list[BaseModel]:
        """Use the QPM quota as the weight for selecting the next default model."""
        if not self.default_model_lb_pool:
            return self.models
        pool = list(self.default_model_lb_pool.keys())
        weights = [self.default_model_lb_pool[model] for model in pool]
        results = random.choices(pool, weights=weights, k=1)  # list with one random model from the pool
        if len(self.models) > 1:
            results.extend(self.models[1:])
        return results

    def run_batch(self, primitive: BasePrimitive, batch: Batch, force_model: BaseModel | None = None, trace: Trace = Trace.OFF) -> Batch:
        """Executes the given Primitive on a batch of Interaction objects, in parallel."""
        if not isinstance(primitive, BasePrimitive):
            raise InvalidArgsError(f'run_batch requires a BasePrimitive object, but received {type(primitive)}.')
        if not isinstance(batch, Batch):
            raise InvalidArgsError(f'run_batch requires a Batch object, but received {type(batch)}.')
        timer = Timer()
        if batch.is_empty():
            return Batch([])
        log(f'[{primitive.id}] -> batch of {len(batch)}', trace)
        for interaction in batch:
            primitive.build_prompt(interaction)
            interaction.primitive = primitive
            interaction.is_output_json = primitive.is_output_json()
            if force_model:
                interaction.models = [force_model]
            else:
                interaction.models = self._load_balanced_models()
        num_non_empty = self._run_primitive(primitive, batch, trace)
        if self.default_trace > 0:
            if num_non_empty < len(batch):
                log(f'{len(batch)} inferences in {timer.seconds:.1f} seconds ({num_non_empty} non-empty)')
            elif len(batch) > 1:
                log(f'{len(batch)} inferences in {timer.seconds:.1f} seconds')
            else:
                log(f'1 inference in {timer.seconds:.1f} seconds', trace)
        if isinstance(primitive, MultiSelectQuestion) and primitive.verify_answers:
            self.verify_multiselect_answers(primitive, batch, trace)
        return batch

    def _run_primitive(self, primitive: BasePrimitive, batch: Batch, trace: Trace = Trace.OFF) -> int:
        timer = Timer()
        tracker = BatchExecution(primitive, batch, datetime.now()) if self.flags.tracking_enabled else None
        self.configure_prompt_tracing(batch, trace)
        self.scaler.run_batch(batch.interactions)
        num_non_empty = self.format_outputs(primitive, batch, trace)
        if tracker:
            tracker.duration_seconds = timer.seconds
            self.history.append(tracker)
        return num_non_empty

    def map_reduce(self, primitive: BasePrimitive, batch: Batch, force_model: BaseModel | None = None, trace: Trace = Trace.OFF) -> Interaction:
        """Runs the Primitive on a batch of text inputs, combines all non-empty results and runs the same primitive on the combined chunk."""
        if not isinstance(primitive, BasePrimitive):
            raise InvalidArgsError(f'map_reduce requires a BasePrimitive object, but received {type(primitive)}.')
        if not isinstance(batch, Batch):
            raise InvalidArgsError(f'map_reduce requires a Batch object, but received {type(batch)}.')
        if batch.is_empty():
            return Interaction.empty()
        batch = self.run_batch(primitive, batch, force_model, trace).non_empty()
        if batch.is_empty():
            log(f'map_reduce got empty results from the first pass.', trace)
            return Interaction.empty()
        if batch.is_single():
            return batch.first()
        combined_bundle = batch.combine_interaction_contexts(defrag_text=True)
        conversation = Conversation.from_data_bundle(combined_bundle)
        model = batch.default_model()
        max_tokens = model.max_input_tokens
        if combined_bundle.is_text() and (len(combined_bundle.to_text()) > max_tokens):
            token_count = model.get_token_count(conversation)
            if token_count > max_tokens:
                if primitive.output_type == DataType.TEXT: # Autorecover by combining outputs
                    log(f'map_reduce uses combined outputs from {len(batch)} interactions because combined inputs are {token_count} tokens.')
                    combined_bundle = batch.combine_outputs(defrag_text=True)
                    conversation = Conversation.from_data_bundle(combined_bundle)
                    token_count = model.get_token_count(conversation)
                    if token_count > max_tokens:
                        log(f'map_reduce truncates combined outputs because they exceed the input token limit ({token_count} > {max_tokens})')
                        combined_bundle.first().value = combined_bundle.first().value[:max_tokens * 2]
                else:
                    log(f'map_reduce is impossible because combined inputs are too large and output_type={primitive.output_type}.')
                    return batch.first()
        log(f'map_reduce combined {len(batch)} outputs into {combined_bundle.hash()}: {combined_bundle}', trace)
        return self.run(primitive, Interaction(conversation, batch.models(), trace))

    def format_outputs(self, primitive: BasePrimitive, batch: Batch, trace: Trace = Trace.OFF) -> int:
        """Convert the output DataBundle from TEXT to primitive.output_type."""
        num_non_empty = 0
        for i, interaction in enumerate(batch):
            if trace == Trace.VERBOSE:
                log(f'output #{i+1}={interaction.output.to_text()}')
            if primitive.output_format == Format.ANSWER_ON_LAST_LINE:
                if interaction.output.is_text() and interaction.output.is_single():
                    lines = interaction.output.to_text().strip().splitlines()
                    if len(lines) > 1:
                        last_line_without_cot_prefix = self.postprocessor.strip_cot_last_line_prefix( lines[-1])
                        interaction.output = DataBundle.from_text( last_line_without_cot_prefix )
                        log(f'Removed the first {len(lines) - 1} output lines due to output_format={primitive.output_format}', trace)
                        if trace > 1:
                            log(f'COT: {lines}')
                            log(f'LAST_LINE: {lines[-1]}')

            if interaction.is_output_empty():
                if trace == Trace.VERBOSE and len(interaction.output.to_text()) > len('NOT_FOUND'):
                    log(f'format_outputs detected "NOT_FOUND" in model output.')
                interaction.output = DataBundle([])
            else:
                data = self.postprocessor.convert_text(
                    interaction.models[0],
                    interaction.output.to_text(),
                    primitive.output_type,
                    primitive,
                    trace=interaction.trace if interaction.trace > 0 else trace
                )
                if data.is_empty():
                    interaction.output = DataBundle([])
                else:
                    interaction.output = DataBundle([data])
                    num_non_empty += 1
        return num_non_empty

    # Ask a True/False multiple choice question to confirm each positive multi-select answer.
    def verify_multiselect_answers(self, question: MultiSelectQuestion, ms_batch: Batch, trace: Trace = Trace.OFF):
        timer = Timer()
        mc_batch = Batch([])
        confirmation = MultipleChoiceQuestion(
            'Is it true that {statement}?',
            ['True', 'False', 'The CONTEXT above does not include a clear answer to this question.'], custom_id='proton_verify_multiselect'
        )
        for answer in ms_batch:
            if answer.output.is_empty():
                continue
            for letter in answer.output.first().value:
                if question.is_empty_answer_label(letter):  # Not verifying empty answers for now.
                    continue
                selected_answer = question.get_answer_by_label(letter).rstrip('.')
                selected_answer = selected_answer[0].lower() + selected_answer[1:]
                interaction = Interaction(answer.context, answer.models, trace)
                confirmation.build_prompt(interaction, statement = selected_answer)
                interaction.parent_interaction = answer
                interaction.tag = letter  # temporary using Tag to store the answer that may need to be removed.
                mc_batch.append(interaction)
        if mc_batch.is_empty():
            return
        log(f'Verifying {len(mc_batch)} multi-select answers')
        self._run_primitive(confirmation, mc_batch, trace)
        num_corrected_answers = 0
        for interaction in mc_batch:
            if interaction.output.to_text() != 'A':  # The first answer option is 'True' so this answer failed verificaiton.
                ms_answer_letters = interaction.parent_interaction.output.first().value
                ms_answer_letters.remove(interaction.tag)
                if len(ms_answer_letters) == 0:
                    interaction.parent_interaction.output.reset(Data.none())
                else:
                    interaction.parent_interaction.output.first().value = ms_answer_letters
                num_corrected_answers += 1
        log(f'{len(mc_batch)} verifications -> {num_corrected_answers} corrected answers in {timer.seconds:.1f} seconds')

    def text_similarity_max(self, query: str, paragraphs: list[str], max_length_chars: int = 0, trace: Trace = Trace.OFF) -> float:
        if not self._text_similarity:
            raise InvalidArgsError('embeddings_model must be configured in order to use text similarity.')
        max_score, avg_score = self._text_similarity.score(query, paragraphs, max_length_chars, trace)
        return max_score

    def text_similarity_avg(self, query: str, paragraphs: list[str], max_length_chars: int = 0, trace: Trace = Trace.OFF) -> float:
        if not self._text_similarity:
            raise InvalidArgsError('embeddings_model must be configured in order to use text similarity.')
        max_score, avg_score = self._text_similarity.score(query, paragraphs, max_length_chars, trace)
        return avg_score

    def configure_prompt_tracing( self, batch: Batch, trace: Trace = Trace.OFF) -> None:
        if trace > 0:
            if trace > 1:
                batch.trace_all_prompts()
            else:
                batch.trace_first_prompt()

    def search(self,
        query: str,
        filter: str = '', # works only with structured data stores
        scope: DataSource | None = None, # adds filter = uri: ANY("scope.location")
        max_results: int = Default.VS_MAX_RESULTS,
        snippets: bool = True,
        query_expansion: bool = False,
        max_answers_per_result: int = Default.VS_NUM_EXTRACTIVE_ANSWERS,
        max_segments_per_result: int = Default.VS_NUM_EXTRACTIVE_SEGMENTS,
        trace: Trace = Trace.OFF
    ) -> SearchResults:
        """Returns a list of documents matching the given query, including snippets, extractive answers, and summary."""
        if not self._search_engine:
            raise InvalidArgsError('search_engine is not intitialized.')
        return self._search_engine.search(
            query,
            filter,
            scope,
            max_results,
            snippets,
            query_expansion,
            max_answers_per_result,
            max_segments_per_result,
            trace
        )

    def search_batch(self,
        queries: list[str],
        filter: str = '', # works only with structured data stores
        scope: DataSource | None = None, # adds filter = uri: ANY("scope.location")
        max_results: int = Default.VS_MAX_RESULTS,
        snippets: bool = True,
        query_expansion: bool = False,
        max_answers_per_result: int = Default.VS_NUM_EXTRACTIVE_ANSWERS,
        max_segments_per_result: int = Default.VS_NUM_EXTRACTIVE_SEGMENTS,
        trace: Trace = Trace.OFF
    ) -> SearchResults:
        """Returns a list of documents matching the given query, including snippets, extractive answers, and summary."""
        if not self._search_engine:
            raise InvalidArgsError('search_engine is not intitialized.')
        return self._search_engine.search_batch(
            queries,
            filter,
            scope,
            max_results,
            snippets,
            query_expansion,
            max_answers_per_result,
            max_segments_per_result,
            trace
        )

    def extract_page_by_number(self, document: str, page_number: int, page_numbers_in_footer: bool = True, trace: Trace = Trace.OFF) -> str:
        '''Returns the full text of the given page.'''
        if not page_number:
            raise InvalidArgsError(f'page_number must be a number or roman numeral.')
        header_or_footer = 'bottom' if page_numbers_in_footer else 'the top'
        prompt = Instructions(f'''
**DOCUMENT**
{document}

**INSTRUCTIONS**
The page numbers in the DOCUMENT above are located on {header_or_footer} of every page.
Find page number {page_number} and copy its full text to the ANSWER:

**ANSWER**
''')
        result = self.run(prompt, context=None, trace=trace).output
        if result.is_empty():
            return ''
        else:
            return result.to_text()

    def find_relevant_text(self, document: str, topics: str, trace: Trace = Trace.OFF) -> str:
        '''Returns a list of section IDs that contain information relevant to the given query.

        Args:
        - document: plain text of a document.
        - topics: target topic(s), comma separated if more than one.
        '''
        sections_by_id = TextChunker().split_into_sections('n/a', document, chunk_length_characters=Default.CHUNK_SIZE, trace=trace)
        combined = '\n'.join(sections_by_id.values())
        prompt = Instructions(f'''
You are a detail oriented document analyzer designed for finding relevant information in large text documents.
You answer all questions in pure JSON, without any additional comments.
You are given a DOCUMENT with multiple sections. Each sections is enclsed in <SECTION> tags with a unique attribute "ID".
Here is the full text of this DOCUMENT:

**DOCUMENT**
{combined}

**INSTRUCTIONS**
1. Find all sections in the DOCUMENT above that include information about {topics}.
2. Save the IDs of all relevant sections to JSON array "section_ids".

**FINAL ANSWER**
```json
{{
    "section_ids": ["''')
        output = self.run(prompt, context=None, trace=trace).output
        if output.is_empty():
            return ''
        json_array = '["' + output.to_text()
        json_array = StringUtils.fix_json_format(json_array)
        json_array = json_array.strip().lstrip('{').rstrip('}')
        if not json_array.endswith(']'):
            json_array = json_array + ']'
        json_object = f'{{"arr": {json_array}}}'
        try:
            obj = json.loads(json_object)
            chunk_ids = obj['arr']
            log(f'find_relevant_text is returning chunks: {chunk_ids}')
            results = []
            for id in chunk_ids:
                if id not in sections_by_id:
                    raise BadPredictionError(f'Model returned unexpected section id: {id}')
                chunk = sections_by_id[id]
                open_tag = f'<SECTION ID="{id}">'
                chunk = chunk[len(open_tag):]
                chunk = chunk[:-len('</SECTION>')]
                results.append(chunk)
            return '\n\n'.join(results)
        except json.decoder.JSONDecodeError:
            log(f'Failed to parse json_object: {json_object}')
            return ''


    def find_citation(self, document: str, extracted_information: str, trace: Trace = Trace.OFF) -> str:
        '''Extracts text sectons from the given document that support the given claims or extracted datapoints.'''
        chunks_by_id = TextChunker().split_into_sections('citations', document, chunk_length_characters=Default.CHUNK_SIZE, trace=trace)
        combined = '\n'.join(chunks_by_id.values())
        prompt = Instructions(f'''
You are a detail oriented document analyzer designed for finding relevant information in large text documents.
You answer all questions in pure JSON format, without any additional comments.
You are given a DOCUMENT with multiple sections. Each sections is enclsed in <SECTION> tags with a unique attribute "ID".
Here is the full text of this DOCUMENT:

**DOCUMENT**
{combined}

**INSTRUCTIONS**
1. Find one SECTION in the DOCUMENT above that is the main source of the following information: {extracted_information}.
2. Return a JSON object with the ID of this SECTION.

```json
{{
    "section_id": "''')
        output = self.run(prompt, context=None, trace=trace).output
        if output.is_empty():
            return ''
        section_id = output.to_text().strip().replace('"', '')
        return section_id

    def remove_irrelevant_text(
            self,
            document: str,
            topic_definitions: list[str],
            preamble: str = '',
            chunk_length_characters: int = 1000,
            filler: str = '\n\n... this page was removed ...\n\n',
            include_adjacent_chunks = False,
            force_model: BaseModel | None = None,
            trace: Trace = Trace.OFF
        ) -> str:
        '''Splits the document in chunks, runs a multi-select classifier to find relevant chunks, returns a shorter document based on relevant chunks.'''
        if not document:
            raise InvalidArgsError(f'document is required.')
        if not topic_definitions:
            raise InvalidArgsError(f'topics must be a list of strings with definitions of target topics.')
        if chunk_length_characters < 100:
            raise InvalidArgsError(f'chunk_length_characters must be an integer greater than 100.')
        batch = TextChunker().get_batch('n/a', document, chunk_length_characters=chunk_length_characters)
        if batch.is_single():
            return batch.first().context.to_text()
        for i, interaction in enumerate(batch):
            interaction.tag = i
        classifier = MultiSelectQuestion(
            question = 'Which of the following statements are true based on the CONTEXT above?',
            preamble=preamble,
            answer_options = topic_definitions,
            verify_answers=True
        )
        saved_error_policy = self.postprocessor.error_policy
        self.postprocessor.error_policy = DataFormatErrorPolicy.RETURN_EMPTY_RESULT
        results = self.run_batch(classifier, batch, force_model=force_model, trace=trace).non_empty()
        self.postprocessor.error_policy = saved_error_policy
        relevant_text = []
        added_chunks = {}
        for result in results:
            chunk = result.context.to_text()
            added_chunks[result.tag] = True
            if include_adjacent_chunks and result.tag > 0:
                index = result.tag - 1
                if index >= 0 and index not in added_chunks:
                    chunk = batch[index].context.to_text() + '\n' + chunk
                    added_chunks[index] = True
                index = result.tag + 1
                if index < len(batch) and index not in added_chunks:
                    chunk = chunk + '\n' + batch[index].context.to_text()
                    added_chunks[index] = True
            relevant_text.append(chunk)
        combined_text = filler.join(relevant_text)
        log(f'remove_irrelevant_text found {len(relevant_text)} relevant chunks, returning {len(combined_text)} characters ({force_model.model_name if force_model else ""}).')
        return combined_text