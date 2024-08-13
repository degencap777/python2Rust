import hashlib
from PIL import Image as PILImage
from typing import Any
from core.conversation import Conversation
from core.data_bundle import DataBundle
from core.data_source import DataSource
from core.interaction import Interaction
from models.base_model import BaseModel
from storage.local_storage import LocalStorage
from utils.exceptions import InvalidArgsError
from utils.logger import Trace

class Batch():
    """Represents a list of interactions to facilitate batch processing in primitives and tasks."""

    def __init__(self, interactions: list[Interaction], data_source: DataSource | None = None):
        self.storage = LocalStorage()
        self.interactions = interactions
        self.data_source = data_source
        if not data_source:
            if interactions:
                self.data_source = interactions[0].data_source

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, index):
        return self.interactions[index]

    def __iter__(self):
        return iter(self.interactions)

    def __repr__(self) -> str:
        if len(self.interactions) == 1:
            return f'Batch of 1: {self.interactions[0]}'
        return f'Batch({len(self.interactions)} items, #{self.context_hash()})'

    def context_hash(self):
        hashes = ''.join([i.context.hash() for i in self.interactions])
        return hashlib.sha256(hashes.encode('utf-8')).hexdigest()[:10]

    def data_source_name(self) -> str:
        if self.interactions:
            interaction = self.first()
            source = interaction.data_source
            return source.name if source else 'NULL_SOURCE'
        return 'NULL_SOURCE'

    def is_empty(self):
        return len(self.interactions) == 0

    def is_single(self):
        return len(self.interactions) == 1

    def first(self) -> Interaction:
        if not self.interactions:
            raise InvalidArgsError(f'Batch is empty.')
        return self.interactions[0]

    def last(self) -> Interaction:
        if not self.interactions:
            raise InvalidArgsError(f'Batch is empty.')
        return self.interactions[-1]

    def append(self, item: Interaction):
        if not isinstance(item, Interaction):
            raise InvalidArgsError(f'Only Interaction objects can be added to a Batch. Received: {type(item)}.')
        self.interactions.append(item)

    def import_interactions(self, items: list[Interaction]):
        self.interactions.extend(items)

    def non_empty(self):
        """Removes iteractions with empty model outputs."""
        return Batch([item for item in self.interactions if not item.is_output_empty()])

    def unique(self):
        by_hash = {}
        for item in self.interactions:
            if not item.is_output_empty():
                hash = item.output.hash()
                if hash not in by_hash:
                    by_hash[hash] = item
        return Batch(list(by_hash.values()))

    def output_values(self) -> list[Any]:
        values = []
        for item in self.interactions:
            if not item.is_output_empty():
                values.extend(item.output.values())
        return values

    def non_empty_count(self):
        return len([item for item in self.interactions if not item.is_output_empty()])

    def filter_by_output(self, value: Any):
        """Creates a batch from previously executed inferences that have the specified output value."""
        if not value:
            raise InvalidArgsError('Cannot filter by an empty value')
        return Batch([item for item in self.interactions if not item.output.is_empty() and item.output.first().value == value])

    def combine_outputs(self, defrag_text=True) -> DataBundle:
        result = DataBundle.empty()
        if self.is_empty():
            return result
        for iteration in self.interactions:
            if not iteration.is_output_empty():
                result.import_bundle(iteration.output)
        if result.is_text() and defrag_text:
            combined_text = '\n\n'.join([data.value for data in result])
            return DataBundle.from_text(combined_text)
        else:
            return result

    def combine_interaction_contexts(self, defrag_text=True) -> DataBundle:
        result = DataBundle.empty()
        if self.is_empty():
            return result
        for interaction in self.interactions:
            bundles = interaction.context.all_data_bundles()
            for bundle in bundles:
                result.import_bundle(bundle)
        if result.is_text() and defrag_text:
            combined_text = '\n\n'.join(result.values())
            result = DataBundle.from_text(combined_text)
        return result

    def combine_interactions(self) -> Interaction:
        """Combine the context databundles of all interactions in this batch into a single interaction."""
        combined_bundle = self.combine_interaction_contexts(defrag_text=True)
        return Interaction(Conversation.from_data_bundle(combined_bundle))

    def models(self) -> list[BaseModel]:
        if self.is_empty():
            raise InvalidArgsError('Batch is empty.')
        return self.interactions[0].models

    def default_model(self) -> BaseModel:
        if self.is_empty():
            raise InvalidArgsError('Batch is empty.')
        if not self.interactions[0].models:
            raise InvalidArgsError('Default model is missing.')
        return self.interactions[0].models[0]

    def trace_first_prompt(self) -> None:
        if not self.is_empty():
            self.interactions[0].trace = Trace.ON

    def trace_all_prompts(self) -> None:
        for item in self.interactions:
            item.trace = Trace.VERBOSE

    @staticmethod
    def from_text_folder(path_to_local_directory: str) -> 'Batch':
        text_by_filename = LocalStorage().read_text_files_from_dir(path_to_local_directory)
        texts = [text_by_filename[filename] for filename in text_by_filename.keys()]
        interactions = [Interaction(Conversation.from_text(text)) for text in texts]
        return Batch(interactions)

    @staticmethod
    def from_list_of_strings(texts: list[str]) -> 'Batch':
        interactions = [Interaction(Conversation.from_text(text)) for text in texts]
        return Batch(interactions)

    @staticmethod
    def from_image_list(images: list[PILImage.Image]) -> 'Batch':
        interactions = [Interaction(Conversation.from_image(image)) for image in images]
        return Batch(interactions)

    @staticmethod
    def from_image_folder(path_to_local_directory: str) -> 'Batch':
        image_by_filename = LocalStorage().read_images_from_dir(path_to_local_directory)
        images = [image_by_filename[filename] for filename in image_by_filename.keys()]
        return Batch.from_image_list(images)

    @staticmethod
    def from_data_bundle(bundle: DataBundle) -> 'Batch':
        return Batch.from_data_bundles([bundle])

    @staticmethod
    def from_data_bundles(bundles: list[DataBundle]) -> 'Batch':
        for bundle in bundles:
            if not isinstance(bundle, DataBundle):
                raise InvalidArgsError(f'from_data_bundles called with invalid list item: {type(bundle)}')
        interactions = [Interaction(Conversation.from_data_bundle(bundle), data_source=bundle.data_source) for bundle in bundles]
        return Batch(interactions)
