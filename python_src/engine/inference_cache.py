import json
import threading
from time import time
from google.cloud import spanner
from core.conversation import Conversation
from core.data_bundle import DataBundle
from models.model_config import ModelConfig
from utils.exceptions import InvalidArgsError
from utils.logger import Trace, log

# The SoT is in Cloud Spanner so that teams can share cached inferences.
# Model-specific cache is preloaded to a local dict to avoid the ~60ms Spanner latency.
# TODO: add option to limit preloading to recent data with following cache misses checked against Spanner.
# TODO: add an option to manage local cache in a separate process to minimize redundant preloads.
# TODO: add an option to disable cache when inference temperature is above zero.
class InferenceCache():
    _lock = threading.Lock()

    def __init__(self, project_id: str, spanner_instance_id: str, spanner_database_id: str, enable_local_cache: bool = True, trace: Trace = Trace.OFF):
        self.enable_local_cache = enable_local_cache
        self.cache_by_model_config = {}
        self.spanner_instance_id = spanner_instance_id
        self.spanner_database_id = spanner_database_id
        log(f'InferenceCache uses DB {spanner_instance_id}/{spanner_database_id} in project {project_id}', trace)
        self.client = spanner.Client(project_id)
        self.instance = self.client.instance(spanner_instance_id)
        self.database = self.instance.database(spanner_database_id)
        self.trace = trace
        self._overwrite_cache = False  # This flag is not intended for Proton users
        self._probe_spanner()
        log(f'InferenceCache is ready', trace)

    def _probe_spanner(self) -> None:
        """Raise exception if the spanner instance or database are missing."""
        self.get('spanner_prober', ModelConfig(), Conversation.from_text('prober'))

    def _lazy_preload_by_model_config(self, model_config_key: str):
        if not self.enable_local_cache:
            return
        with InferenceCache._lock:
            if model_config_key not in self.cache_by_model_config:
                start = time()
                self.cache_by_model_config[model_config_key] = {}
                with self.database.snapshot() as snapshot:
                    results = snapshot.execute_sql('SELECT input_hash, output FROM InferenceCache WHERE model = @model',
                        params={'model': model_config_key},
                        param_types={'model': spanner.param_types.STRING}
                    )
                    for row in results:
                        self.cache_by_model_config[model_config_key][row[0]] = row[1]
                    if 'spanner_prober' not in model_config_key:
                        log(f'Preloaded {len(self.cache_by_model_config[model_config_key])} inferences for model {model_config_key} in {time() - start:.1f} seconds')

    def _model_config_key(self, model_name: str, model_config: ModelConfig) -> str:
        """Create a short key that identifies the base model, the tuned adapter, and generation config."""
        model_params_hash = model_config.hash_generation_config()
        if model_params_hash:
            return model_name + '_' + model_params_hash
        else:
            return model_name

    def get(self, model_name: str, model_config: ModelConfig, prompt: Conversation) -> None | DataBundle:
        """Gets model output from cache.

        Args:
            model_name: {base_model_name_version}_{optional_tuned_adapter_id}
            model_config: provides the hash of the generation config which is a necessary part of the cache key.
            prompt: Conversation representing multi-modal inputs.
        """
        if not model_name:
           raise InvalidArgsError(f'You must pass a model name including version')
        if not prompt:
           raise InvalidArgsError(f'Prompt is required')
        if not model_config:
           raise InvalidArgsError(f'ModelConfig is required')

        if self._overwrite_cache:
            return None
        model_config_key = self._model_config_key(model_name, model_config)
        inference_key = prompt.hash()
        if self.trace == Trace.VERBOSE:
            log(f'Cache key: {model_config_key}.{inference_key} for config: {model_config}')
        if self.enable_local_cache:
            self._lazy_preload_by_model_config(model_config_key)
            if inference_key in self.cache_by_model_config[model_config_key]:
                serialized = self.cache_by_model_config[model_config_key][inference_key]
                bundle = DataBundle._deserialize(json.loads(serialized))
                return bundle
            else:
                return None
        else:
            with self.database.snapshot() as snapshot:
                results = snapshot.execute_sql("""
                    SELECT output
                    FROM InferenceCache
                    WHERE model = @model AND input_hash = @input_hash
                    """,
                    params={'model': model_config_key, 'input_hash': prompt.hash()},
                    param_types={'model': spanner.param_types.STRING, 'input_hash': spanner.param_types.STRING}
                )
                for row in results:
                    return DataBundle._deserialize(json.loads(row[0]))
                log(f'{model_name} cache miss on prompt {prompt.hash()}', self.trace)
                return None

    def set(self, model_name: str, model_config: ModelConfig, prompt: Conversation, output: DataBundle):
        """Adds one inference to cache.

        Args:
            model_name: {base_model_name_version}_{optional_tuned_adapter_id}
            model_config: provides the hash of the generation config which is a necessary part of the cache key.
            prompt: Conversation representing multi-modal inputs.
            outpu: DataBundle representing multimodal output to be cached.
        """
        if not model_name:
           raise InvalidArgsError(f'You must pass a model name including version')
        if not prompt:
           raise InvalidArgsError(f'Prompt is required')
        if not model_config:
           raise InvalidArgsError(f'ModelConfig is required')
        serialized_output = json.dumps(output._serialize())
        model_config_key = self._model_config_key(model_name, model_config)
        if self.enable_local_cache:
            self._lazy_preload_by_model_config(model_config_key)
            self.cache_by_model_config[model_config_key][prompt.hash()] = serialized_output
        # TODO: add an option to persist local cache to Spanner asyncronously.
        with self.database.batch() as batch:
            batch.insert_or_update(
                table = 'InferenceCache',
                columns = ('model', 'input_hash', 'output', 'insert_date'),
                values = [(model_config_key, prompt.hash(), serialized_output, spanner.COMMIT_TIMESTAMP)]
            )

    # Cannot run a one shot delete because Spanner is limited to 80K mutations per transaction
    def purge_cache_for_model_config( self, model_name: str, model_config: ModelConfig ):
        model_config_key = self._model_config_key(model_name, model_config)
        with self.database.snapshot() as snapshot:
            results = snapshot.execute_sql('SELECT input_hash FROM InferenceCache WHERE model = @model',
            params={"model": model_config_key},
            param_types={"model": spanner.param_types.STRING}
        )
        hashes = [row[0] for row in results]
        log(f'purge_cache_for_model is deleting all {len(hashes)} cached inferences for {model_config_key}')
        num_chunks = round(len(hashes) / 1000)  # smaller chunks to avoid slowing down other spanner users.
        chunks = [hashes[i::num_chunks] for i in range(num_chunks)]
        for i, chunk in enumerate( chunks ):
            with self.database.batch() as batch:
                keys = spanner.KeySet(keys=[[model_config_key, hash] for hash in chunk])
                batch.delete('InferenceCache', keys)
            log(f'Deleted chunk #{i}')
        log('purge_cache_for_model is finished')