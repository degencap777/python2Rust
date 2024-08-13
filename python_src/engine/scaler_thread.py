import logging
import threading
from tenacity import retry, wait_incrementing, retry_if_exception_type, before_sleep_log
from google.api_core.exceptions import ResourceExhausted, BadGateway, InternalServerError, InvalidArgument, ServerError, ServiceUnavailable
from openai import RateLimitError
from config.defaults import Default
from core.interaction import Interaction
from engine.inference_cache import InferenceCache
from models.base_model import BaseModel
from utils.exceptions import Http429RateLimitError, Http500ServerError
from utils.logger import log
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ScalerThread(threading.Thread):
    """Leveraging BoundedSemaphore for concurrency, Tenacity for recovering from common errors such as quota."""
    semaphore = threading.BoundedSemaphore(Default.MAX_INFERENCE_THREADS)

    def __init__(self, interaction: Interaction, cache: None | InferenceCache = None) -> None:
        super().__init__()
        self.interaction = interaction
        self.cache = cache
        self.abort_flag = False  # Scaler can set this flag to prevent further attempts to retry inference.

    def run(self) -> None:
        ScalerThread.semaphore.acquire()  # This will block if we already reached the maximum number of threads.
        if self.abort_flag:
            log(f'ScalerThread is exiting due to abort_flag:1')
            return
        try:
            is_default_model = True
            for model in self.interaction.models:  # Multiple models can help workaround blocked responses.
                if self.abort_flag:
                    log(f'ScalerThread is exiting due to abort_flag:2')
                    return
                self.predict(model)
                if not self.interaction.output.is_error():
                    break
                # If failback models are configured, the next loop will try them.
                if is_default_model:
                    is_default_model = False
                    self.interaction.flags.default_model_error_code = self.interaction.output.error_code
                else:
                    self.interaction.flags.fallback_model_error_code = self.interaction.output.error_code
                log(f'Detected blocked response from {model.model_name}')
            if self.interaction.output.is_error():
                log(f'Inference failed with {self.interaction.output.error_code}. Consider adding a fallback model to autorecover (current: {len(self.interaction.models)} models).')
        finally:
            ScalerThread.semaphore.release()

    @retry(retry=retry_if_exception_type(ResourceExhausted)
        | retry_if_exception_type(Http429RateLimitError)
        | retry_if_exception_type(Http500ServerError)
        | retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(BadGateway)
        | retry_if_exception_type(InternalServerError)
        | retry_if_exception_type(InvalidArgument)
        | retry_if_exception_type(ServerError)
        | retry_if_exception_type(ServiceUnavailable),
        wait=wait_incrementing(0.5, 1, max=60),
        # before_sleep=before_sleep_log(logger, logging.INFO) enable for verbose logging
    )
    def predict(self, model: BaseModel) -> None:
        timers = self.interaction.timers
        if model.model_name.lower().startswith('gemini-1.5'):
            model.config.generation['response_mime_type'] = 'application/json' if self.interaction.is_output_json else 'text/plain'
        if model.model_name.lower().startswith('gpt-4'):
            model.config.generation['response_format'] = {'type': 'json_object' if self.interaction.is_output_json else 'text'}
        timers.cache_read_latency.start()
        cached_output = self.cache.get(model.model_name, model.config, self.interaction.prompt) if self.cache else None
        timers.cache_read_latency.stop()
        if self.abort_flag:
            log(f'ScalerThread is exiting due to abort_flag:3')
            return
        if cached_output:
            self.interaction.output = cached_output
            self.interaction.flags.cache_hit = True
        else:
            log(f'Cache miss on {model.model_name}', self.interaction.trace)
            if self.interaction.prompt.is_text() and len(self.interaction.prompt.to_text()) > model.max_input_tokens:
                log(f'Submitting long prompt to {model.model_name}: {len(self.interaction.prompt.to_text())} chars.')
            timers.inference_latency.start()
            try:
                self.interaction.output = model.predict(self.interaction.prompt, model.config)
            except Exception as e:
                if self.interaction.trace:
                    log(f'Retrying exception: {e}')
                raise e
            if self.abort_flag:
                log(f'ScalerThread is exiting due to abort_flag:4')
                return
            self.interaction.raw_model_output_text = self.interaction.output.to_text()
            timers.inference_latency.stop()
            if hasattr(self.interaction.output, 'inference_metadata') and self.interaction.output.inference_metadata:
                self.interaction.inference_metadata = self.interaction.output.inference_metadata
            if self.cache:
                timers.cache_write_latency.start()
                self.cache.set(model.model_name, model.config, self.interaction.prompt, self.interaction.output)
                timers.cache_write_latency.stop()
        self.interaction.output.fix_apostrophes_in_text()
        self.interaction.selected_model = model
        if self.interaction.trace > 0:
            if self.interaction.trace > 1:
                log(model.config)
            log(self.interaction)