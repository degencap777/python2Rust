import threading
from config.defaults import Default
from core.interaction import Interaction
from engine.scaler_thread import ScalerThread
from engine.inference_cache import InferenceCache
from utils.exceptions import InvalidArgsError, InferenceThreadTimeout
from utils.logger import Trace, log

class Scaler():
   """Run multi-threaded inference on any model with automatic recovery from common errors."""
   _lock = threading.Lock()

   def __init__(self, inference_cache: None | InferenceCache = None) -> None:
      self.inference_cache = inference_cache
      self.track_inferences = False
      self.history: list[Interaction] = []
      self.num_inferences = 0
      self.num_input_tokens = 0
      self.num_output_tokens = 0
      self.num_cached_inferences = 0

   def start_inference_tracking(self):
      self.history = []
      self.track_inferences = True

   def stop_inference_tracking(self) -> list[Interaction]:
      self.track_inferences = False
      return self.history

   def reset_stats(self) -> None:
      self.num_inferences = 0
      self.num_input_tokens = 0
      self.num_output_tokens = 0
      self.num_cached_inferences = 0

   def log_stats(self) -> None:
      log(f'Inference stats: {self.num_cached_inferences} cached + {self.num_inferences} real ({self.num_input_tokens}->{self.num_output_tokens} tokens)')

   def _run(self, interactions: list[Interaction], trace: Trace = Trace.OFF) -> list[Interaction]:
      """Run threads, enforce timeouts, return the list of timed out threads."""
      threads = []
      interactions_timed_out = []
      with Scaler._lock:
         for interaction in interactions:
            if trace > 0:
               interaction.trace = trace
            thread = ScalerThread(interaction, self.inference_cache)
            thread.start()
            threads.append(thread)
      for thread in threads:
         thread.join(timeout=Default.INFERENCE_THREAD_TIMEOUT_SECONDS)
         if thread.is_alive():
            interactions_timed_out.append(thread.interaction)
      return interactions_timed_out

   def _run_with_timeout_retry(self, interactions: list[Interaction], trace: Trace = Trace.OFF) -> None:
      """Run inferences in parallel threads, retry timeouts if needed."""
      for interaction in interactions:
         if not interaction.models:
            raise InvalidArgsError(f'Interaction {interaction} does not have a model.')
      attempt_count = 0
      interactions_remaining = interactions.copy()
      # Retry any timed out interactions
      while True:
         if attempt_count > 0:
            log(f'Scaler is retrying {len(interactions_remaining)} timed out inferences...')
         interactions_timed_out = self._run(interactions_remaining, trace)
         if len(interactions_timed_out) == 0:
            break
         attempt_count += 1
         if attempt_count >= Default.INFERENCE_THREAD_MAX_TIMEOUT_RETRIES:
            raise InferenceThreadTimeout(f'{len(interactions_timed_out)} out of {len(interactions)} threads timed out {attempt_count} times.')
         interactions_remaining = interactions_timed_out
      if self.track_inferences:
         self.history += interactions
      for interaction in interactions:
         if interaction.flags.cache_hit:
            self.num_cached_inferences += 1
         else:
            self.num_inferences += 1
            self.num_input_tokens += interaction.inference_metadata.num_input_tokens
            self.num_output_tokens += interaction.inference_metadata.num_output_tokens


   def run_batch(self, interactions: list[Interaction], trace: Trace = Trace.OFF):
      self._run_with_timeout_retry(interactions, trace)

   def run_single(self, interaction: Interaction, trace: Trace = Trace.OFF):
      self._run_with_timeout_retry([interaction], trace)