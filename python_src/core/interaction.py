from typing import Any
from dataclasses import dataclass
from core.conversation import Conversation
from core.data_bundle import DataBundle
from core.data_source import DataSource
from core.error_code import ErrorCode
from core.inference_metadata import InferenceMetadata
from models.base_model import BaseModel
from utils.logger import Trace
from utils.timer import Timer

class InteractionTimers():
    def __init__(self):
        self.inference_latency = Timer()
        self.cache_read_latency = Timer()
        self.cache_write_latency = Timer()

@dataclass
class InteractionFlags:
    cache_hit = False
    default_model_error_code = ErrorCode.UNDEFINED
    fallback_model_error_code = ErrorCode.UNDEFINED

@dataclass
class Interaction():
    """Represents a multimodal inference including context, prompt, outputs, model configuration, and instrumentation."""
    def __init__(
        self,
        context: Conversation,
        model: list[BaseModel] | BaseModel | None = None,
        trace: Trace = Trace.OFF,
        data_source: DataSource | None = None
    ):
        self.task: Any              # optional link to the partent Task for instrumentation
        self.primitive: Any | None = None  # optional link to the Primitive for instrumentation
        self.context = context      # data for prompt generation
        self.prompt: Conversation   # prompt template with injected context
        self.output: DataBundle
        self.is_output_json: bool = False
        self.models: list[BaseModel] = []
        if isinstance(model, BaseModel):
            self.models.append(model)
        elif isinstance(model, list):
            self.models.extend(model)
        self.selected_model: BaseModel | None = None
        self.raw_model_output_text: str = ''  # preserves raw model output for tracking in Spanner.Inferences
        self.trace = trace
        self.flags = InteractionFlags()
        self.timers = InteractionTimers()
        self.inference_metadata = InferenceMetadata()
        self.parent_interaction: Interaction   # optional link to the parent in a chain of Interactions
        self.tag: Any                          # used for internal processing in Proton (temporary mapping)
        self.data_source = data_source         # optional metadata about the source of data for this interaction
        if not data_source:
            if context and not context.is_empty():
                self.data_source = context.first_data_bundle().data_source