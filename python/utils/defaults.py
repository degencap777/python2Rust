import os
from dataclasses import dataclass

@dataclass
class Default:
    TEMPERATURE = 0
    TEMPERATURE_GEMINI = 0.36    # 0.36 is based on a Bayesian optimization for GeminiPro on document extraction tasks
    TOP_K = 40
    TOP_P = 0.97
    TOP_P_GEMINI = 1.0
    TOP_P_OPENAI = None         # official recommendation is to use either temperature or top_p
    MAX_OUTPUT_TOKENS = 2048    # tokens
    MAX_OUTPUT_TOKENS_GEMINI15 = 8192
    MAX_OUTPUT_TOKENS_ANTHROPIC = 4096
    MAX_OUTPUT_TOKENS_LLAMA_70B = 2048
    CHUNK_SIZE = 1000           # characters
    CHUNK_OVERLAP = 0           # characters
    CHUNKER_MAX_SENTENCE = 1000 # characters
    FILE_ENCODING = 'utf-8' # used to be 'ISO-8859-1'
    GCP_LOCATION = 'us-central1'
    SPANNER_INSTANCE = 'proton'
    SPANNER_TIMEOUT = 300       # seconds, the slowest operation is index creation
    SPANNER_CACHE_DB = 'inference_cache'
    SPANNER_EVAL_DB = 'evaluations'
    SPANNER_WORKER_DB = 'worker'
    MODEL_PALM = 'text-bison-32k@002'
    MODEL_GECKO = 'textembedding-gecko@003'
    MODEL_GEMINI_TEXT = 'gemini-1.5-flash-preview-0514'
    MODEL_GEMINI_MULTIMODAL = 'gemini-1.5-flash-preview-0514'
    MODEL_OPENAI_TEXT = 'gpt-4-turbo-preview'
    MODEL_CLAUDE3_HAIKU = 'claude-3-haiku@20240307'
    MODEL_CLAUDE3_SONNET = 'claude-3-sonnet@20240229'
    MODEL_CLAUDE3_OPUS = 'claude-3-opus@20240229'
    MODEL_LLAMA_70B = 'llama-3-70b@001'
    MODEL_LLAMA_70B_IT = 'llama-3-70b-it@001'
    MODEL_SHORTDOC_PRIMARY = 'gemini-1.0-pro-001'
    MODEL_SHORTDOC_FALLBACK = 'gemini-1.5-flash-preview-0514'
    MISTRAL_MODEL = 'Mistral-7B-IT-01'
    GEMMA_MODEL = 'gemma-7b-it'
    MAX_INFERENCE_THREADS = 20
    INFERENCE_THREAD_TIMEOUT_SECONDS = 60*4  # Larger timeout to accommodate potential wait due to enter the thread pool
    INFERENCE_THREAD_MAX_TIMEOUT_RETRIES = 3
    NOT_FOUND_TAG = 'NOT_FOUND'
    SECTION_NOT_FOUND_TAG = 'NO_SECTION'  # Used in QuestionSet primitive to avoid NOT_FOUND_TAG which causes the entire inference to be filtered out.
    PROJECT_ID = os.getenv('PROJECT_ID', '')
    SEARCH_DATASTORE = os.getenv('SEARCH_DATASTORE', '')
    BUCKET_NAME = os.getenv('BUCKET_NAME', '')
    VS_LOCATION = 'global'      # vertex search
    VS_DEFAULT_CONFIG = 'default_config'
    VS_MAX_RESULTS = 10         # maximum = 100
    VS_NUM_SUMMARY_SOURCES = 5  # maximum = 5
    VS_NUM_EXTRACTIVE_ANSWERS = 1   # per document, maximum = 5
    VS_NUM_EXTRACTIVE_SEGMENTS = 5  # per document, maximum = 10
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_ORG_ID = os.getenv('OPENAI_ORG_ID')
    WORKER_DEFAULT_NAMESPACE = 'DEFAULT'
    WORKER_MAX_TASK_RETRIES = 3
    WORKER_POLLING_INTERVAL_SECONDS = 1.0
    API_PAGE_SIZE = 100
    SHORTDOC_PREAMBLE = 'You are a detail oriented AI assistant responsible for reading documents and answering questions about them.'
    SHORTDOC_CHUNK_SIZE = 3000
    