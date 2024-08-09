use std::fs;
use std::path::Path;
extern crate unidecode;
use chrono::{DateTime, Utc};
use image::{DynamicImage, GenericImageView};
use regex::Regex;
use std::collections::HashMap;
use std::fmt;
use unidecode::unidecode;

/// A struct containing default values for various parameters used throughout the application.
pub struct Defaults;

// Constants in `Defaults` struct provide configuration settings for various functionalities.
impl Defaults {
    pub const TEMPERATURE: f32 = 0.0;
    pub const TEMPERATURE_GEMINI: f32 = 0.36; 
    pub const TOP_K: usize = 40;
    pub const TOP_P: f32 = 0.97;
    pub const TOP_P_GEMINI: f32 = 1.0;
    pub const TOP_P_OPENAI: Option<f32> = None; 
    pub const MAX_OUTPUT_TOKENS: usize = 2048; 
    pub const MAX_OUTPUT_TOKENS_GEMINI15: usize = 8192;
    pub const MAX_OUTPUT_TOKENS_ANTHROPIC: usize = 4096;
    pub const MAX_OUTPUT_TOKENS_LLAMA_70B: usize = 2048;
    pub const CHUNK_SIZE: usize = 1000; 
    pub const CHUNK_OVERLAP: usize = 0; 
    pub const CHUNKER_MAX_SENTENCE: usize = 1000; 
    pub const FILE_ENCODING: &'static str = "utf-8"; 
    pub const GCP_LOCATION: &'static str = "us-central1";
    pub const SPANNER_INSTANCE: &'static str = "proton";
    pub const SPANNER_TIMEOUT: u64 = 300; // seconds
    pub const SPANNER_CACHE_DB: &'static str = "inference_cache";
    pub const SPANNER_EVAL_DB: &'static str = "evaluations";
    pub const SPANNER_WORKER_DB: &'static str = "worker";
    pub const MODEL_PALM: &'static str = "text-bison-32k@002";
    pub const MODEL_GECKO: &'static str = "textembedding-gecko@003";
    pub const MODEL_GEMINI_TEXT: &'static str = "gemini-1.5-flash-preview-0514";
    pub const MODEL_GEMINI_MULTIMODAL: &'static str = "gemini-1.5-flash-preview-0514";
    pub const MODEL_OPENAI_TEXT: &'static str = "gpt-4-turbo-preview";
    pub const MODEL_CLAUDE3_HAIKU: &'static str = "claude-3-haiku@20240307";
    pub const MODEL_CLAUDE3_SONNET: &'static str = "claude-3-sonnet@20240229";
    pub const MODEL_CLAUDE3_OPUS: &'static str = "claude-3-opus@20240229";
    pub const MODEL_LLAMA_70B: &'static str = "llama-3-70b@001";
    pub const MODEL_LLAMA_70B_IT: &'static str = "llama-3-70b-it@001";
    pub const MODEL_SHORTDOC_PRIMARY: &'static str = "gemini-1.0-pro-001";
    pub const MODEL_SHORTDOC_FALLBACK: &'static str = "gemini-1.5-flash-preview-0514";
    pub const MISTRAL_MODEL: &'static str = "Mistral-7B-IT-01";
    pub const GEMMA_MODEL: &'static str = "gemma-7b-it";
    pub const MAX_INFERENCE_THREADS: usize = 20;
    pub const INFERENCE_THREAD_TIMEOUT_SECONDS: u64 = 60 * 4; 
    pub const INFERENCE_THREAD_MAX_TIMEOUT_RETRIES: usize = 3;
    pub const NOT_FOUND_TAG: &'static str = "NOT_FOUND";
    pub const SECTION_NOT_FOUND_TAG: &'static str = "NO_SECTION"; 
    pub const VS_LOCATION: &'static str = "global"; 
    pub const VS_DEFAULT_CONFIG: &'static str = "default_config";
    pub const VS_MAX_RESULTS: usize = 10; 
    pub const VS_NUM_SUMMARY_SOURCES: usize = 5; 
    pub const VS_NUM_EXTRACTIVE_ANSWERS: usize = 1; 
    pub const VS_NUM_EXTRACTIVE_SEGMENTS: usize = 5; 
    pub const WORKER_DEFAULT_NAMESPACE: &'static str = "DEFAULT";
    pub const WORKER_MAX_TASK_RETRIES: usize = 3;
    pub const WORKER_POLLING_INTERVAL_SECONDS: f64 = 1.0;
    pub const API_PAGE_SIZE: usize = 100;
    pub const SHORTDOC_PREAMBLE: &'static str = "You are a detail oriented AI assistant responsible for reading documents and answering questions about them.";
    pub const SHORTDOC_CHUNK_SIZE: usize = 3000;
}

/// Represents the type of data source (e.g., Document, Image, Directory).
#[derive(Debug, Clone)]
pub enum DataSourceType {
    Document = 1,
    Image = 2,
    Directory = 3,
}

/// Represents a data source with its type, name, location, and version.
#[derive(Debug, Clone)]
pub struct DataSource {
    pub source_type: DataSourceType,
    pub name: String,
    pub location: String,
    pub version: String,
}

impl Default for DataSource {
    // Default implementation for DataSource struct
    fn default() -> Self {
        Self {
            source_type: DataSourceType::Document,
            name: String::new(), 
            location: String::new(),
            version: String::new(),
        }
    }
}

/// Represents different types of data (e.g., Text, Int, Float).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Undefined = 1, // Triggers automatic type detection
    Text = 2,      // str
    Int = 3,
    Float = 4,
    Bool = 5,
    Char = 6,
    Date = 7,
    JsonArray = 8,           // TODO: should we rename it to LIST_OF_STRINGS?
    JsonDict = 9,            // TODO: should we rename it to DICT?
    Image = 10,              // PIL.Image
    MultiselectAnswers = 11, // List of answer option labels like ['B', 'F']
    Pdf = 12,
    QuestionSetAnswers = 13, // List of answers
    DateTime = 14,
}

/// Represents a piece of data with its value, ID, and type.
#[derive(Debug)]
pub struct Data {
    value: DataValue,
    id: String,
    data_type: DataType,
}

impl Data {
    /// Creates a new `Data` instance with a given value, type, and ID.
    pub fn new(value: DataValue, data_type: DataType, id: &str) -> Self {
        let mut data_type = data_type;

        if data_type == DataType::Undefined {
            data_type = Self::detect_data_type(&value);
        }

        Data {
            value,
            id: id.to_string(),
            data_type,
        }
    }

    /// Detects the data type of a given `DataValue`.
    pub fn detect_data_type(value: &DataValue) -> DataType {
        match value {
            DataValue::Bool(_) => DataType::Bool,
            DataValue::Int(_) => DataType::Int,
            DataValue::Text(_) => DataType::Text,
            DataValue::Float(_) => DataType::Float,
            DataValue::Date(_) => DataType::Date,
            DataValue::Image(_) => DataType::Image,
            DataValue::MultiselectAnswers(_) => DataType::MultiselectAnswers,
            DataValue::JsonArray(_) => DataType::JsonArray,
            DataValue::JsonDict(_) => DataType::JsonDict,
            DataValue::Pdf(_) => DataType::Pdf,
            &DataValue::Char(_) => todo!(),
        }
    }
}

impl fmt::Display for Data {
    /// Formats the `Data` instance as a string for display.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            DataValue::Image(img) => {
                let (width, height) = img.dimensions();
                write!(f, "{}: [{}x{}]", self.data_type as u8, width, height)
            }
            DataValue::Pdf(content) => {
                write!(f, "PDF: [{}]", content.len())
            }
            DataValue::Text(text) => {
                let truncated = &text[..text.len().min(100)]; // Truncate to 100 characters
                write!(
                    f,
                    "{}[{}]: {}",
                    self.data_type as u8,
                    text.len(),
                    truncated.replace("\n", " ")
                )
            }
            _ => write!(f, "{}: {:?}", self.data_type as u8, self.value), // Default case
        }
    }
}

/// Represents the various possible values for `Data`.
#[derive(Debug)]
pub enum DataValue {
    Text(String),
    Int(i32),
    Float(f64),
    Bool(bool),
    Char(char),
    Date(DateTime<Utc>),
    Image(DynamicImage),
    MultiselectAnswers(Vec<String>),
    JsonArray(Vec<String>),
    JsonDict(HashMap<String, String>),
    Pdf(Vec<u8>),
}

impl DataValue {
    /// Checks if the `DataValue` is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            DataValue::Text(s) => s.is_empty(),
            DataValue::MultiselectAnswers(vec) => vec.is_empty(),
            DataValue::JsonArray(vec) => vec.is_empty(),
            _ => false, 
        }
    }
}

/// Represents error codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    Undefined = 1,
    ResponseBlocked = 2, 
}

/// Contains metadata information related to inference.
#[derive(Debug, Default)]
pub struct InferenceMetadata {
    num_input_tokens: usize,
    num_output_tokens: usize,
}

impl InferenceMetadata {
    /// Creates a new instance of `InferenceMetadata` with specified input and output token counts.
    pub fn new(num_input_tokens: usize, num_output_tokens: usize) -> Self {
        InferenceMetadata {
            num_input_tokens,
            num_output_tokens,
        }
    }
}

impl fmt::Display for InferenceMetadata {
    /// Formats `InferenceMetadata` as a string for display.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "num_input_tokens={}, num_output_tokens={}",
            self.num_input_tokens, self.num_output_tokens
        )
    }
}

/// Represents a bundle of data items with associated metadata.
#[derive(Debug)]
pub struct DataBundle {
    items: Vec<Data>,                      
    error_code: ErrorCode,                 
    inference_metadata: InferenceMetadata, 
    data_source: Option<DataSource>,      
}

impl DataBundle {
    /// Creates a new `DataBundle` with specified items, error code, and data source.
    pub fn new(items: Vec<Data>, error_code: ErrorCode, data_source: Option<DataSource>) -> Self {
        DataBundle {
            items,
            error_code,
            inference_metadata: InferenceMetadata::default(),
            data_source,
        }
    }

    /// Returns a string representation of the `DataBundle`.
    fn repr(&self) -> String {
        if self.is_empty() {
            return "EMPTY".to_string();
        }

        if self.items.len() == 1 {
            return format!("Bundle[{}]", self.items[0]);
        } else {
            return format!(
                "Bundle[{} items starting with {}]",
                self.items.len(),
                self.items[0]
            );
        }
    }

    /// Checks if the `DataBundle` is empty.
    pub fn is_empty(&self) -> bool {
        if self.items.is_empty() {
            return true;
        }
        self.items.iter().all(|item| item.value.is_empty()) 
    }
}

impl fmt::Display for DataBundle {
    /// Implements the Display trait for `DataBundle` to format it as a string.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.repr())
    }
}

/// Splits a text into sentences based on specific punctuation marks.
pub fn split_into_sentences(text: &str) -> Vec<String> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    let re: Regex = Regex::new(r"(^|[^a-zA-Z0-9])([.!?]+)(\s+|\n|\r|\r\n|$)").unwrap();

    let mut sentences: Vec<String> = Vec::new();
    let mut previous_end: usize = 0;

    for mat in re.find_iter(text) {
        let start: usize = previous_end;
        let end: usize = mat.end();
        previous_end = end;

        let sentence = &text[start..end].trim_end();
        sentences.push(sentence.to_string());
    }

    if previous_end < text.len() {
        let remaining_sentence = &text[previous_end..].trim_start();
        if !remaining_sentence.is_empty() {
            sentences.push(remaining_sentence.to_string());
        }
    }

    sentences
}

/// Merges chunks of text into a single string based on starting index and number of chunks.
pub fn _merge_chunks(chunks: &[String], start_index: usize, num_chunks: usize) -> String {
    let mut group = Vec::new();

    for i in 0..num_chunks {
        if start_index + i < chunks.len() {
            group.push(chunks[start_index + i].clone()); 
        }
    }
    group.join("\n") 
}

/// Creates non-overlapping chunks from provided text based on specified chunk size.
pub fn create_nonoverlapping_chunks(text: &str, chunk_length_characters: usize) -> Vec<String> {
    let sentences = split_into_sentences(text);
    let mut chunks = Vec::new();
    let mut chunk = Vec::new();
    let mut chunk_size = 0;
    let mut max_chunk = 0;

    for sentence in sentences {
        if sentence.len() > Defaults::CHUNKER_MAX_SENTENCE {
            println!("Warning: long sentence ({} chars)", sentence.len());
        }
        chunk.push(sentence.clone());
        chunk_size += sentence.len();

        if chunk_size >= chunk_length_characters {
            let combined = chunk.join("");
            max_chunk = max_chunk.max(combined.len());
            chunks.push(combined);
            chunk.clear();
            chunk_size = 0;
        }
    }

    if !chunk.is_empty() {
        let combined = chunk.join(" ").trim().to_string();
        chunks.push(combined);
    }

    println!(
        "Created {} non-overlapping chunks from {} characters, longest chunk={}",
        chunks.len(),
        text.len(),
        max_chunk
    );
    chunks
}

/// Splits a given text into lines, accounting for different line break characters.
pub fn split_lines(text: &str) -> Vec<&str> {
    text.split("\n") 
        .flat_map(|line| line.split('\r')) // Additionally split by carriage return
        .collect() 
}

/// Fixes line breaks in a given text by removing carriage returns.
pub fn fix_line_breaks(text: &str) -> String {
    text.replace('\r', "")
}

/// Fixes apostrophes and normalizes characters in the given text.
pub fn fix_apostrophes(text: &str) -> String {
    let text = unidecode(text);
    text.replace("\\u2019", "'") // Replace the Unicode right single quotation mark with a regular apostrophe
}

/// Normalizes special characters in the input text by cleaning it.
pub fn normalize_special_characters(text: &str) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    let second_text = fix_line_breaks(text); 
    fix_apostrophes(&second_text) 
}

/// Reads a text file and optionally normalizes its special characters.
pub fn read_text_file(path_to_text_file: &str, normalize_special_characters_flag: bool) -> std::io::Result<String> {
    let text = fs::read_to_string(path_to_text_file)?;
    if normalize_special_characters_flag {
        let normalized_text = normalize_special_characters(&text);
        Ok(normalized_text)
    } else {
        Ok(text)
    }
}

/// Extracts text chunks from the provided document URI and text, based on specified parameters.
pub fn get_text_chunks(
    document_uri: &str,
    document_text: &str,
    chunk_length_characters: usize,
    overlap_factor: &mut usize,
    normalize_special_characters_flag: bool,
) -> Vec<DataBundle> {
    // Get the file name from the document URI
    let file_name = Path::new(document_uri)
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    // Create a DataSource instance for the document
    let source = DataSource {
        source_type: DataSourceType::Document,
        name: file_name.clone(),
        location: document_uri.to_string(),
        version: String::from("1.0"), 
    };

    // Set overlap_factor to 0 if it's less than 2
    if *overlap_factor < 2 {
        *overlap_factor = 0;
    }

    // Normalize document text if the flag is set
    let document_text = if normalize_special_characters_flag {
        normalize_special_characters(document_text)
    } else {
        document_text.to_string()
    };

    // If chunk length is zero or negative, return the entire document as a single chunk
    if chunk_length_characters <= 0 {
        let data_item = Data {
            value: DataValue::Text(document_text),
            id: "1".to_string(), 
            data_type: DataType::Text,
        };

        return vec![DataBundle::new(
            vec![data_item],
            ErrorCode::Undefined,
            Some(source),
        )];
    }
    // Create chunk strings based on overlap factor
    let mut chunk_strings;

    if *overlap_factor == 0 {
        chunk_strings = create_nonoverlapping_chunks(&document_text, chunk_length_characters);
    } else {
        let adjusted_chunk_length =
            (chunk_length_characters as f64 / *overlap_factor as f64).round() as usize;
        let small_chunks = create_nonoverlapping_chunks(&document_text, adjusted_chunk_length);
        chunk_strings = Vec::new();

        for i in (0..small_chunks.len()).step_by(*overlap_factor - 1) {
            let merged_chunk = _merge_chunks(&small_chunks, i, *overlap_factor);
            chunk_strings.push(merged_chunk);
        }

        println!(
            "Merged {} non-overlapping chunks into {} chunks with 1/{:?} overlap.",
            small_chunks.len(),
            chunk_strings.len(),
            overlap_factor
        );
    }

    // Create DataBundle instances for each chunk and return them in a vector
    let mut chunks = Vec::new();

    for (i, chunk_string) in chunk_strings.iter().enumerate() {
        let chunk = DataBundle::new(
            vec![Data {
                value: DataValue::Text(chunk_string.clone()),
                id: (i + 1).to_string(),
                data_type: DataType::Text,
            }],
            ErrorCode::Undefined,
            Some(source.clone()),
        );
        chunks.push(chunk);
    }

    println!(
        "Created {} chunks for {}[{}] ({} chars, overlap={})",
        chunks.len(),
        document_uri,
        document_text.len(),
        chunk_length_characters,
        overlap_factor
    );

    chunks
}

