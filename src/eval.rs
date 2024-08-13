use py2rs::text_chunker;

use std::collections::HashMap;
use std::error::Error;
use std::fmt
/// Represents the type of data source (e.g., Document, Image, Directory).
#[derive(Debug, Clone)]
enum SamplingMethod {
    NONE = 0,
    RANDOM = 1,
    HEAD = 2,
    TAIL = 3
}

// Placeholder types
struct BaseFileStore;
struct DataBundle;
struct Trace;
struct LocalStorage;

struct Dataset {
    records: Vec<HashMap<String, String>>,
    ground_truth_loader: Box<dyn Fn(&BaseFileStore, &str, &HashMap<String, String>) -> DataBundle>,
    context_loader: Option<Box<dyn Fn(&BaseFileStore, &str, &HashMap<String, String>) -> DataBundle>>,
    batch_context_loader: Option<Box<dyn Fn(&BaseFileStore, &str, &HashMap<String, String>, Trace) -> Vec<DataBundle>>>,
    metadata_loader: Option<Box<dyn Fn(&str, &HashMap<String, String>) -> HashMap<String, String>>>,
    result_loader: Option<Box<dyn Fn(&str, &HashMap<String, String>) -> DataBundle>>,
    storage: LocalStorage,
    task_id: String,
    trace: Trace,
}

impl Dataset {
    pub fn new(
        records: Vec<HashMap<String, String>>,
        ground_truth_loader: Box<dyn Fn(&BaseFileStore, &str, &HashMap<String, String>) -> DataBundle>,
        context_loader: Option<Box<dyn Fn(&BaseFileStore, &str, &HashMap<String, String>) -> DataBundle>>,
        batch_context_loader: Option<Box<dyn Fn(&BaseFileStore, &str, &HashMap<String, String>, Trace) -> Vec<DataBundle>>>,
        metadata_loader: Option<Box<dyn Fn(&str, &HashMap<String, String>) -> HashMap<String, String>>>,
        result_loader: Option<Box<dyn Fn(&str, &HashMap<String, String>) -> DataBundle>>,
        storage: LocalStorage,
        task_id: String,
        trace: Trace,
    ) -> Result<Self, Box<dyn Error>> {
        if ground_truth_loader.is_none() {
            return Err(Box::new(InvalidArgsError("ground_truth_loader must be provided.".to_string())));
        }
        if context_loader.is_none() && batch_context_loader.is_none() {
            return Err(Box::new(InvalidArgsError("Either context_loader or batch_context_loader must be provided.".to_string())));
        }
        if context_loader.is_some() && batch_context_loader.is_some() {
            return Err(Box::new(InvalidArgsError("Only one of context_loader or batch_context_loader can be provided.".to_string())));
        }
        
        Ok(Dataset {
            records,
            ground_truth_loader,
            context_loader,
            batch_context_loader,
            metadata_loader,
            result_loader,
            storage,
            task_id,
            trace,
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (Option<DataBundle>, DataBundle, HashMap<String, String>, Option<DataBundle>)> {
        self.records.iter().map(|item| {
            let ground_truth = (self.ground_truth_loader)(&self.storage, &self.task_id, item);
            let metadata = self.metadata_loader.as_ref().map(|f| f(&self.task_id, item)).unwrap_or_default();
            let cached_result = self.result_loader.as_ref().map(|f| f(&self.task_id, item));

            let context = match &self.context_loader {
                Some(loader) => Some(loader(&self.storage, &self.task_id, item)),
                None => {
                    match &self.batch_context_loader {
                        Some(loader) => Some(loader(&self.storage, &self.task_id, item, self.trace)),
                        None => None,
                    }
                }
            };

            (context, ground_truth, metadata, cached_result)
        })
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn get_sample(&self, sampling_method: SamplingMethod, sample_size: usize) -> Result<Self, Box<dyn Error>> {
        if sample_size < 1 {
            return Err(Box::new(InvalidArgsError(format!("sample_size must be >= 1, received {}", sample_size))));
        }
        let actual_size = self.records.len();
        if sample_size > actual_size {
            println!("Warning: requested sample_size={} while dataset has only {} rows.", sample_size, actual_size);
        }
        
        let mut sample = self.records.clone();
        match sampling_method {
            SamplingMethod::RANDOM => {
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                sample.shuffle(&mut rng);
                sample.truncate(sample_size);
            }
            SamplingMethod::HEAD => {
                sample.truncate(sample_size);
            }
            SamplingMethod::TAIL => {
                let start = actual_size.saturating_sub(sample_size);
                sample = sample[start..].to_vec();
            }
            SamplingMethod::NONE => {
                return Ok(self.clone());
            }
        }

        Ok(Dataset::new(
            sample,
            self.ground_truth_loader.clone(),
            self.context_loader.clone(),
            self.batch_context_loader.clone(),
            self.metadata_loader.clone(),
            self.result_loader.clone(),
            self.storage,
            self.task_id.clone(),
            self.trace,
        )?)
    }
}

// Placeholder for the necessary implementation for cloning functional callbacks
impl Clone for Dataset {
    fn clone(&self) -> Self {
        Dataset {
            records: self.records.clone(),
            ground_truth_loader: self.ground_truth_loader.clone(),
            context_loader: self.context_loader.clone(),
            batch_context_loader: self.batch_context_loader.clone(),
            metadata_loader: self.metadata_loader.clone(),
            result_loader: self.result_loader.clone(),
            storage: self.storage.clone(),
            task_id: self.task_id.clone(),
            trace: self.trace,
        }
    }
}