/// Represents the type of data source (e.g., Document, Image, Directory).
#[derive(Debug, Clone)]
pub enum SamplingMethod {
    NONE = 0,
    RANDOM = 1,
    HEAD = 2,
    TAIL = 3
}

