use std::{error::Error as E, fs::File};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Error: {0}")]
    Error(String),
    #[error("Io Error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("{0}")]
    SerializeError(#[from] serde_json::error::Error),
    #[error("Triton inference error: {0}")]
    TritonInferenceError(String),
    #[error("Invalid Triton GRPC URI: {0}")]
    InvalidUri(String, String),
    #[error("GRPC error: {0}")]
    GrpcError(String),
    #[error("Tonic error: {0}")]
    TonicError(#[from] tonic::Status),
    #[error("Tokenizer error: {0}")]
    TokenizerError(#[from] tokenizers::Error),
    #[error("Shape error: {0}")]
    ShapeError(#[from] ndarray::ShapeError),
    #[error("Ort Error: {0}")]
    OrtError(#[from] ort::error::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn new(message: &str) -> Self {
        Error::Error(message.to_string())
    }
    pub fn from_error(error: &dyn E) -> Self {
        Error::Error(error.to_string())
    }
    pub fn from_string(error: String) -> Self {
        Error::Error(error)
    }
}