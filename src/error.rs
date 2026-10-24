//! Error type shared by every backend.

/// Errors returned by this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Filesystem or stream failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// JSON (de)serialization failure.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// ONNX Runtime failure.
    #[error(transparent)]
    Ort(#[from] ort::Error),
    /// Candle tensor failure.
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    /// Tokenizer failure.
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    /// Image decoding failure.
    #[error(transparent)]
    Image(#[from] image::ImageError),
    /// Image resizing failure.
    #[error(transparent)]
    Resize(#[from] fast_image_resize::ResizeError),
    /// The model file or directory is malformed or incomplete.
    #[error("{0}")]
    InvalidModel(String),
    /// The model path has an unsupported extension.
    #[error("Unsupported model filetype. Please use a .safetensors model for GPU use or .mf model for CPU use.")]
    UnsupportedModelFiletype,
    /// The model does not support the requested capability.
    #[error("{0}")]
    Unsupported(String),
    /// An [`crate::EncodedImage`] from one backend was passed to another.
    #[error("encoded image was produced by a different backend")]
    EncodedImageMismatch,
}

/// Result alias using [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;
