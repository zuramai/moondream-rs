//! Rust port of the Moondream vision-language model client.
//!
//! Backends: [`OnnxVL`] (`.mf` ONNX bundles or unpacked model directories) and
//! [`TorchVL`] (`.safetensors` weights run with candle). [`vl`] picks one by path.

#![deny(missing_docs)]

mod error;
mod moonfile;
pub mod onnx;
pub mod server;
pub mod torch;
mod types;

use std::path::Path;

pub use error::{Error, Result};
pub use onnx::OnnxVL;
pub use torch::TorchVL;
pub use types::*;

/// Loads a model, choosing the backend from the path.
///
/// Directories and `.mf` / `.mf.gz` files load with [`OnnxVL`];
/// `.safetensors` files load with [`TorchVL`].
pub fn vl(model: impl AsRef<Path>) -> Result<Box<dyn Vlm>> {
    let path = model.as_ref();
    if path.is_dir() {
        return Ok(Box::new(OnnxVL::from_path(path)?));
    }
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    match path.extension().and_then(|e| e.to_str()) {
        Some("safetensors") => Ok(Box::new(TorchVL::new(path)?)),
        Some("mf") => Ok(Box::new(OnnxVL::from_path(path)?)),
        _ if name.ends_with(".mf.gz") => Ok(Box::new(OnnxVL::from_path(path)?)),
        _ => Err(Error::UnsupportedModelFiletype),
    }
}
