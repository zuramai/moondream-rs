use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::ArrayD;
use ndarray_npy::read_npy;
use tokenizers::Tokenizer;
use super::config::{self, MoondreamConfig};
use crate::error::{Error, Result};
use super::engine::{self, Engine};

pub struct Moondream {
    vision_encoder: engine::Engine,
    vision_projection: engine::Engine,
    text_encoder: engine::Engine,
    text_decoder: engine::Engine,
    size_encoder: engine::Engine,
    size_decoder: engine::Engine,
    coord_encoder: engine::Engine,
    coord_decoder: engine::Engine,
    tokenizer: Tokenizer,
    initial_kv_cache: ArrayD<f32>,
    config: MoondreamConfig,
}

impl Moondream {
    pub fn from_path(model_path: &str, config_path: &str) -> Result<Self> {
        let path = Path::new(model_path);
        let initial_kv_cache = read_npy(&path.join("initial_kv_cache.npy"))
            .map_err(|e| Error::Error(e.to_string()))?;
        Ok(Self {
            vision_encoder: Engine::new(path.join("vision_encoder.onnx"))?,
            vision_projection: Engine::new(path.join("vision_projection.onnx"))?,
            config: MoondreamConfig::from_file(&path.join("config.json"))?,
            coord_decoder: Engine::new(path.join("coord_decoder.onnx"))?,
            coord_encoder: Engine::new(path.join("coord_encoder.onnx"))?,
            size_decoder: Engine::new(path.join("size_decoder.onnx"))?,
            size_encoder: Engine::new(path.join("size_encoder.onnx"))?,
            text_decoder: Engine::new(path.join("text_decoder.onnx"))?,
            text_encoder: Engine::new(path.join("text_encoder.onnx"))?,
            initial_kv_cache: initial_kv_cache,
            tokenizer: Tokenizer::from_file(path.join("tokenizer.json"))?
        })
    }
    pub fn encode_image() {

    }
    pub fn caption(&self) -> String {
        "".into()
    }
    pub fn query(&self) -> String {
        "".into()
    }
    pub fn detect(&self) -> String {
        "".into()
    }
    pub fn point(&self) -> String {
        "".into()
    }
}