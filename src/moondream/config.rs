use std::{fs::File, path::Path};

use serde::Deserialize;
use crate::error::{Error, Result};

#[derive(Deserialize)]
pub struct MoondreamConfig {
    pub model_version: i32,
    pub templates: TemplateConfig,
    pub special_tokens: SpecialTokensConfig,
}

impl MoondreamConfig {
    pub fn from_file(p: &Path) -> Result<Self> {
        let file = File::open(p)?;
        let json: MoondreamConfig = serde_json::from_reader(file)?;

        Ok(MoondreamConfig { 
            model_version: json.model_version,
            templates: json.templates,
            special_tokens: json.special_tokens,
        })
    }
}

#[derive(Deserialize)]
pub struct SpecialTokensConfig {
    pub bos: usize,
    pub eos: usize,
    pub coord: usize,
    pub size: usize,
}

#[derive(Deserialize)]
pub struct TemplateConfig {
    pub caption: CaptionConfig,
    pub query: PrefixSuffixConfig,
    pub detect: PrefixSuffixConfig,
    pub point: Option<PrefixSuffixConfig>,
}
#[derive(Deserialize)]
pub struct CaptionConfig {
    pub short: Vec<i64>,
    pub normal: Vec<i64>
}
#[derive(Deserialize)]
pub struct PrefixSuffixConfig {
    pub prefix: Vec<u32>,
    pub suffix: Vec<u32>
}