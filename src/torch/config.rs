//! Model hyperparameters (`config.py` defaults).

use std::collections::HashMap;

/// Text transformer configuration.
#[derive(Clone, Debug)]
pub(crate) struct TextConfig {
    pub dim: usize,
    pub ff_dim: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub max_context: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    /// Positions (BOS + image tokens) attended bidirectionally.
    pub prefix_attn: usize,
}

/// Vision encoder configuration.
#[derive(Clone, Debug)]
pub(crate) struct VisionConfig {
    pub enc_dim: usize,
    pub enc_patch_size: usize,
    pub enc_n_layers: usize,
    pub enc_ff_dim: usize,
    pub enc_n_heads: usize,
    pub proj_out_dim: usize,
    pub crop_size: usize,
    pub in_channels: usize,
    pub max_crops: usize,
    pub overlap_margin: usize,
    pub proj_inner_dim: usize,
}

/// Region (coordinate/size) head configuration.
#[derive(Clone, Debug)]
pub(crate) struct RegionConfig {
    pub dim: usize,
    pub coord_feat_dim: usize,
    pub coord_out_dim: usize,
    pub size_feat_dim: usize,
    pub size_out_dim: usize,
    pub inner_dim: usize,
}

/// Prompt prefix/suffix token ids.
#[derive(Clone, Debug)]
pub(crate) struct PrefixSuffix {
    pub prefix: Vec<u32>,
    pub suffix: Vec<u32>,
}

/// Special tokens and prompt templates.
#[derive(Clone, Debug)]
pub(crate) struct TokenizerConfig {
    pub bos_id: u32,
    pub eos_id: u32,
    pub caption: Option<HashMap<&'static str, Vec<u32>>>,
    pub query: Option<PrefixSuffix>,
    pub detect: Option<PrefixSuffix>,
    pub point: Option<PrefixSuffix>,
}

/// Full model configuration.
#[derive(Clone, Debug)]
pub(crate) struct TorchConfig {
    pub text: TextConfig,
    pub vision: VisionConfig,
    pub region: RegionConfig,
    pub tokenizer: TokenizerConfig,
}

impl Default for TorchConfig {
    fn default() -> Self {
        let ps = |prefix: &[u32], suffix: &[u32]| {
            Some(PrefixSuffix {
                prefix: prefix.to_vec(),
                suffix: suffix.to_vec(),
            })
        };
        Self {
            text: TextConfig {
                dim: 2048,
                ff_dim: 8192,
                n_layers: 24,
                vocab_size: 51200,
                max_context: 2048,
                n_heads: 32,
                n_kv_heads: 32,
                prefix_attn: 730,
            },
            vision: VisionConfig {
                enc_dim: 1152,
                enc_patch_size: 14,
                enc_n_layers: 27,
                enc_ff_dim: 4304,
                enc_n_heads: 16,
                proj_out_dim: 2048,
                crop_size: 378,
                in_channels: 3,
                max_crops: 12,
                overlap_margin: 4,
                proj_inner_dim: 8192,
            },
            region: RegionConfig {
                dim: 2048,
                coord_feat_dim: 256,
                coord_out_dim: 1024,
                size_feat_dim: 512,
                size_out_dim: 2048,
                inner_dim: 8192,
            },
            tokenizer: TokenizerConfig {
                bos_id: 50256,
                eos_id: 50256,
                caption: Some(HashMap::from([
                    ("short", vec![198, 198, 16438, 8305, 25]),
                    ("normal", vec![198, 198, 24334, 1159, 25]),
                    ("long", vec![198, 198, 14617, 8305, 25]),
                ])),
                query: ps(&[198, 198, 24361, 25], &[198, 198, 33706, 25]),
                detect: ps(&[198, 198, 47504, 25], &[628]),
                point: ps(&[198, 198, 12727, 25], &[628]),
            },
        }
    }
}
