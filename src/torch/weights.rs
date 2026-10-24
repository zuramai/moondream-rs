//! Safetensors loading with key normalization (`weights.py`).

use std::{collections::HashMap, path::Path};

use candle_core::{safetensors::MmapedSafetensors, DType, Device};
use candle_nn::VarBuilder;

use crate::Result;

const VISUAL: &str = "vision_encoder.encoder.model.visual";

/// Weights plus whether they use the legacy (pre-2025) key layout, in which
/// `region.coord_features` / `region.size_features` are stored transposed.
pub(crate) struct Weights {
    pub vb: VarBuilder<'static>,
    pub legacy: bool,
}

/// Opens `path` and exposes tensors under canonical names (`vision.*`, `text.*`,
/// `region.*`), cast to `dtype` on `device`.
pub(crate) fn load(path: &Path, dtype: DType, device: &Device) -> Result<Weights> {
    // SAFETY: the weights file is read-only and not modified while mapped.
    let names: Vec<String> = unsafe { MmapedSafetensors::new(path)? }
        .tensors()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    // SAFETY: as above.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? };

    let has = |k: &str| names.iter().any(|n| n == k);
    if has("model.vision.blocks.0.attn.proj.bias") {
        let vb = vb.rename_f(|n: &str| format!("model.{n}"));
        return Ok(Weights { vb, legacy: false });
    }
    if has("vision.blocks.0.attn.proj.bias") {
        return Ok(Weights { vb, legacy: false });
    }
    // Legacy checkpoints may carry `._orig_mod` from torch.compile.
    let originals: HashMap<String, String> = names
        .iter()
        .map(|n| (n.replace("._orig_mod", ""), n.clone()))
        .collect();
    let vb = vb.rename_f(move |n: &str| {
        let legacy = legacy_name(n);
        originals.get(&legacy).cloned().unwrap_or(legacy)
    });
    Ok(Weights { vb, legacy: true })
}

/// Maps a canonical tensor name to the legacy checkpoint name (`_load_weights`).
fn legacy_name(name: &str) -> String {
    match name {
        "vision.pos_emb" => return format!("{VISUAL}.pos_embed"),
        "text.wte" => return "text_model.transformer.embd.wte.weight".into(),
        "region.coord_features" => return "region_model.coordinate_features.weight".into(),
        "region.size_features" => return "region_model.size_features.weight".into(),
        _ => {}
    }
    let prefixes: [(&str, String); 9] = [
        ("vision.patch_emb.", format!("{VISUAL}.patch_embed.linear.")),
        ("vision.post_ln.", format!("{VISUAL}.norm.")),
        ("vision.proj_mlp.", "vision_encoder.projection.mlp.".into()),
        ("text.post_ln.", "text_model.lm_head.ln.".into()),
        ("text.lm_head.", "text_model.lm_head.linear.".into()),
        ("region.coord_encoder.", "region_model.coordinate_encoder.".into()),
        ("region.coord_decoder.", "region_model.coordinate_decoder.".into()),
        ("region.size_encoder.", "region_model.size_encoder.".into()),
        ("region.size_decoder.", "region_model.size_decoder.".into()),
    ];
    for (canonical, legacy) in &prefixes {
        if let Some(rest) = name.strip_prefix(canonical) {
            return format!("{legacy}{rest}");
        }
    }
    if let Some(rest) = name.strip_prefix("vision.blocks.") {
        let rest = rest.replacen(".ln1.", ".norm1.", 1).replacen(".ln2.", ".norm2.", 1);
        return format!("{VISUAL}.blocks.{rest}");
    }
    if let Some(rest) = name.strip_prefix("text.blocks.") {
        let rest = rest
            .replacen(".attn.qkv.", ".mixer.Wqkv.", 1)
            .replacen(".attn.proj.", ".mixer.out_proj.", 1);
        return format!("text_model.transformer.h.{rest}");
    }
    name.to_string()
}
