//! Text transformer with KV cache (`text.py`, `rope.py`).

use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder};

use super::{config::TextConfig, mlp, Mlp};
use crate::Result;

const LN_EPS: f64 = 1e-5;
/// Leading head dimensions rotated by RoPE.
const ROT_DIM: usize = 32;
const ROPE_THETA: f64 = 10000.0;

struct Block {
    ln: LayerNorm,
    qkv: Linear,
    proj: Linear,
    mlp: Mlp,
}

/// Per-layer `(k, v)` caches, each `[1, kv_heads, max_context, head_dim]`.
pub(crate) type KvCaches = Vec<(Tensor, Tensor)>;

pub(crate) struct TextModel {
    wte: Tensor,
    blocks: Vec<Block>,
    post_ln: LayerNorm,
    lm_head: Linear,
    /// RoPE tables `[max_context, ROT_DIM / 2]` in f32.
    cos: Tensor,
    sin: Tensor,
    n_heads: usize,
    n_kv_heads: usize,
    max_context: usize,
    prefix_attn: usize,
}

impl TextModel {
    pub(crate) fn load(vb: VarBuilder, c: &TextConfig) -> Result<Self> {
        let qkv_dim = c.dim + 2 * c.n_kv_heads * (c.dim / c.n_heads);
        let blocks = (0..c.n_layers)
            .map(|i| {
                let b = vb.pp(format!("blocks.{i}"));
                Ok(Block {
                    ln: layer_norm(c.dim, LN_EPS, b.pp("ln"))?,
                    qkv: linear(c.dim, qkv_dim, b.pp("attn.qkv"))?,
                    proj: linear(c.dim, c.dim, b.pp("attn.proj"))?,
                    mlp: mlp(c.dim, c.ff_dim, c.dim, b.pp("mlp"))?,
                })
            })
            .collect::<Result<_>>()?;

        let half = ROT_DIM / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| (1.0 / ROPE_THETA.powf((2 * i) as f64 / ROT_DIM as f64)) as f32)
            .collect();
        let mut cos = Vec::with_capacity(c.max_context * half);
        let mut sin = Vec::with_capacity(c.max_context * half);
        for t in 0..c.max_context {
            for &f in &inv_freq {
                let a = t as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        let dev = vb.device();
        Ok(Self {
            wte: vb.get((c.vocab_size, c.dim), "wte")?,
            blocks,
            post_ln: layer_norm(c.dim, LN_EPS, vb.pp("post_ln"))?,
            lm_head: linear(c.dim, c.vocab_size, vb.pp("lm_head"))?,
            cos: Tensor::from_vec(cos, (c.max_context, half), dev)?,
            sin: Tensor::from_vec(sin, (c.max_context, half), dev)?,
            n_heads: c.n_heads,
            n_kv_heads: c.n_kv_heads,
            max_context: c.max_context,
            prefix_attn: c.prefix_attn,
        })
    }

    /// Fresh zeroed caches for every layer.
    pub(crate) fn new_caches(&self, dtype: DType, device: &Device) -> Result<KvCaches> {
        let hd = self.wte.dim(1)? / self.n_heads;
        let shape = (1, self.n_kv_heads, self.max_context, hd);
        self.blocks
            .iter()
            .map(|_| Ok((Tensor::zeros(shape, dtype, device)?, Tensor::zeros(shape, dtype, device)?)))
            .collect()
    }

    /// Token embeddings `[1, len, dim]`.
    pub(crate) fn embed(&self, ids: &[u32]) -> Result<Tensor> {
        let ids = Tensor::new(ids, self.wte.device())?;
        Ok(self.wte.index_select(&ids, 0)?.unsqueeze(0)?)
    }

    /// Additive mask for rows `pos..pos+len` over keys `0..pos+len`, or `None`
    /// when every key is visible. Rows inside the image prefix attend to the
    /// whole prefix; all others are causal.
    fn mask(&self, pos: usize, len: usize, dtype: DType, dev: &Device) -> Result<Option<Tensor>> {
        if len <= 1 {
            return Ok(None);
        }
        let total = pos + len;
        let p = self.prefix_attn;
        let mut m = vec![0f32; len * total];
        for r in 0..len {
            let i = pos + r;
            for j in 0..total {
                if !(j <= i || (i < p && j < p)) {
                    m[r * total + j] = f32::NEG_INFINITY;
                }
            }
        }
        Ok(Some(Tensor::from_vec(m, (len, total), dev)?.to_dtype(dtype)?))
    }

    fn rope(&self, x: &Tensor, pos: usize, len: usize) -> Result<Tensor> {
        let hd = x.dim(D::Minus1)?;
        let rot = x.narrow(D::Minus1, 0, ROT_DIM)?.to_dtype(DType::F32)?.contiguous()?;
        let cos = self.cos.narrow(0, pos, len)?;
        let sin = self.sin.narrow(0, pos, len)?;
        let rot = candle_nn::rotary_emb::rope(&rot, &cos, &sin)?.to_dtype(x.dtype())?;
        let pass = x.narrow(D::Minus1, ROT_DIM, hd - ROT_DIM)?;
        Ok(Tensor::cat(&[&rot, &pass], D::Minus1)?)
    }

    fn attn(&self, b: &Block, x: &Tensor, cache: &(Tensor, Tensor), pos: usize, mask: Option<&Tensor>) -> Result<Tensor> {
        let (bsz, len, dim) = x.dims3()?;
        let hd = dim / self.n_heads;
        let qkv = b.qkv.forward(x)?;
        let heads = |offset: usize, n: usize| -> Result<Tensor> {
            Ok(qkv
                .narrow(D::Minus1, offset, n * hd)?
                .reshape((bsz, len, n, hd))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let q = self.rope(&heads(0, self.n_heads)?, pos, len)?;
        let k = self.rope(&heads(dim, self.n_kv_heads)?, pos, len)?.contiguous()?;
        let v = heads(dim + self.n_kv_heads * hd, self.n_kv_heads)?;

        let (k_cache, v_cache) = cache;
        k_cache.slice_set(&k, 2, pos)?;
        v_cache.slice_set(&v, 2, pos)?;
        let k = k_cache.narrow(2, 0, pos + len)?;
        let v = v_cache.narrow(2, 0, pos + len)?;

        let scores = (q * (1.0 / (hd as f64).sqrt()))?.matmul(&k.t()?)?.to_dtype(DType::F32)?;
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };
        let probs = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(x.dtype())?;
        let out = probs.matmul(&v)?.transpose(1, 2)?.reshape((bsz, len, dim))?;
        Ok(b.proj.forward(&out)?)
    }

    /// Runs the decoder over `x` `[1, len, dim]` at positions `pos..pos+len`,
    /// updating `caches`; returns hidden states before the final layer norm.
    pub(crate) fn forward(&self, x: &Tensor, caches: &KvCaches, pos: usize) -> Result<Tensor> {
        let len = x.dim(1)?;
        let mask = self.mask(pos, len, DType::F32, x.device())?;
        let mut x = x.clone();
        for (b, cache) in self.blocks.iter().zip(caches) {
            let l_in = b.ln.forward(&x)?;
            let attn = self.attn(b, &l_in, cache, pos, mask.as_ref())?;
            let mlp = b.mlp.forward(&l_in)?;
            x = ((x + attn)? + mlp)?;
        }
        Ok(x)
    }

    /// Logits `[vocab]` (f32) for the last position of `hidden`.
    pub(crate) fn logits(&self, hidden: &Tensor) -> Result<Vec<f32>> {
        let len = hidden.dim(1)?;
        let last = hidden.i((.., len - 1, ..))?;
        let logits = self.lm_head.forward(&self.post_ln.forward(&last)?)?;
        Ok(logits.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Result<TextModel> {
        let c = TextConfig {
            dim: 128,
            ff_dim: 32,
            n_layers: 1,
            vocab_size: 4,
            max_context: 8,
            n_heads: 2,
            n_kv_heads: 2,
            prefix_attn: 3,
        };
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        TextModel::load(vb, &c)
    }

    #[test]
    fn mask_should_be_bidirectional_in_prefix_and_causal_after() -> Result<()> {
        let m = tiny()?;
        let mask: Vec<Vec<f32>> = m
            .mask(0, 5, DType::F32, &Device::Cpu)?
            .expect("mask")
            .to_vec2()?;
        let visible: Vec<Vec<bool>> = mask
            .iter()
            .map(|r| r.iter().map(|v| v.is_finite()).collect())
            .collect();
        assert_eq!(visible[0], vec![true, true, true, false, false]);
        assert_eq!(visible[2], vec![true, true, true, false, false]);
        assert_eq!(visible[3], vec![true, true, true, true, false]);
        assert!(m.mask(4, 1, DType::F32, &Device::Cpu)?.is_none());
        Ok(())
    }
}
