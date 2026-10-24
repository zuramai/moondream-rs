//! Vision encoder and projection (`vision.py`).

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder};

use super::{config::VisionConfig, mlp, Mlp};
use crate::Result;

const LN_EPS: f64 = 1e-5;
/// Crops encoded per forward pass; bounds attention score memory.
const CROP_BATCH: usize = 4;

struct Block {
    ln1: LayerNorm,
    qkv: Linear,
    proj: Linear,
    ln2: LayerNorm,
    mlp: Mlp,
}

pub(crate) struct VisionModel {
    patch_emb: Linear,
    pos_emb: Tensor,
    blocks: Vec<Block>,
    post_ln: LayerNorm,
    proj: Mlp,
    n_heads: usize,
    grid: usize,
}

impl VisionModel {
    pub(crate) fn load(vb: VarBuilder, c: &VisionConfig) -> Result<Self> {
        let patch_dim = c.enc_patch_size * c.enc_patch_size * c.in_channels;
        let grid = c.crop_size / c.enc_patch_size;
        let blocks = (0..c.enc_n_layers)
            .map(|i| {
                let b = vb.pp(format!("blocks.{i}"));
                Ok(Block {
                    ln1: layer_norm(c.enc_dim, LN_EPS, b.pp("ln1"))?,
                    qkv: linear(c.enc_dim, 3 * c.enc_dim, b.pp("attn.qkv"))?,
                    proj: linear(c.enc_dim, c.enc_dim, b.pp("attn.proj"))?,
                    ln2: layer_norm(c.enc_dim, LN_EPS, b.pp("ln2"))?,
                    mlp: mlp(c.enc_dim, c.enc_ff_dim, c.enc_dim, b.pp("mlp"))?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            patch_emb: linear(patch_dim, c.enc_dim, vb.pp("patch_emb"))?,
            pos_emb: vb.get((1, grid * grid, c.enc_dim), "pos_emb")?,
            blocks,
            post_ln: layer_norm(c.enc_dim, LN_EPS, vb.pp("post_ln"))?,
            proj: mlp(2 * c.enc_dim, c.proj_inner_dim, c.proj_out_dim, vb.pp("proj_mlp"))?,
            n_heads: c.enc_n_heads,
            grid,
        })
    }

    fn attn(&self, b: &Block, x: &Tensor) -> Result<Tensor> {
        let (bsz, len, dim) = x.dims3()?;
        let hd = dim / self.n_heads;
        let qkv = b.qkv.forward(x)?;
        let split = |i: usize| -> Result<Tensor> {
            Ok(qkv
                .narrow(D::Minus1, i * dim, dim)?
                .reshape((bsz, len, self.n_heads, hd))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let (q, k, v) = (split(0)?, split(1)?, split(2)?);
        let scores = (q.matmul(&k.t()?)? * (1.0 / (hd as f64).sqrt()))?;
        let probs = candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?
            .to_dtype(x.dtype())?;
        let out = probs.matmul(&v)?.transpose(1, 2)?.reshape((bsz, len, dim))?;
        Ok(b.proj.forward(&out)?)
    }

    /// Encodes `[n, 729, 588]` patches to `[n, 729, enc_dim]`.
    pub(crate) fn encode(&self, patches: &Tensor) -> Result<Tensor> {
        let n = patches.dim(0)?;
        let mut outs = Vec::with_capacity(n.div_ceil(CROP_BATCH));
        for start in (0..n).step_by(CROP_BATCH) {
            let chunk = patches.narrow(0, start, CROP_BATCH.min(n - start))?;
            let mut x = self.patch_emb.forward(&chunk)?.broadcast_add(&self.pos_emb)?;
            for b in &self.blocks {
                x = (&x + self.attn(b, &b.ln1.forward(&x)?)?)?;
                x = (&x + b.mlp.forward(&b.ln2.forward(&x)?)?)?;
            }
            outs.push(self.post_ln.forward(&x)?);
        }
        Ok(Tensor::cat(&outs, 0)?)
    }

    /// Pools reconstructed local features `[H, W, C]` to the crop grid and
    /// projects `[global, pooled]` to text embeddings `[729, proj_out]`.
    pub(crate) fn project(&self, global: &Tensor, reconstructed: &Tensor) -> Result<Tensor> {
        let pooled = adaptive_avg_pool2d(reconstructed, self.grid, self.grid)?;
        let (_, _, c) = pooled.dims3()?;
        let pooled = pooled.reshape((self.grid * self.grid, c))?;
        let features = Tensor::cat(&[global, &pooled], D::Minus1)?;
        Ok(self.proj.forward(&features)?)
    }
}

/// `[out, in]` averaging matrix with torch adaptive pooling windows
/// `[floor(i*in/out), ceil((i+1)*in/out))`.
fn pool_matrix(input: usize, output: usize) -> Vec<f32> {
    let mut m = vec![0f32; output * input];
    for i in 0..output {
        let start = i * input / output;
        let end = ((i + 1) * input).div_ceil(output);
        let w = 1.0 / (end - start) as f32;
        m[i * input + start..i * input + end].fill(w);
    }
    m
}

/// Torch-semantics adaptive average pooling of `[H, W, C]` to `[out_h, out_w, C]`,
/// as two matmuls so it runs on any device.
pub(crate) fn adaptive_avg_pool2d(x: &Tensor, out_h: usize, out_w: usize) -> Result<Tensor> {
    let (h, w, c) = x.dims3()?;
    let dev: &Device = x.device();
    let ph = Tensor::from_vec(pool_matrix(h, out_h), (out_h, h), dev)?.to_dtype(x.dtype())?;
    let pw = Tensor::from_vec(pool_matrix(w, out_w), (out_w, w), dev)?.to_dtype(x.dtype())?;
    let y = ph.matmul(&x.reshape((h, w * c))?)?.reshape((out_h, w, c))?;
    Ok(pw.broadcast_left(out_h)?.contiguous()?.matmul(&y)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn torch_pool_should_use_overlapping_windows() -> Result<()> {
        let x = Tensor::arange(0f32, 5.0, &Device::Cpu)?.reshape((5, 1, 1))?;
        let y: Vec<f32> = adaptive_avg_pool2d(&x, 3, 1)?.flatten_all()?.to_vec1()?;
        assert_eq!(y, vec![0.5, 2.0, 3.5]);
        Ok(())
    }
}
