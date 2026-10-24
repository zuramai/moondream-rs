//! Coordinate and size heads (`region.py`).

use std::f64::consts::PI;

use candle_core::{Module, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder};

use super::{config::RegionConfig, mlp, Mlp};
use crate::Result;

pub(crate) struct RegionModel {
    coord_features: Tensor,
    coord_encoder: Linear,
    coord_decoder: Mlp,
    size_features: Tensor,
    size_encoder: Linear,
    size_decoder: Mlp,
}

impl RegionModel {
    /// `legacy` checkpoints store the Fourier feature matrices transposed.
    pub(crate) fn load(vb: VarBuilder, c: &RegionConfig, legacy: bool) -> Result<Self> {
        let features = |name: &str, inputs: usize, n: usize| -> Result<Tensor> {
            Ok(if legacy {
                vb.get((n, inputs), name)?.t()?.contiguous()?
            } else {
                vb.get((inputs, n), name)?
            })
        };
        Ok(Self {
            coord_features: features("coord_features", 1, c.coord_feat_dim / 2)?,
            coord_encoder: linear(c.coord_feat_dim, c.dim, vb.pp("coord_encoder"))?,
            coord_decoder: mlp(c.dim, c.inner_dim, c.coord_out_dim, vb.pp("coord_decoder"))?,
            size_features: features("size_features", 2, c.size_feat_dim / 2)?,
            size_encoder: linear(c.size_feat_dim, c.dim, vb.pp("size_encoder"))?,
            size_decoder: mlp(c.dim, c.inner_dim, c.size_out_dim, vb.pp("size_decoder"))?,
        })
    }

    fn fourier(x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let f = (x * (2.0 * PI))?.matmul(w)?;
        Ok(Tensor::cat(&[f.cos()?, f.sin()?], D::Minus1)?)
    }

    /// Embeds one coordinate in `[0, 1]` as `[1, 1, dim]`.
    pub(crate) fn encode_coordinate(&self, v: f32) -> Result<Tensor> {
        let w = &self.coord_features;
        let x = Tensor::new(&[[v]], w.device())?.to_dtype(w.dtype())?;
        Ok(self.coord_encoder.forward(&Self::fourier(&x, w)?)?.unsqueeze(0)?)
    }

    /// Coordinate bin logits `[coord_out]` (f32) from hidden `[1, 1, dim]`.
    pub(crate) fn decode_coordinate(&self, hidden: &Tensor) -> Result<Vec<f32>> {
        Ok(self.coord_decoder.forward(hidden)?.flatten_all()?.to_dtype(candle_core::DType::F32)?.to_vec1()?)
    }

    /// Embeds `(width, height)` as `[1, 1, dim]`.
    pub(crate) fn encode_size(&self, w: f32, h: f32) -> Result<Tensor> {
        let feats = &self.size_features;
        let x = Tensor::new(&[[w, h]], feats.device())?.to_dtype(feats.dtype())?;
        Ok(self.size_encoder.forward(&Self::fourier(&x, feats)?)?.unsqueeze(0)?)
    }

    /// Size bin logits as `(width_logits, height_logits)` (f32).
    pub(crate) fn decode_size(&self, hidden: &Tensor) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut v: Vec<f32> = self
            .size_decoder
            .forward(hidden)?
            .flatten_all()?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1()?;
        let h = v.split_off(v.len() / 2);
        Ok((v, h))
    }
}
