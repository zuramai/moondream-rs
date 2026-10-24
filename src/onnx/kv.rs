//! KV cache buffer for the ONNX text decoder.
//!
//! Layout `[rows, max, head_dim]` where `rows = layers * 2 * batch * heads`;
//! this matches the decoder's `[24, 2, 1, H, seq, 64]` tensors row by row.

use half::f16;

use super::OnnxEncodedImage;

/// Maximum context length of the text decoder.
pub(crate) const MAX_CONTEXT: usize = 2048;

pub(crate) struct KvBuffer {
    data: Vec<f16>,
    rows: usize,
    head_dim: usize,
    /// Leading dims of the decoder's kv tensor (all but `seq` and `head_dim`).
    lead: [usize; 4],
    scratch: Vec<f16>,
}

impl KvBuffer {
    /// Copies the encoded image's cache into a zero-padded `MAX_CONTEXT` buffer.
    pub(crate) fn from_encoded(enc: &OnnxEncodedImage) -> Self {
        let (rows, hd, pos) = (enc.rows, enc.head_dim, enc.pos);
        let mut data = vec![f16::ZERO; rows * MAX_CONTEXT * hd];
        for (dst, src) in data
            .chunks_exact_mut(MAX_CONTEXT * hd)
            .zip(enc.kv_cache.chunks_exact(pos * hd))
        {
            dst[..pos * hd].copy_from_slice(src);
        }
        let s = enc.kv_shape;
        Self {
            data,
            rows,
            head_dim: hd,
            lead: [s[0], s[1], s[2], s[3]],
            scratch: Vec::with_capacity(rows * MAX_CONTEXT * hd),
        }
    }

    /// Returns the first `pos` positions as a contiguous `[.., pos, head_dim]` tensor.
    pub(crate) fn prefix(&mut self, pos: usize) -> ([usize; 6], &[f16]) {
        let row = pos * self.head_dim;
        self.scratch.clear();
        for src in self.data.chunks_exact(MAX_CONTEXT * self.head_dim) {
            self.scratch.extend_from_slice(&src[..row]);
        }
        let [a, b, c, d] = self.lead;
        ([a, b, c, d, pos, self.head_dim], &self.scratch)
    }

    /// Writes an `[.., n, head_dim]` update at positions `pos..pos + n`.
    pub(crate) fn write(&mut self, pos: usize, n: usize, update: &[f16]) {
        let hd = self.head_dim;
        debug_assert_eq!(update.len(), self.rows * n * hd);
        for (dst, src) in self
            .data
            .chunks_exact_mut(MAX_CONTEXT * hd)
            .zip(update.chunks_exact(n * hd))
        {
            dst[pos * hd..(pos + n) * hd].copy_from_slice(src);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_prefix_should_return_rows_in_order() {
        let enc = OnnxEncodedImage {
            pos: 1,
            kv_cache: vec![f16::from_f32(1.0), f16::from_f32(2.0)],
            rows: 2,
            head_dim: 1,
            kv_shape: [1, 2, 1, 1, 1, 1],
        };
        let mut kv = KvBuffer::from_encoded(&enc);
        kv.write(1, 1, &[f16::from_f32(3.0), f16::from_f32(4.0)]);
        let (shape, data) = kv.prefix(2);
        assert_eq!(shape, [1, 2, 1, 1, 2, 1]);
        let v: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
        assert_eq!(v, vec![1.0, 3.0, 2.0, 4.0]);
    }
}
