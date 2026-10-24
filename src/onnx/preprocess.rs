//! Image patching for the ONNX backend (`preprocess.py`).

use fast_image_resize::{
    images::Image, FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
};
use half::f16;
use image::{DynamicImage, RgbImage};
use rayon::prelude::*;

use crate::Result;

/// Side length of every patch fed to the vision encoder.
pub(crate) const PATCH_SIZE: usize = 378;
const PATCH_LEN: usize = 3 * PATCH_SIZE * PATCH_SIZE;
const TEMPLATES: [(usize, usize); 3] = [(1, 2), (2, 1), (2, 2)];

/// Chooses the `(rows, cols)` patch template for an image of `width × height`.
pub(crate) fn select_template(width: u32, height: u32) -> (usize, usize) {
    if (width.max(height) as f64) < PATCH_SIZE as f64 * 1.4 {
        return (1, 1);
    }
    let aspect = width as f64 / height as f64;
    let mut best = TEMPLATES[0];
    let mut best_diff = f64::INFINITY;
    for t in TEMPLATES {
        let diff = (t.1 as f64 / t.0 as f64 - aspect).abs();
        if diff < best_diff {
            best = t;
            best_diff = diff;
        }
    }
    best
}

/// Splits an image into a global patch plus template crops, each resized to
/// 378×378 (bicubic), normalized to `[-1, 1]`, and laid out as `[n, 3, 378, 378]`.
///
/// Returns `(data, n, template)`.
pub(crate) fn create_patches(image: &DynamicImage) -> Result<(Vec<f16>, usize, (usize, usize))> {
    let rgb: RgbImage = image.to_rgb8();
    let (w, h) = rgb.dimensions();
    let template = select_template(w, h);

    // Crop rectangles (left, top, width, height); the first is the whole image.
    let mut crops = vec![(0u32, 0u32, w, h)];
    if template != (1, 1) {
        let (rows, cols) = (template.0 as u32, template.1 as u32);
        let (pw, ph) = (w / cols, h / rows);
        for r in 0..rows {
            for c in 0..cols {
                crops.push((c * pw, r * ph, pw, ph));
            }
        }
    }

    let lut: [f16; 256] =
        std::array::from_fn(|v| f16::from_f64((v as f64 / 255.0 - 0.5) / 0.5));
    let n = crops.len();
    let mut out = vec![f16::ZERO; n * PATCH_LEN];
    out.par_chunks_mut(PATCH_LEN)
        .zip(crops.par_iter())
        .try_for_each(|(dst, &(left, top, cw, ch))| -> Result<()> {
            let mut resized = Image::new(PATCH_SIZE as u32, PATCH_SIZE as u32, PixelType::U8x3);
            let opts = ResizeOptions::new()
                .resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom))
                .crop(left as f64, top as f64, cw as f64, ch as f64);
            Resizer::new().resize(&rgb, &mut resized, &opts)?;
            write_chw(resized.buffer(), dst, &lut);
            Ok(())
        })?;
    Ok((out, n, template))
}

/// Converts interleaved RGB bytes into normalized planar CHW.
fn write_chw(hwc: &[u8], dst: &mut [f16], lut: &[f16; 256]) {
    let plane = PATCH_SIZE * PATCH_SIZE;
    let (r, rest) = dst.split_at_mut(plane);
    let (g, b) = rest.split_at_mut(plane);
    for (i, px) in hwc.chunks_exact(3).enumerate() {
        r[i] = lut[px[0] as usize];
        g[i] = lut[px[1] as usize];
        b[i] = lut[px[2] as usize];
    }
}

/// Adaptive average pooling with numpy-reference semantics (`preprocess.py`):
/// fixed stride `in / out` and kernel `in - (out - 1) * stride`.
///
/// `x` is `[h, w, c]`; returns `[out_h, out_w, c]`.
pub(crate) fn adaptive_avg_pool2d_numpy(
    x: &[f16],
    h: usize,
    w: usize,
    c: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f16> {
    let (sh, sw) = (h / out_h, w / out_w);
    let (kh, kw) = (h - (out_h - 1) * sh, w - (out_w - 1) * sw);
    let inv = 1.0 / (kh * kw) as f32;
    let mut out = vec![f16::ZERO; out_h * out_w * c];
    out.par_chunks_mut(c).enumerate().for_each(|(idx, dst)| {
        let (i, j) = (idx / out_w, idx % out_w);
        let mut acc = vec![0f32; c];
        for y in i * sh..i * sh + kh {
            for xw in j * sw..j * sw + kw {
                let src = &x[(y * w + xw) * c..][..c];
                for (a, v) in acc.iter_mut().zip(src) {
                    *a += v.to_f32();
                }
            }
        }
        for (d, a) in dst.iter_mut().zip(&acc) {
            *d = f16::from_f32(a * inv);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_template_should_match_python_reference() {
        assert_eq!(select_template(400, 300), (1, 1));
        assert_eq!(select_template(1000, 500), (1, 2));
        assert_eq!(select_template(500, 1000), (2, 1));
        assert_eq!(select_template(1000, 1000), (2, 2));
        assert_eq!(select_template(1000, 800), (2, 2));
    }

    fn pool(input: &[f32], h: usize, w: usize, oh: usize, ow: usize) -> Vec<f32> {
        let x: Vec<f16> = input.iter().map(|&v| f16::from_f32(v)).collect();
        adaptive_avg_pool2d_numpy(&x, h, w, 1, oh, ow)
            .into_iter()
            .map(f16::to_f32)
            .collect()
    }

    #[test]
    fn numpy_pool_should_average_fixed_stride_windows() {
        let x: Vec<f32> = (0..24).map(|v| v as f32).collect();
        assert_eq!(pool(&x, 4, 6, 2, 3), vec![3.5, 5.5, 7.5, 15.5, 17.5, 19.5]);
        let x: Vec<f32> = (0..5).map(|v| v as f32).collect();
        assert_eq!(pool(&x, 5, 1, 3, 1), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn create_patches_should_add_crops_only_for_large_images() {
        let small = DynamicImage::new_rgb8(400, 300);
        let (data, n, t) = create_patches(&small).unwrap();
        assert_eq!((n, t, data.len()), (1, (1, 1), PATCH_LEN));
        let wide = DynamicImage::new_rgb8(1000, 500);
        let (data, n, t) = create_patches(&wide).unwrap();
        assert_eq!((n, t, data.len()), (3, (1, 2), 3 * PATCH_LEN));
        // Black pixels normalize to -1.
        assert_eq!(data[0], f16::from_f32(-1.0));
    }
}
