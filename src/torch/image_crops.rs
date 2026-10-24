//! Overlapping crop extraction and feature reconstruction (`image_crops.py`).

use candle_core::Tensor;
use fast_image_resize::{images::Image, FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use half::f16;
use image::RgbImage;
use rayon::prelude::*;

use crate::Result;

/// Crop side length in pixels.
pub(crate) const CROP: usize = 378;
/// Patch side length in pixels.
pub(crate) const PATCH: usize = 14;
/// Patches per crop side.
pub(crate) const GRID: usize = CROP / PATCH;
/// Values per flattened patch (`3 * 14 * 14`).
pub(crate) const PATCH_DIM: usize = 3 * PATCH * PATCH;

/// Optimal `(rows, cols)` tile count covering `height × width` with `crop_size` windows.
pub(crate) fn select_tiling(height: i64, width: i64, crop_size: i64, max_crops: i64) -> (usize, usize) {
    if height <= crop_size || width <= crop_size {
        return (1, 1);
    }
    let (h, w, crop, max) = (height as f64, width as f64, crop_size as f64, max_crops as f64);
    let min_h = (h / crop).ceil();
    let min_w = (w / crop).ceil();
    if min_h * min_w > max {
        let ratio = (max / (min_h * min_w)).sqrt();
        return (
            ((min_h * ratio).floor() as usize).max(1),
            ((min_w * ratio).floor() as usize).max(1),
        );
    }
    let mut h_tiles = (max * h / w).sqrt().floor().max(min_h);
    let mut w_tiles = (max * w / h).sqrt().floor().max(min_w);
    if h_tiles * w_tiles > max {
        if w_tiles > h_tiles {
            w_tiles = (max / h_tiles).floor();
        } else {
            h_tiles = (max / w_tiles).floor();
        }
    }
    ((h_tiles as usize).max(1), (w_tiles as usize).max(1))
}

fn resize(src: &RgbImage, width: usize, height: usize) -> Result<Image<'static>> {
    let mut dst = Image::new(width as u32, height as u32, PixelType::U8x3);
    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
    Resizer::new().resize(src, &mut dst, &opts)?;
    Ok(dst)
}

/// Global crop plus overlapping local crops, normalized to `[-1, 1]` and laid
/// out as patches: `[n, 729, 588]` with patch features ordered `(channel, row, col)`.
///
/// Returns `(data, n, tiling)`.
pub(crate) fn overlap_crop_image(
    rgb: &RgbImage,
    max_crops: usize,
    overlap_margin: usize,
) -> Result<(Vec<f16>, usize, (usize, usize))> {
    let (w, h) = (rgb.width() as i64, rgb.height() as i64);
    let margin = PATCH * overlap_margin;
    let window = (GRID - 2 * overlap_margin) * PATCH;
    let total_margin = 2 * margin as i64;
    let tiling = select_tiling(h - total_margin, w - total_margin, window as i64, max_crops as i64);

    let target_h = tiling.0 * window + 2 * margin;
    let target_w = tiling.1 * window + 2 * margin;
    let (resized, global) = rayon::join(
        || resize(rgb, target_w, target_h),
        || resize(rgb, CROP, CROP),
    );
    let (resized, global) = (resized?, global?);

    // Torch normalizes op by op in f16: ((v / 255) - 0.5) / 0.5.
    let lut: [f16; 256] = std::array::from_fn(|v| {
        ((f16::from_f32(v as f32) / f16::from_f32(255.0)) - f16::from_f32(0.5)) / f16::from_f32(0.5)
    });
    let n = tiling.0 * tiling.1 + 1;
    let crop_len = GRID * GRID * PATCH_DIM;
    let mut out = vec![lut[0]; n * crop_len];
    out.par_chunks_mut(crop_len).enumerate().for_each(|(idx, dst)| {
        let (src, src_w, src_h, y0, x0) = if idx == 0 {
            (global.buffer(), CROP, CROP, 0, 0)
        } else {
            let (i, j) = ((idx - 1) / tiling.1, (idx - 1) % tiling.1);
            (resized.buffer(), target_w, target_h, i * window, j * window)
        };
        let rows = CROP.min(src_h - y0);
        let cols = CROP.min(src_w - x0);
        for y in 0..rows {
            let line = &src[((y0 + y) * src_w + x0) * 3..][..cols * 3];
            let (py, iy) = (y / PATCH, y % PATCH);
            for (x, px) in line.chunks_exact(3).enumerate() {
                let base = (py * GRID + x / PATCH) * PATCH_DIM + iy * PATCH + x % PATCH;
                for (c, &v) in px.iter().enumerate() {
                    dst[base + c * PATCH * PATCH] = lut[v as usize];
                }
            }
        }
    });
    Ok((out, n, tiling))
}

/// Stitches local crop features `[k, g, g, C]` into `[(g-2m)*th + 2m, (g-2m)*tw + 2m, C]`,
/// keeping outer margins only at the image border.
pub(crate) fn reconstruct_from_crops(
    crops: &Tensor,
    tiling: (usize, usize),
    overlap_margin: usize,
) -> Result<Tensor> {
    let (_, ch, cw, _) = crops.dims4()?;
    let (th, tw) = tiling;
    let m = overlap_margin;
    let mut rows = Vec::with_capacity(th);
    for ty in 0..th {
        let y_start = if ty == 0 { 0 } else { m };
        let y_end = if ty == th - 1 { ch } else { ch - m };
        let mut pieces = Vec::with_capacity(tw);
        for tx in 0..tw {
            let x_start = if tx == 0 { 0 } else { m };
            let x_end = if tx == tw - 1 { cw } else { cw - m };
            let crop = crops.get(ty * tw + tx)?;
            pieces.push(
                crop.narrow(0, y_start, y_end - y_start)?
                    .narrow(1, x_start, x_end - x_start)?,
            );
        }
        rows.push(Tensor::cat(&pieces, 1)?);
    }
    Ok(Tensor::cat(&rows, 0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn select_tiling_should_match_python_reference() {
        let tiling = |h: i64, w: i64| select_tiling(h - 112, w - 112, 266, 12);
        assert_eq!(tiling(300, 300), (1, 1));
        assert_eq!(tiling(768, 1024), (3, 4));
        assert_eq!(tiling(1080, 1920), (2, 4));
        assert_eq!(tiling(4000, 300), (1, 1));
        assert_eq!(tiling(5000, 5000), (3, 3));
        assert_eq!(tiling(640, 480), (4, 2));
    }

    #[test]
    fn reconstruct_should_keep_margins_only_at_borders() -> Result<()> {
        // Two 4x4 crops side by side with margin 1; value = crop index.
        let crops = Tensor::stack(
            &[
                Tensor::zeros((4, 4, 1), DType::F32, &Device::Cpu)?,
                Tensor::ones((4, 4, 1), DType::F32, &Device::Cpu)?,
            ],
            0,
        )?;
        let out = reconstruct_from_crops(&crops, (1, 2), 1)?;
        assert_eq!(out.dims(), &[4, 6, 1]);
        let row: Vec<f32> = out.get(0)?.flatten_all()?.to_vec1()?;
        assert_eq!(row, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn overlap_crop_image_should_lay_out_patches_channel_major() -> Result<()> {
        let mut img = RgbImage::new(378, 378);
        img.put_pixel(15, 1, image::Rgb([255, 0, 0]));
        let (data, n, tiling) = overlap_crop_image(&img, 12, 4)?;
        assert_eq!((n, tiling), (2, (1, 1)));
        // Pixel (y=1, x=15) is in patch 1 at (row 1, col 1); red channel offset 0.
        let local = &data[GRID * GRID * PATCH_DIM..];
        assert_eq!(local[PATCH_DIM + PATCH + 1], f16::from_f32(1.0));
        assert_eq!(local[PATCH_DIM + 196 + PATCH + 1], f16::from_f32(-1.0));
        Ok(())
    }
}
