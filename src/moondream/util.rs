use std::{fs::File, path::Path};
use ndarray::{s, ArrayD, Axis};
use image::{DynamicImage, GenericImageView};
use ndarray::Array3;
use half::f16;
use crate::error::Result;

pub fn normalize_std_mean(arr: Array3<f32>, mean: [f32; 3], std: [f32; 3]) -> Array3<f32> {
    arr.map(|x| (x - mean[0]) / std[0])
}

pub fn normalize_image(image: DynamicImage) -> Array3<f32> {
    // Convert to RGB
    let rgb_image = image.to_rgb8();
    let (width, height) = rgb_image.dimensions();
    
    // Create a 3D array with shape CHW [channels, height, width]
    ndarray::Array3::from_shape_fn(
        (3, height as usize, width as usize),
        |(c, y, x)| {
            // use BGR format
            let pixel = image.get_pixel(x as u32, y as u32);
            match c {
                0 => pixel[2] as f32 / 255.0,
                1 => pixel[1] as f32 / 255.0,
                2 => pixel[0] as f32 / 255.0,
                _ => 0.0
            }
        }
    )
}

pub fn read_npy(path: &Path) -> Result<ndarray::ArrayD<f16>> {
    let npy_bytes = std::fs::read(path).unwrap();
    let reader = npyz::NpyFile::new(&npy_bytes[..])?;
    let shape = reader.shape().to_vec();
    let order = reader.order();
    let data = reader.into_vec::<f16>()?;

    let result = to_array_d(data, shape, order);
    println!("result: {:?}", result.shape());
    
    return Ok(result);
}

fn to_array_d<T>(data: Vec<T>, shape: Vec<u64>, order: npyz::Order) -> ndarray::ArrayD<T> {
    use ndarray::ShapeBuilder;

    let shape = shape.into_iter().map(|x| x as usize).collect::<Vec<_>>();
    let true_shape = shape.set_f(order == npyz::Order::Fortran);

    ndarray::ArrayD::from_shape_vec(true_shape, data)
        .unwrap_or_else(|e| panic!("shape error: {}", e))
}

/// Applies 2D adaptive average pooling over an input signal.
///
/// Resizes input to a target size by averaging values in local neighborhoods.
/// The neighborhoods are computed to evenly cover the input image while
/// maintaining approximately equal size. Similar to PyTorch's
/// adaptive_avg_pool2d but expects input shape (H,W,C) rather than (N,C,H,W).
///
/// Args:
///     x: Input tensor of shape (height, width, channels)
///     output_size: Target output size. Can be:
///         - Single integer for square output (size, size)
///         - Tuple of two ints (out_height, out_width)
///
/// Returns:
///     Tensor of shape (out_height, out_width, channels)
///
/// Example:
///     >>> img = np.random.randn(32, 32, 3)  # 32x32 RGB image
///     >>> pooled = adaptive_avg_pool2d(img, (7, 7))  # Resize to 7x7
///     >>> pooled.shape
///     (7, 7, 3)
pub fn adaptive_avg_pool2d(arr: Array3<f16>, output_size: (usize, usize)) -> Array3<f16> {
    let shape = arr.shape();
    let height = shape[0];
    let width = shape[1];
    let channels = shape[2];

    let out_height = output_size.0;
    let out_width = output_size.1;

    let stride_h = ((height as f32) / (out_height as f32)).floor() as usize;
    let stride_w = ((height as f32) / (out_width as f32)).floor() as usize;
    let kernel_h = height - (out_height - 1) * stride_h;
    let kernel_w = width - (out_width - 1) * stride_w;

    let mut output: Array3<f32> = Array3::zeros((out_height, out_width, channels));

    for i in 0..out_height {
        for j in 0..out_width {
            let h_start = i * stride_h;
            let h_end = h_start + kernel_h;
            let w_start = j * stride_w;
            let w_end = w_start + kernel_w;
            let mean = arr.slice(s![h_start..h_end, w_start..w_end, ..]).mapv(|v| v.to_f32());
            let mean = mean.mean_axis(Axis(0)).unwrap().mean_axis(Axis(0)).unwrap();


            output.slice_mut(s![i, j, ..]).assign(&mean);
        }
    }

    return output.mapv(|v| f16::from_f32(v));
}

pub fn argmax<T>(arr: &ArrayD<T>, axis: isize) -> ArrayD<usize> 
where T: PartialOrd + Copy 
{
    let actual_axis = if axis < 0 {
        arr.ndim() as isize + axis
    } else {
        axis
    } as usize;

    arr.map_axis(Axis(actual_axis), |lane| {
        lane.iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    })
}