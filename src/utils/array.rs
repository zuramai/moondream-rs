use image::DynamicImage;
use ndarray::Array3;

pub fn normalize_image(image: DynamicImage) -> Array3<f32> {
    
    // Convert to RGB
    let rgb_image = image.to_rgb8();
    let (width, height) = rgb_image.dimensions();
    
    // Create a 3D array with shape CHW [channels, height, width]
    ndarray::Array3::from_shape_fn(
        (3, height as usize, width as usize),
        |(c, y, x)| {
            // use BGR format
            let pixel = rgb_image.get_pixel(x as u32, y as u32);
            match c {
                0 => pixel[2] as f32 / 255.0,
                1 => pixel[1] as f32 / 255.0,
                2 => pixel[0] as f32 / 255.0,
                _ => 0.0
            }
        }
    )
}