use std::cmp::max;

use image::{DynamicImage, GenericImageView};
use ndarray::{Array3, Array4, Axis};

use super::util::{normalize_image, normalize_std_mean};

pub fn create_patches(image: DynamicImage, image_patch_size: u32) -> (Array4<f32>, (u32, u32)) {
    // start with global patch
    let mut patches = vec![
        image.resize_exact(image_patch_size, image_patch_size, image::imageops::FilterType::Nearest)
    ];


    // Find the closest resolution template.
    //
    // (1, 2)              (2, 1)              (2, 2)
    // +-------+-------+   +-----------+       +-------+-------+
    // |   1   |   2   |   |     1     |       |   1   |   2   |
    // +-------+-------+   +-----------+       +-------+-------+
    //                     |     2     |       |   3   |   4   |
    //                     +-----------+       +-------+-------+
    let res_templates = vec![(1,2), (2, 1), (2, 2)];

    let (width, height) = image.dimensions();
    let max_dimension = max(width, height);
    let selected_template: (u32, u32);
    
    if (max_dimension as f32) < (image_patch_size as f32) * 1.4 {
        selected_template = (1,1);
    } else {
        let aspect_ratio = width / height;
        let template_aspect_ratios_diff: Vec<u32> = res_templates.iter().map(|ratio| ((ratio.0 / ratio.1) as i32).abs_diff(aspect_ratio as i32)).collect();
        let min = template_aspect_ratios_diff.iter().enumerate().min().unwrap().0;
        selected_template = res_templates[min];
    }

    let patch_width = width / selected_template.1;
    let patch_height = height / selected_template.0;

    for row in 0..selected_template.0 {
        for col in 0..selected_template.1 {
            let x_min = col * patch_width;
            let y_min = row * patch_height;
            patches.push(
                image.crop_imm(x_min, y_min, patch_width, patch_height)
                    .resize_exact(image_patch_size, image_patch_size, image::imageops::FilterType::Nearest)
            );
        }
    }
    let patches_vec: Vec<Array3<f32>> = patches.into_iter().map(|patch| normalize_std_mean(normalize_image(patch), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])).collect();
    
    let views: Vec<_> = patches_vec.iter().map(|patch| patch.view()).collect();
    let normalized_patches = ndarray::stack(Axis(0), &views)
        .expect("Failed to stack patches into Array4");


    return (normalized_patches, selected_template);
}