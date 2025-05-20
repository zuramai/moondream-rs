use std::io::Read;
use std::path::{Path, PathBuf};

use half::f16;
use image::DynamicImage;
use ndarray::{s, Array1, Array3, ArrayD, Axis, Ix2, Ix3};
use tokenizers::Tokenizer;
use super::config::{self, MoondreamConfig};
use super::preprocess;
use super::types::EncodedImage;
use crate::error::{Error, Result};
use crate::moondream::util;
use super::engine::{self, Engine};

pub struct Moondream {
    vision_encoder: engine::Engine,
    vision_projection: engine::Engine,
    text_encoder: engine::Engine,
    text_decoder: engine::Engine,
    size_encoder: engine::Engine,
    size_decoder: engine::Engine,
    coord_encoder: engine::Engine,
    coord_decoder: engine::Engine,
    tokenizer: Tokenizer,
    initial_kv_cache: ArrayD<f16>,
    config: MoondreamConfig,
}

impl Moondream {
    pub fn from_path(model_path: &str) -> Result<Self> {
        let path = Path::new(model_path);
        println!("reading npy from {}", &path.join("initial_kv_cache.npy").to_str().unwrap());
        let initial_kv_cache = util::read_npy(&path.join("initial_kv_cache.npy"))?;
        
        println!("reading npy done");
        Ok(Self {
            vision_encoder: Engine::new(path.join("vision_encoder.onnx"))?,
            vision_projection: Engine::new(path.join("vision_projection.onnx"))?,
            config: MoondreamConfig::from_file(&path.join("config.json"))?,
            coord_decoder: Engine::new(path.join("coord_decoder.onnx"))?,
            coord_encoder: Engine::new(path.join("coord_encoder.onnx"))?,
            size_decoder: Engine::new(path.join("size_decoder.onnx"))?,
            size_encoder: Engine::new(path.join("size_encoder.onnx"))?,
            text_decoder: Engine::new(path.join("text_decoder.onnx"))?,
            text_encoder: Engine::new(path.join("text_encoder.onnx"))?,
            initial_kv_cache: initial_kv_cache,
            tokenizer: Tokenizer::from_file(path.join("tokenizer.json"))?
        })
    }
    pub fn encode_image(&self, image: DynamicImage) -> Result<EncodedImage> {
        // Run vision encoder
        let (image_patches, template) = preprocess::create_patches(image, 378);
        dbg!(&image_patches.shape());
        let mut patch_emb = self.vision_encoder.run(image_patches.into_dyn())?;
        let patch_emb_shape = patch_emb.shape();

        dbg!(&patch_emb.shape());
        // Reassemble patches into a single image embedding
        let global_patch = patch_emb.slice(s![0, .., ..]).into_dyn();
        if template == (1, 1) {
            patch_emb = ndarray::concatenate(Axis(2), &[global_patch.view(), global_patch.view()])?;
        } else {
            // shape 729
            let seq_len = patch_emb.shape()[1] as f32;

            // we get the sqrt of 729 (27). 
            let w = seq_len.sqrt().round() as i32;

            let mut rows = vec![];
            for r in 0..template.0 {
                let mut row = Vec::new();
                for c in 0..template.1 {
                    // we would like to make a patch with shape (x, 729, 1152)
                    let x = r * template.1 + c;
                    let patch = patch_emb.slice(s![x as usize, .., ..]).into_dimensionality::<Ix2>()?;

                    // reshape it to (w, w, 1152) which is (27, 27, 1152)
                    let patch = patch.to_shape((w as usize, w as usize, patch_emb_shape[2]))?.to_owned();
                    row.push(patch);
                }
                let concat = ndarray::concatenate(Axis(1), &row.iter().map(|x| x.view()).collect::<Vec<_>>())?;
                rows.push(concat);
            }
            // concat everything, if the template is (2, 2) we will get (54, 54, 1152)
            let result = ndarray::concatenate(Axis(0), &rows.iter().map(|x| x.view()).collect::<Vec<_>>())?;
            let result = util::adaptive_avg_pool2d(result, (w as usize, w as usize));
            dbg!(&result.shape());
            let result = result.into_shape_with_order(((w * w) as usize, patch_emb_shape[2]))?.into_dyn();
            dbg!(&result.shape());
            patch_emb = ndarray::concatenate![Axis(1), global_patch, result];
            dbg!(patch_emb.shape());
        }

        patch_emb.insert_axis(Axis(0));

        


        Ok(EncodedImage{
            kv_cache: Array1::from(Vec::from([1,2,3])).into_dyn(),
            pos: 1
        })
    }

    pub fn caption(&self) -> String {
        "".into()
    }
    pub fn query(&self) -> String {
        "".into()
    }
    pub fn detect(&self) -> String {
        "".into()
    }
    pub fn point(&self) -> String {
        "".into()
    }
}

#[cfg(test)]
mod tests {
    use super::Moondream;

    #[test]
    pub fn test_encode_image() {
        let md = Moondream::from_path("./model").expect("Failed to initialize moondream");
        let img = image::open("demo-1.jpg").expect("Failed to open image person.webp");
        assert!(md.encode_image(img).is_ok());
    }
}