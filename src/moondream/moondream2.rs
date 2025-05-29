use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use half::f16;
use image::DynamicImage;
use ndarray::{s, Array1, Array2, Array3, Array6, ArrayD, Axis, Ix2, Ix3};
use tokenizers::Tokenizer;
use super::config::{self, MoondreamConfig};
use super::{preprocess, types};
use super::types::EncodedImage;
use crate::error::{Error, Result};
use crate::moondream::util;
use super::engine::{self, Engine};

const DEFAULT_MAX_TOKENS: i32 = 512;

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
    pub fn encode_image(&mut self, image: DynamicImage) -> Result<EncodedImage> {
        // Convert image to patches
        let (image_patches, template) = preprocess::create_patches(image, 378);

        // Encode the patches using vision_encoder
        dbg!("running vision encoder");
        let mut vision_encoded = self.vision_encoder.run(HashMap::from([("input", image_patches.view().into_dyn())]), vec!["output"])?;
        let mut patch_emb= vision_encoded.remove("output").unwrap();
        let patch_emb_shape = patch_emb.shape();

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
            let result = result.into_shape_with_order(((w * w) as usize, patch_emb_shape[2]))?.into_dyn();
            patch_emb = ndarray::concatenate![Axis(1), global_patch, result];
        }

        let patch_emb = patch_emb.insert_axis(Axis(0));

        // run vision projection
        dbg!("running vision projection");
        let mut vision_projection = self.vision_projection.run(HashMap::from([("input", patch_emb.as_standard_layout().view())]), vec!["output"])?;
        let input_embeds = vision_projection.remove("output").unwrap();

        // kv_cache shape is [24, 2, 1, 32, 730, 64]
        // 24 is transformer layers
        // 2 is key and value
        // 1 is batch size 
        // 730 is sequence length
        // the length of each key or value
        let kv_cache = self.initial_kv_cache.view();
        let pos = input_embeds.shape()[input_embeds.ndim() - 2] + kv_cache.shape()[kv_cache.ndim() - 2];

        dbg!("running text decoder for vision projection patches result");
        let mut text_decoder  = self.text_decoder.run(HashMap::from([
            ("input_embeds", input_embeds.as_standard_layout().view()),
            ("kv_cache", kv_cache.as_standard_layout().view())
        ]), vec!["new_kv_cache"])?;
        let kv_cache_update = text_decoder.remove("new_kv_cache").unwrap();

        let kv_cache = ndarray::concatenate![Axis(kv_cache.ndim()-2), kv_cache, kv_cache_update];

        Ok(EncodedImage{
            kv_cache: kv_cache,
            pos: pos
        })
    }

    pub fn caption(&mut self, image: DynamicImage, length: types::CaptionLength) -> Result<String> {
        let input_ids = match length {
            types::CaptionLength::Normal => self.config.templates.caption.normal.clone(),
            types::CaptionLength::Short => self.config.templates.caption.short.clone(),
        };

        let input_ids = Array1::from_vec(input_ids)
            .insert_axis(Axis(0)) // convert [input_ids] to [[input_ids]]
            .into_dyn();
    
        // encode the prompt
        let mut text_encoded = self.text_encoder.run::<i64, f16>(HashMap::from([
            ("input_ids", input_ids.as_standard_layout().view())
        ]), vec!["input_embeds"])?;
        let input_embeds = text_encoded.remove("input_embeds").unwrap();

        // encode the image
        let encoded_image = self.encode_image(image)?;
        let max_tokens = DEFAULT_MAX_TOKENS;

        // generate result
        self.generate(input_embeds, encoded_image, max_tokens)
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
    fn generate(&mut self, mut input_embeds: ArrayD<f16>, encoded_image: EncodedImage, max_tokens: i32) -> Result<String> {
        dbg!("Generating..");
        let mut pos = encoded_image.pos;
        let mut kv_cache = prepare_kv_cache(encoded_image);
        let mut generated_tokens = 0;
        let input_length = input_embeds.shape()[input_embeds.ndim()-2];

        let mut tokens = Vec::with_capacity(max_tokens as usize);

        while generated_tokens < max_tokens {
            let start = Instant::now();
            let kv_cache_seqlen = kv_cache.slice(s![.., .., .., .., ..pos, ..]);
            let mut decoder  = self.text_decoder.run::<f16, f16>(HashMap::from([
                ("input_embeds", input_embeds.as_standard_layout().view()),
                ("kv_cache", kv_cache_seqlen.as_standard_layout().into_dyn().view())
                ]), vec!["logits", "new_kv_cache"])?;

            let end = start.elapsed();

            let logits = decoder.remove("logits").unwrap(); 
            let kv_cache_update = decoder.remove("new_kv_cache").unwrap(); 
            
            // Only update the position by the actual new token length (1) after the first iteration
            let update_length = if generated_tokens == 0 { input_length } else { 1 };
            kv_cache.slice_mut(s![.., .., .., .., pos..pos+update_length, ..]).assign(&kv_cache_update);
            pos += update_length;

            let next_token = util::argmax(&logits, -1)[0];
            if next_token as i32 == self.config.special_tokens.eos {
                break;               
            };

            tokens.push(next_token as u32);
            println!("text decoder duration: {}ms: Text: {}", end.as_millis(), self.tokenizer.decode(&[next_token as u32], true)?);
            
            generated_tokens += 1;

            let text_encoder_input = Array1::from_vec(vec![next_token as i64]).insert_axis(Axis(0)).into_dyn();
            let text_encoded = self.text_encoder.run::<i64, f16>(HashMap::from([
                ("input_ids", text_encoder_input.as_standard_layout().view())
                ]), vec!["input_embeds"]);
                
            input_embeds = text_encoded.unwrap().remove("input_embeds").unwrap();
        }
        Ok(self.tokenizer.decode(&tokens, true)?)
    }
}


/// creates a copy of the encoded image kv cache with max sequence length 2048.
/// returns a copy of KV cache expanded to max sequence length of 2048
fn prepare_kv_cache(encoded_image: EncodedImage) -> ArrayD<f16> {
    let original_shape = encoded_image.kv_cache.shape().to_vec();
    let original_shape_len = original_shape.len();
    let new_shape = [
        &original_shape[0..original_shape_len-2],
        &[2048, original_shape[original_shape_len-1]],
    ].concat();
    
    let mut kv_cache: ArrayD<f16> = ArrayD::zeros(new_shape).mapv(|v| f16::from_f32(v));
    kv_cache.slice_mut(s![.., .., .., .., ..original_shape[original_shape_len-2], ..]).assign(&encoded_image.kv_cache);
    return kv_cache;
}

#[cfg(test)]
mod tests {
    use super::Moondream;

    pub fn test_encode_image() {
        let mut md = Moondream::from_path("./model").expect("Failed to initialize moondream");
        let img = image::open("demo-1.jpg").expect("Failed to open image person.webp");
        assert!(md.encode_image(img).is_ok());
    }

    #[test]
    pub fn test_caption() {
        let mut md = Moondream::from_path("./model").expect("Failed to initialize moondream");
        let img = image::open("demo-1.jpg").expect("Failed to open image demo-1.jpg");
        let v = md.caption(img, crate::moondream::types::CaptionLength::Normal);
        dbg!(&v);
        assert!(v.is_ok());
    }
}