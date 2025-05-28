use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

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
    pub fn encode_image(&self, image: DynamicImage) -> Result<EncodedImage> {
        // Run vision encoder
        let (image_patches, template) = preprocess::create_patches(image, 378);
        dbg!(&image_patches.shape());
        let mut vision_encoded = self.vision_encoder.run(HashMap::from([("input", image_patches.view().into_dyn())]), vec!["output"])?;
        let mut patch_emb= vision_encoded.remove("output").unwrap();
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

        let patch_emb = patch_emb.insert_axis(Axis(0));

        // run vision projection
        let mut vision_projection = self.vision_projection.run(HashMap::from([("input", patch_emb.view())]), vec!["output"])?;
        let input_embeds = vision_projection.remove("output").unwrap();

        let kv_cache = self.initial_kv_cache.view();
        let pos = input_embeds.shape()[input_embeds.ndim() - 2] + kv_cache.shape()[kv_cache.ndim() - 2];

        let mut text_decoder  = self.text_decoder.run(HashMap::from([
            ("input_embeds", input_embeds.view()),
            ("kv_cache", kv_cache.view())
        ]), vec!["new_kv_cache"])?;
        let mut kv_cache_update = text_decoder.remove("new_kv_cache").unwrap();

        dbg!(&kv_cache_update.shape());

        let kv_cache = ndarray::concatenate![Axis(kv_cache.ndim()-2), kv_cache, kv_cache_update];

        dbg!(&kv_cache.shape());
        dbg!(&pos);

        Ok(EncodedImage{
            kv_cache: kv_cache,
            pos: pos
        })
    }

    pub fn caption(&self, image: DynamicImage, length: types::CaptionLength) -> Result<String> {
        let input_ids = match length {
            types::CaptionLength::Normal => self.config.templates.caption.normal.clone(),
            types::CaptionLength::Short => self.config.templates.caption.short.clone(),
        };
        dbg!(&input_ids);

        let input_ids = Array2::from_shape_vec((1, input_ids.len()), input_ids)?.into_dyn();
        let mut text_encoded = self.text_encoder.run::<i64, f16>(HashMap::from([
            ("input_ids", input_ids.view())
        ]), vec!["input_embeds"])?;

        let input_embeds = text_encoded.remove("input_embeds").unwrap();


        let encoded_image = self.encode_image(image)?;
        let max_tokens = DEFAULT_MAX_TOKENS;

        let generate_result = self.generate(input_embeds, encoded_image, max_tokens);

        generate_result
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
    fn generate(&self, mut input_embeds: ArrayD<f16>, encoded_image: EncodedImage, max_tokens: i32) -> Result<String> {
        let mut pos = encoded_image.pos;
        let mut kv_cache = prepare_kv_cache(encoded_image);
        let mut generated_tokens = 0;
        let input_length = input_embeds.shape()[input_embeds.ndim()-2];
        let start = std::time::Instant::now();

        dbg!(&generated_tokens);
        dbg!(&max_tokens);

        let mut tokens = vec![];

        while generated_tokens < max_tokens {
            dbg!("masuk ke", &generated_tokens);
            let mut decoder  = self.text_decoder.run::<f16, f16>(HashMap::from([
                ("input_embeds", input_embeds.view()),
                ("kv_cache", kv_cache.slice(s![.., .., .., .., ..pos, ..]).into_dyn())
                ]), vec!["logits", "new_kv_cache"])?;
            dbg!("masuk1");

            let logits = decoder.remove("logits").unwrap(); 
            let kv_cache_update = decoder.remove("new_kv_cache").unwrap(); 
            kv_cache.slice_mut(s![.., .., .., .., pos..pos+input_length, ..]).assign(&kv_cache_update);
            dbg!("masuk2");
            pos += input_length;

            let next_token = util::argmax(&logits, -1)[0];
            if next_token as i32 == self.config.special_tokens.eos {
                break;               
            };
            dbg!(&next_token);
            tokens.push(next_token as u32);

            generated_tokens += 1;

            let text_encoder_input = Array1::from_vec(vec![next_token as i64]).insert_axis(Axis(0)).into_dyn();
            let text_encoded = self.text_encoder.run::<i64, f16>(HashMap::from([
                ("input_ids", text_encoder_input.view())
                ]), vec!["input_embeds"]);
                
            input_embeds = text_encoded.unwrap().remove("input_embeds").unwrap();
        }
        let end = start.elapsed();
        println!("time elapsed text decoder: {:?}", end);
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
    dbg!(&kv_cache.shape());
    return kv_cache;
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

    #[test]
    pub fn test_caption() {
        let md = Moondream::from_path("./model").expect("Failed to initialize moondream");
        let img = image::open("demo-1.jpg").expect("Failed to open image person.webp");
        let v = md.caption(img, crate::moondream::types::CaptionLength::Normal);
        dbg!(&v);
        assert!(v.is_ok());
    }
}