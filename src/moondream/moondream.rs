// use image::imageops::FilterType;
// use ndarray::{Array1, Array2, Array3, Array4, ArrayD, Axis, IxDyn};
// use serde::{Deserialize, Serialize};
// use std::collections::HashMap;
// use tokenizers::Tokenizer;
// use ndarray::s;
// use crate::error::{Error, Result};
// use crate::triton::client::{InferInputs, TritonGRPCClient};
// use crate::utils::array::normalize_image;
// use image::DynamicImage;

// const DEFAULT_MAX_TOKENS: usize = 512;
// const MIN_SUPPORTED_VERSION: u32 = 1;
// const MAX_SUPPORTED_VERSION: u32 = 1;

// #[derive(Debug, Serialize, Deserialize)]
// pub struct Config {
//     pub model_version: u32,
//     pub templates: Templates,
//     pub special_tokens: SpecialTokens,
// }

// #[derive(Debug, Serialize, Deserialize)]
// pub struct Templates {
//     pub caption: Option<CaptionTemplate>,
//     pub query: Option<QueryTemplate>,
//     pub detect: Option<DetectTemplate>,
// }

// #[derive(Debug, Serialize, Deserialize)]
// pub struct CaptionTemplate {
//     pub short: Vec<u32>,
//     pub normal: Vec<u32>,
// }

// #[derive(Debug, Serialize, Deserialize)]
// pub struct QueryTemplate {
//     pub prefix: Vec<u32>,
//     pub suffix: Vec<u32>,
// }

// #[derive(Debug, Serialize, Deserialize)]
// pub struct DetectTemplate {
//     pub prefix: Vec<i64>,
//     pub suffix: Vec<i64>,
// }

// #[derive(Debug, Serialize, Deserialize)]
// pub struct SpecialTokens {
//     pub bos: u32,
//     pub eos: u32,
//     pub coord: u32,
//     pub size: u32,
// }

// #[derive(Debug)]
// pub struct EncodedImage {
//     pub pos: usize,
//     pub kv_cache: ArrayD<f32>,
// }

// #[derive(Debug)]
// pub struct DetectOutput {
//     pub objects: Vec<BoundingBox>,
// }

// #[derive(Debug)]
// pub struct BoundingBox {
//     pub x_min: f32,
//     pub y_min: f32,
//     pub x_max: f32,
//     pub y_max: f32,
// }

// pub struct Moondream {
//     client: TritonGRPCClient,
//     tokenizer: Tokenizer,
//     config: Config,
//     initial_kv_cache: ArrayD<f32>,
// }

// impl Moondream {
//     pub fn new(client: TritonGRPCClient, tokenizer: Tokenizer, config: Config, initial_kv_cache: ArrayD<f32>) -> Self {
//         Self {
//             client,
//             tokenizer,
//             config,
//             initial_kv_cache,
//         }
//     }

//     fn prepare_kv_cache(&self, encoded_image: &EncodedImage) -> Result<ArrayD<f32>> {
//         let original_shape = encoded_image.kv_cache.shape();
//         let len = original_shape.len();
//         let mut new_shape = original_shape.to_vec();
//         new_shape[len - 2] = 2048;
        
//         let mut kv_cache = ArrayD::zeros(IxDyn(&new_shape));
//         let mut view = kv_cache.slice_mut(s![.., .., .., .., ..original_shape[original_shape.len()-2], ..]);
//         view.assign(&encoded_image.kv_cache);
        
//         Ok(kv_cache)
//     }

//     // async fn run_decoder(
//     //     &self,
//     //     input_embeds: ArrayD<f32>,
//     //     kv_cache: &mut ArrayD<f32>,
//     //     pos: usize,
//     // ) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
//     //     let kv_cache_slice = kv_cache.slice(s![.., .., .., .., ..pos, ..]);
//     //     let mut inputs = InferInputs::new_f16();
//     //     inputs.add_input_fp16("input_embeds", input_embeds);
//     //     inputs.add_input_fp16("kv_cache", kv_cache_slice.to_owned().into_dyn());

//     //     let mut outputs = self.client
//     //         .infer_http("text_decoder".to_string(), "1".to_string(), inputs, vec!["logits", "hidden"])
//     //         .await?;

//     //     let hidden = outputs.remove("hidden").expect("Missing hidden output");
//     //     let logits = outputs.remove("logits").expect("Missing logits output");

//     //     Ok((logits.get_f32(), hidden.get_f32()))
//     // }

//     pub async fn load_image(&self, image: &DynamicImage) -> Result<Array4<f32>> {
//         let image = image.resize(378, 378, FilterType::CatmullRom);
        
//         // convert to Array3
//         let image = normalize_image(image);

//         // Create mean and std arrays with shape [3, 1, 1]
//         let mean = Array3::from_shape_vec((3, 1, 1), vec![0.5f32, 0.5, 0.5]).unwrap();
//         let std = Array3::from_shape_vec((3, 1, 1), vec![0.5f32, 0.5, 0.5]).unwrap();

//         // Normalize using broadcasting
//         let image = &image - &mean;
//         let image = &image / &std;

//         let image = image.insert_axis(Axis(0));

//         Ok(image)
//     }

//     pub async fn encode_image(&self, image: Array4<f32>) -> Result<EncodedImage> {
//         let vision_encoder_input = InferInputs::new_f16()
//             .add_input_fp16("input", image.into_dyn());

//         // shape: [-1,729,720]
//         let mut vision_encoder_infer = self.client
//             .infer_http("vision_encoder".to_string(), "1".to_string(), vision_encoder_input, vec!["output"])
//             .await?;
//         let output = vision_encoder_infer.remove("output").ok_or(Error::new("Missing output"))?;
//         let output = output.get_f32();
//         // Get global patch (first patch)
//         let global_patch = output.slice(s![.., 0..1, ..]).to_owned();
        
//         // Concatenate global patch with itself along last axis
//         let output = ndarray::concatenate(
//             Axis(2), 
//             &[global_patch.view(), global_patch.view()]
//         )?;

//         let mut vision_projection_input = InferInputs::new_f16()
//             .add_input_fp16("input", output.into_dyn());

//         // Run vision projection. Output shape: [1,729,1024]
//         let mut vision_projection_infer = self.client
//             .infer_http("vision_projection".to_string(), "1".to_string(), vision_projection_input, vec!["output"])
//             .await?;
//         let output = vision_projection_infer.remove("output").ok_or(Error::new("Missing output"))?;
//         let input_embeds = output.get_f32();
        
//         // Get KVCache
//         let kv_cache = &self.initial_kv_cache;
//         let pos = input_embeds.shape()[input_embeds.ndim() - 2] + kv_cache.shape()[kv_cache.ndim() - 2];

//         let mut text_decoder_input = InferInputs::new_f16()
//             .add_input_fp16("input_embeds", input_embeds.into_dyn())
//             .add_input_fp16("kv_cache", kv_cache.into_dyn());

//         let mut text_decoder_infer = self.client
//             .infer_http("text_decoder".to_string(), "1".to_string(), text_decoder_input, vec!["new_kv_cache"])
//             .await?;
//         let output = text_decoder_infer.remove("new_kv_cache").ok_or(Error::new("Missing output"))?;
//         let kv_cache = output.get_f32();

        
//         Ok(EncodedImage {
//             pos,
//             kv_cache
//         })
//     }
    

//     pub async fn detect(&self, image: DynamicImage, object: &str) -> Result<DetectOutput> {
//         let image = self.load_image(&image).await?;
//         let encoded_image = self.encode_image(image).await?;
//         Err(Error::Error("au".into()))
//         // self.run(encoded_image, object).await
//     }

//     pub async fn text_decode(&self, kv_cache: ArrayD<f32>, input_embeds: ArrayD<f32>) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
//         let mut inputs = InferInputs::new_f16();
//         inputs.add_input_fp16("kv_cache", kv_cache.into_dyn());
//         inputs.add_input_fp16("input_embeds", input_embeds.into_dyn());

//         let mut outputs = self.client
//             .infer_http("text_decoder".to_string(), "1".to_string(), inputs, vec!["logits", "hidden"])
//             .await?;

//         let new_kv_cache = outputs.remove("new_kv_cache").expect("Missing new_kv_cache output").get_f32();
//         let logits = outputs.remove("logits").expect("Missing logits output").get_f32();
        
//         Ok((new_kv_cache, logits))
//     }

//     // async fn run(&self, image: Array3<f32>, object: &str) -> Result<DetectOutput> {
//     //     let detect_template = self.config.templates.detect.as_ref()
//     //         .ok_or(Error::new("Model does not support detect"))?;

//     //     let tokens = self.tokenizer.encode(
//     //         format!(" {}", object),
//     //         true
//     //     )?;
//     //     if tokens.is_empty() {
//     //         return Err(Error::new("Empty prompts are not supported"));
//     //     }

//     //     let mut tokens = tokens.get_ids().to_vec();
//     //     let mut generated_tokens = 0usize;

//     //     let bos_token = self.config.special_tokens.bos;
//     //     let eos_token = self.config.special_tokens.eos;


//     //     let start_gen = std::time::Instant::now();
//     //     let mut load_t = std::time::Duration::from_secs_f64(0f64);

//     //     let max_tokens = DEFAULT_MAX_TOKENS;
//     //     for i in 0..max_tokens {
//     //         let context_size = if i > 0 { 1 } else { tokens.len() };
//     //         let ctx = &tokens[tokens.len().saturating_sub(context_size)..];

//     //         let (logits, new_kv_cache) = self.text_decode(kv_cache, input_embeds).await?;
//     //     }
//     //     let prompt_tokens: Vec<i64> = detect_template.prefix.iter()
//     //         .chain(object_tokens.iter())
//     //         .chain(detect_template.suffix.iter())
//     //         .copied()
//     //         .collect();

//     //     let mut inputs = InferInputs::new_f16();
//     //     inputs.add_input_i64("input_ids", &Array2::from_shape_vec((1, prompt_tokens.len()), prompt_tokens)?.into_dyn());
        
//     //     let outputs = self.client
//     //         .infer_http("text_encoder".to_string(), "1".to_string(), inputs, vec!["hidden"])
//     //         .await?;
        
//     //     let mut hidden = outputs.get("hidden").ok_or(Error::new("Missing hidden output"))?.data.clone();
//     //     let mut kv_cache = self.prepare_kv_cache(&image)?;
//     //     let mut pos = image.pos;
//     //     let mut objects = Vec::new();
//     //     const MAX_OBJECTS: usize = 50;

//     //     while objects.len() < MAX_OBJECTS {
//     //         let (logits, new_hidden) = self.run_decoder(hidden.clone(), &mut kv_cache, pos).await?;
//     //         pos += new_hidden.shape()[new_hidden.ndim() - 2];
//     //         hidden = new_hidden;

//     //         let next_token = logits
//     //             .index_axis(Axis(1), logits.shape()[1] - 1)
//     //             .iter()
//     //             .enumerate()
//     //             .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
//     //             .map(|(i, _)| i as u32)
//     //             .unwrap();

//     //         if next_token == self.config.special_tokens.eos {
//     //             break;
//     //         }

//     //         let (x_center, y_center, width, height) = self.decode_box(&mut hidden, &mut kv_cache, &mut pos).await?;

//     //         objects.push(BoundingBox {
//     //             x_min: x_center - width / 2.0,
//     //             y_min: y_center - height / 2.0,
//     //             x_max: x_center + width / 2.0,
//     //             y_max: y_center + height / 2.0,
//     //         });
//     //     }

//     //     Ok(DetectOutput { objects })
//     // }

//     // async fn decode_box(
//     //     &self,
//     //     hidden: &mut ArrayD<f32>,
//     //     kv_cache: &mut ArrayD<f32>,
//     //     pos: &mut usize,
//     // ) -> Result<(f32, f32, f32, f32)> {
//     //     let mut inputs = InferInputs::new_f16();
//     //     inputs.add_input("input", &hidden.slice(s![0..1, -1..-1+1, ..]).to_owned().into_dyn());
        
//     //     let outputs = self.client
//     //         .infer_http("coord_decoder".to_string(), "1".to_string(), inputs, vec!["output"])
//     //         .await?;
        
//     //     let x_logits = outputs.get("output").ok_or(Error::new("Missing output"))?.data.clone();
//     //     let x_center = self.argmax_normalize(&x_logits)?;

//     //     let mut inputs = InferInputs::new_f16();
//     //     inputs.add_input("input", &Array1::from_vec(vec![x_center]).into_dyn());
        
//     //     let outputs = self.client
//     //         .infer_http("coord_encoder".to_string(), "1".to_string(), inputs, vec!["output"])
//     //         .await?;
        
//     //     *hidden = outputs.get("output").ok_or(Error::new("Missing output"))?.data.clone();

//     //     let mut inputs = InferInputs::new_f16();
//     //     inputs.add_input("input", &hidden.slice(s![0..1, -1..-1+1, ..]).to_owned().into_dyn());
        
//     //     let outputs = self.client
//     //         .infer_http("coord_decoder".to_string(), "1".to_string(), inputs, vec!["output"])
//     //         .await?;
        
//     //     let y_logits = outputs.get("output").ok_or(Error::new("Missing output"))?.data.clone();
//     //     let y_center = self.argmax_normalize(&y_logits)?;

//     //     let mut inputs = InferInputs::new_f16();
//     //     inputs.add_input("input", &Array1::from_vec(vec![y_center]).into_dyn());
        
//     //     let outputs = self.client
//     //         .infer_http("coord_encoder".to_string(), "1".to_string(), inputs, vec!["output"])
//     //         .await?;
        
//     //     *hidden = outputs.get("output").ok_or(Error::new("Missing output"))?.data.clone();

//     //     let mut inputs = InferInputs::new_f16();
//     //     inputs.add_input("input", &hidden.slice(s![0..1, -1..-1+1, ..]).to_owned().into_dyn());
        
//     //     let outputs = self.client
//     //         .infer_http("coord_decoder".to_string(), "1".to_string(), inputs, vec!["output"])
//     //         .await?;
        
//     //     let width_logits = outputs.get("output").ok_or(Error::new("Missing output"))?.data.clone();
//     //     let width = self.argmax_normalize(&width_logits)?;

//     //     let mut inputs = InferInputs::new_f16();
//     //     inputs.add_input("input", &Array1::from_vec(vec![width]).into_dyn());
        
//     //     let outputs = self.client
//     //         .infer_http("coord_encoder".to_string(), "1".to_string(), inputs, vec!["output"])
//     //         .await?;
        
//     //     *hidden = outputs.get("output").ok_or(Error::new("Missing output"))?.data.clone();

//     //     let mut inputs = InferInputs::new_f16();
//     //     inputs.add_input("input", &hidden.slice(s![0..1, -1..-1+1, ..]).to_owned().into_dyn());
        
//     //     let outputs = self.client
//     //         .infer_http("coord_decoder".to_string(), "1".to_string(), inputs, vec!["output"])
//     //         .await?;
        
//     //     let height_logits = outputs.get("output").ok_or(Error::new("Missing output"))?.data.clone();
//     //     let height = self.argmax_normalize(&height_logits)?;

//     //     let mut inputs = InferInputs::new_f16();
//     //     inputs.add_input("input", &Array1::from_vec(vec![height]).into_dyn());
        
//     //     let outputs = self.client
//     //         .infer_http("coord_encoder".to_string(), "1".to_string(), inputs, vec!["output"])
//     //         .await?;
        
//     //     *hidden = outputs.get("output").ok_or(Error::new("Missing output"))?.data.clone();
//     //     Ok((x_center, y_center, width, height))
//     // }

//     fn argmax_normalize(&self, logits: &ArrayD<f32>) -> Result<f32> {
//         let shape = logits.shape();
//         let flat_view = logits.view().into_shape(shape[shape.len()-1])?;
        
//         let max_idx = flat_view
//             .iter()
//             .enumerate()
//             .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
//             .map(|(i, _)| i)
//             .unwrap();
            
//         Ok(max_idx as f32 / shape[shape.len()-1] as f32)
//     }
// }
