//! Candle backend for `.safetensors` weights (`torch_vl.py`, `moondream/torch/*`).

mod config;
mod generate;
mod image_crops;
mod region;
mod text;
mod vision;
mod weights;

use std::{fmt, path::Path};

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear, Linear, VarBuilder};
use half::f16;
use parking_lot::{Mutex, MutexGuard};
use tokenizers::{FromPretrainedParameters, Tokenizer};

use self::{
    config::{PrefixSuffix, TorchConfig},
    generate::{argmax, sample, StreamDetokenizer},
    region::RegionModel,
    text::{KvCaches, TextModel},
    vision::VisionModel,
};
use crate::{
    CaptionLength, DetectOutput, EncodedImage, Error, ImageInput, Point, PointOutput, Region,
    Result, SamplingSettings, TokenStream, Vlm,
};

const DEFAULT_MAX_TOKENS: usize = 768;
const DEFAULT_TEMPERATURE: f32 = 0.5;
const DEFAULT_TOP_P: f32 = 0.3;
const MAX_OBJECTS: usize = 50;

/// Two-layer MLP with tanh-approximated GELU.
pub(crate) struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu()?)
    }
}

pub(crate) fn mlp(input: usize, inner: usize, output: usize, vb: VarBuilder) -> Result<Mlp> {
    Ok(Mlp {
        fc1: linear(input, inner, vb.pp("fc1"))?,
        fc2: linear(inner, output, vb.pp("fc2"))?,
    })
}

/// Image encoded by [`TorchVL`]: per-layer KV cache of the image prefix.
#[derive(Clone)]
pub struct TorchEncodedImage {
    pos: usize,
    caches: Vec<(Tensor, Tensor)>,
}

impl fmt::Debug for TorchEncodedImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TorchEncodedImage")
            .field("pos", &self.pos)
            .field("layers", &self.caches.len())
            .finish()
    }
}

/// Moondream running on candle (CPU, CUDA, or Metal).
pub struct TorchVL {
    vision: VisionModel,
    text: TextModel,
    region: RegionModel,
    config: TorchConfig,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    /// Working KV cache; one generation at a time (as in the reference model).
    caches: Mutex<KvCaches>,
}

fn select_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    #[cfg(feature = "metal")]
    if let Ok(d) = Device::new_metal(0) {
        return d;
    }
    Device::Cpu
}

fn to_f16_f32(v: f32) -> f32 {
    f16::from_f32(v).to_f32()
}

impl TorchVL {
    /// Loads weights from a `.safetensors` file. The tokenizer is fetched from
    /// the Hugging Face hub (`vikhyatk/moondream2@2025-01-09`, cached locally).
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let config = TorchConfig::default();
        let device = select_device();
        // Candle CPU matmuls are fastest in f32; accelerators use f16.
        let dtype = if device.is_cpu() { DType::F32 } else { DType::F16 };
        let w = weights::load(model_path.as_ref(), dtype, &device)?;
        let vb = w.vb;
        let vision = VisionModel::load(vb.pp("vision"), &config.vision)?;
        let text = TextModel::load(vb.pp("text"), &config.text)?;
        let region = RegionModel::load(vb.pp("region"), &config.region, w.legacy)?;
        let tokenizer = Tokenizer::from_pretrained(
            "vikhyatk/moondream2",
            Some(FromPretrainedParameters {
                revision: "2025-01-09".into(),
                ..Default::default()
            }),
        )?;
        let caches = Mutex::new(text.new_caches(dtype, &device)?);
        Ok(Self {
            vision,
            text,
            region,
            config,
            tokenizer,
            device,
            dtype,
            caches,
        })
    }

    /// Vision encoder + projection: image embeddings `[729, dim]`.
    fn image_embeddings(&self, image: &image::DynamicImage) -> Result<Tensor> {
        let c = &self.config.vision;
        let (crops, n, tiling) =
            image_crops::overlap_crop_image(&image.to_rgb8(), c.max_crops, c.overlap_margin)?;
        let g = image_crops::GRID;
        let patches = Tensor::from_vec(crops, (n, g * g, image_crops::PATCH_DIM), &self.device)?
            .to_dtype(self.dtype)?;
        let out = self.vision.encode(&patches)?;
        let global = out.get(0)?;
        let local = out.narrow(0, 1, n - 1)?.reshape((n - 1, g, g, c.enc_dim))?;
        let reconstructed = image_crops::reconstruct_from_crops(&local, tiling, c.overlap_margin)?;
        self.vision.project(&global, &reconstructed)
    }

    /// BOS + image embeddings `[1, 730, dim]`; runs without the cache lock.
    fn prefix_embeddings(&self, image: &image::DynamicImage) -> Result<Tensor> {
        let img = self.image_embeddings(image)?.unsqueeze(0)?;
        let bos = self.text.embed(&[self.config.tokenizer.bos_id])?;
        Ok(Tensor::cat(&[&bos, &img], 1)?)
    }

    /// Encodes `image` (if needed) and loads its prefix into the working cache.
    fn load_image(&self, image: ImageInput<'_>) -> Result<(MutexGuard<'_, KvCaches>, usize)> {
        match image {
            ImageInput::Image(img) => {
                let x = self.prefix_embeddings(img)?;
                let caches = self.caches.lock();
                self.text.forward(&x, &caches, 0)?;
                Ok((caches, x.dim(1)?))
            }
            ImageInput::Encoded(EncodedImage::Torch(enc)) => {
                let caches = self.caches.lock();
                for ((k, v), (ek, ev)) in caches.iter().zip(&enc.caches) {
                    k.slice_set(ek, 2, 0)?;
                    v.slice_set(ev, 2, 0)?;
                }
                Ok((caches, enc.pos))
            }
            ImageInput::Encoded(EncodedImage::Onnx(_)) => Err(Error::EncodedImageMismatch),
        }
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode(text, true)?.get_ids().to_vec())
    }

    fn prompt(&self, t: &PrefixSuffix, text: &str) -> Result<Vec<u32>> {
        let mut p = t.prefix.clone();
        p.extend(self.tokenize(&format!(" {text}"))?);
        p.extend_from_slice(&t.suffix);
        Ok(p)
    }

    fn text_stream<'a>(
        &'a self,
        image: ImageInput<'_>,
        prompt: &[u32],
        settings: Option<SamplingSettings>,
    ) -> Result<TokenStream<'a>> {
        let s = settings.unwrap_or_default();
        let max_tokens = s.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        let temperature = s.temperature.unwrap_or(DEFAULT_TEMPERATURE);
        let top_p = s.top_p.unwrap_or(DEFAULT_TOP_P);
        let (caches, pos) = self.load_image(image)?;
        let hidden = self.text.forward(&self.text.embed(prompt)?, &caches, pos)?;
        let mut rng = rand::thread_rng();
        let next = sample(&self.text.logits(&hidden)?, temperature, top_p, &mut rng);
        Ok(Box::new(TorchTextStream {
            vl: self,
            caches,
            pos: pos + prompt.len(),
            next,
            generated: 0,
            max_tokens,
            temperature,
            top_p,
            detok: StreamDetokenizer::new(Box::new(move |ids: &[u32]| {
                Ok(self.tokenizer.decode(ids, true)?)
            })),
            finished: false,
            rng,
        }))
    }

    /// Greedy point/box decoding (`_generate_points`).
    fn locate(
        &self,
        image: ImageInput<'_>,
        template: Option<&PrefixSuffix>,
        what: &str,
        object: &str,
        include_size: bool,
    ) -> Result<(Vec<Region>, Vec<Point>)> {
        let t = template
            .ok_or_else(|| Error::Unsupported(format!("Model does not support {what}.")))?;
        let prompt = self.prompt(t, object)?;
        let (caches, mut pos) = self.load_image(image)?;
        let hidden = self.text.forward(&self.text.embed(&prompt)?, &caches, pos)?;
        let mut next = argmax(&self.text.logits(&hidden)?) as u32;
        let mut hidden = hidden.narrow(1, prompt.len() - 1, 1)?;
        pos += prompt.len();

        let eos = self.config.tokenizer.eos_id;
        let mut regions = Vec::new();
        let mut points = Vec::new();
        let step = |emb: &Tensor, pos: &mut usize| -> Result<Tensor> {
            let h = self.text.forward(emb, &caches, *pos)?;
            *pos += 1;
            Ok(h)
        };
        while next != eos && regions.len() + points.len() < MAX_OBJECTS {
            let x_logits = self.region.decode_coordinate(&hidden)?;
            let x = argmax(&x_logits) as f32 / x_logits.len() as f32;
            let emb = self.region.encode_coordinate(to_f16_f32(x))?;
            hidden = step(&emb, &mut pos)?;
            let y_logits = self.region.decode_coordinate(&hidden)?;
            let y = argmax(&y_logits) as f32 / y_logits.len() as f32;
            let mut emb = self.region.encode_coordinate(to_f16_f32(y))?;

            if include_size {
                hidden = step(&emb, &mut pos)?;
                let (w_logits, h_logits) = self.region.decode_size(&hidden)?;
                let bin = |b: usize| 2f32.powf((b as f32 / 1023.0) * 10.0 - 10.0);
                let w = bin(argmax(&w_logits));
                let h = bin(argmax(&h_logits));
                emb = self.region.encode_size(to_f16_f32(w), to_f16_f32(h))?;
                let (x, y, w, h) = (x as f64, y as f64, w as f64, h as f64);
                regions.push(Region {
                    x_min: x - w / 2.0,
                    y_min: y - h / 2.0,
                    x_max: x + w / 2.0,
                    y_max: y + h / 2.0,
                });
            } else {
                points.push(Point { x: x as f64, y: y as f64 });
            }
            hidden = step(&emb, &mut pos)?;
            next = argmax(&self.text.logits(&hidden)?) as u32;
        }
        Ok((regions, points))
    }
}

type DecodeFn<'a> = Box<dyn FnMut(&[u32]) -> Result<String> + 'a>;

/// Sampled token stream (`_generate_text`); holds the KV cache lock until dropped.
struct TorchTextStream<'a> {
    vl: &'a TorchVL,
    caches: MutexGuard<'a, KvCaches>,
    pos: usize,
    next: u32,
    generated: usize,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    detok: StreamDetokenizer<DecodeFn<'a>>,
    finished: bool,
    rng: rand::rngs::ThreadRng,
}

impl TorchTextStream<'_> {
    fn step(&mut self) -> Result<Option<String>> {
        while !self.finished {
            if self.next == self.vl.config.tokenizer.eos_id || self.generated >= self.max_tokens {
                self.finished = true;
                return self.detok.finish();
            }
            let token = self.next;
            let printable = self.detok.push(token)?;
            let emb = self.vl.text.embed(&[token])?;
            let hidden = self.vl.text.forward(&emb, &self.caches, self.pos)?;
            self.pos += 1;
            let logits = self.vl.text.logits(&hidden)?;
            self.next = sample(&logits, self.temperature, self.top_p, &mut self.rng);
            self.generated += 1;
            if printable.is_some() {
                return Ok(printable);
            }
        }
        Ok(None)
    }
}

impl Iterator for TorchTextStream<'_> {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let item = self.step().transpose();
        if matches!(item, Some(Err(_))) {
            self.finished = true;
        }
        item
    }
}

impl Vlm for TorchVL {
    fn encode_image(&self, image: ImageInput<'_>) -> Result<EncodedImage> {
        if let ImageInput::Encoded(EncodedImage::Torch(enc)) = image {
            return Ok(EncodedImage::Torch(enc.clone()));
        }
        let (caches, pos) = self.load_image(image)?;
        // `copy` detaches the snapshot from storage later mutated by `slice_set`.
        let caches = caches
            .iter()
            .map(|(k, v)| Ok((k.narrow(2, 0, pos)?.copy()?, v.narrow(2, 0, pos)?.copy()?)))
            .collect::<Result<_>>()?;
        Ok(EncodedImage::Torch(TorchEncodedImage { pos, caches }))
    }

    fn caption_stream<'a>(
        &'a self,
        image: ImageInput<'_>,
        length: CaptionLength,
        settings: Option<SamplingSettings>,
    ) -> Result<TokenStream<'a>> {
        let templates = self
            .config
            .tokenizer
            .caption
            .as_ref()
            .ok_or_else(|| Error::Unsupported("Model does not support captioning.".into()))?;
        let prompt = templates.get(length.as_str()).ok_or_else(|| {
            Error::Unsupported(format!(
                "Model does not support caption length '{}'.",
                length.as_str()
            ))
        })?;
        self.text_stream(image, prompt, settings)
    }

    fn query_stream<'a>(
        &'a self,
        image: ImageInput<'_>,
        question: &str,
        settings: Option<SamplingSettings>,
    ) -> Result<TokenStream<'a>> {
        let t = self
            .config
            .tokenizer
            .query
            .as_ref()
            .ok_or_else(|| Error::Unsupported("Model does not support querying.".into()))?;
        let prompt = self.prompt(t, question)?;
        self.text_stream(image, &prompt, settings)
    }

    fn detect(&self, image: ImageInput<'_>, object: &str) -> Result<DetectOutput> {
        let t = self.config.tokenizer.detect.as_ref();
        let (objects, _) = self.locate(image, t, "object detection", object, true)?;
        Ok(DetectOutput { objects })
    }

    fn point(&self, image: ImageInput<'_>, object: &str) -> Result<PointOutput> {
        let t = self.config.tokenizer.point.as_ref();
        let (_, points) = self.locate(image, t, "pointing", object, false)?;
        Ok(PointOutput { points })
    }
}
