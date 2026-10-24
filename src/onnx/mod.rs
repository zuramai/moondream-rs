//! ONNX Runtime backend (`onnx_vl.py`).

mod kv;
mod preprocess;

use std::{
    borrow::Cow,
    collections::HashMap,
    fmt,
    path::{Path, PathBuf},
};

use half::f16;
use ort::{
    environment::Environment,
    ep::ExecutionProviderDispatch,
    session::{
        builder::{AutoDevicePolicy, GraphOptimizationLevel},
        Session,
    },
    value::TensorRef,
};
use parking_lot::Mutex;
use serde::Deserialize;
use tokenizers::Tokenizer;

use self::{
    kv::KvBuffer,
    preprocess::{adaptive_avg_pool2d_numpy, create_patches},
};
use crate::{
    moonfile, CaptionLength, DetectOutput, EncodedImage, Error, ImageInput, Point, PointOutput,
    Region, Result, SamplingSettings, TokenStream, Vlm,
};

const DEFAULT_MAX_TOKENS: usize = 512;
const MAX_OBJECTS: usize = 50;

const SESSION_KEYS: [&str; 8] = [
    "vision_encoder",
    "vision_projection",
    "text_encoder",
    "text_decoder",
    "size_encoder",
    "size_decoder",
    "coord_encoder",
    "coord_decoder",
];

#[derive(Debug, Deserialize)]
struct PrefixSuffix {
    prefix: Vec<u32>,
    suffix: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct Templates {
    caption: Option<HashMap<String, Vec<u32>>>,
    query: Option<PrefixSuffix>,
    detect: Option<PrefixSuffix>,
    point: Option<PrefixSuffix>,
}

#[derive(Debug, Deserialize)]
struct SpecialTokens {
    eos: u32,
}

#[derive(Debug, Deserialize)]
struct OnnxConfig {
    templates: Templates,
    special_tokens: SpecialTokens,
}

/// Image encoded by [`OnnxVL`]: the text decoder's KV cache after the image prefix.
#[derive(Clone)]
pub struct OnnxEncodedImage {
    /// Number of cached positions.
    pub(crate) pos: usize,
    /// `[rows, pos, head_dim]` cache, `rows = layers * 2 * batch * heads`.
    pub(crate) kv_cache: Vec<f16>,
    pub(crate) rows: usize,
    pub(crate) head_dim: usize,
    /// Decoder kv tensor shape (the `seq` entry is ignored).
    pub(crate) kv_shape: [usize; 6],
}

impl fmt::Debug for OnnxEncodedImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OnnxEncodedImage")
            .field("pos", &self.pos)
            .field("kv_shape", &self.kv_shape)
            .finish_non_exhaustive()
    }
}

/// Moondream running on ONNX Runtime.
pub struct OnnxVL {
    vision_encoder: Mutex<Session>,
    vision_projection: Mutex<Session>,
    text_encoder: Mutex<Session>,
    text_decoder: Mutex<Session>,
    size_encoder: Mutex<Session>,
    size_decoder: Mutex<Session>,
    coord_encoder: Mutex<Session>,
    coord_decoder: Mutex<Session>,
    tokenizer: Tokenizer,
    initial_kv_cache: Vec<f16>,
    kv_shape: [usize; 6],
    config: OnnxConfig,
}

enum Source<'a> {
    Memory(&'a [u8]),
    File(PathBuf),
}

fn execution_providers() -> Vec<ExecutionProviderDispatch> {
    vec![
        #[cfg(feature = "cuda")]
        ort::ep::CUDA::default().build(),
        #[cfg(feature = "coreml")]
        ort::ep::CoreML::default().build(),
    ]
}

/// The process-wide ONNX Runtime environment (ort allows only one).
fn environment() -> Result<Environment> {
    static ENV: Mutex<Option<Environment>> = Mutex::new(None);
    let mut env = ENV.lock();
    if let Some(env) = env.as_ref() {
        return Ok(env.clone());
    }
    let built = ort::init().with_name("moondream").build()?;
    // Like `ort.set_default_logger_severity(3)` in the Python client; ORT
    // otherwise formats verbose messages during session creation.
    built.set_log_level(ort::logging::LogLevel::Error);
    *env = Some(built.clone());
    Ok(built)
}

fn build_session(env: &Environment, source: &Source<'_>) -> Result<Session> {
    // Builder setters return `ort::Error<SessionBuilder>`; erase the payload.
    fn erase(e: ort::Error<ort::session::builder::SessionBuilder>) -> Error {
        Error::Ort(e.into())
    }
    // Python's ORT_ENABLE_ALL (`Level3` is only ENABLE_LAYOUT). ort defaults to
    // the `MaxEfficiency` device policy, which can route models to slower
    // devices; `PreferCPU` matches the Python client unless an accelerator
    // feature registers explicit providers.
    let builder = Session::builder(env)?
        .with_log_level(ort::logging::LogLevel::Error)
        .map_err(erase)?
        .with_auto_device(AutoDevicePolicy::PreferCPU)
        .map_err(erase)?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(erase)?;
    let mut builder = builder
        .with_execution_providers(execution_providers())
        .map_err(erase)?;
    Ok(match source {
        Source::Memory(bytes) => builder.commit_from_memory(bytes)?,
        Source::File(path) => builder.commit_from_file(path)?,
    })
}

fn missing(key: &str) -> Error {
    Error::InvalidModel(format!("Missing model component: {key}"))
}

/// Index of the first maximum (numpy `argmax` semantics).
fn argmax<T: Copy + PartialOrd>(xs: &[T]) -> usize {
    let mut best = 0;
    for (i, &x) in xs.iter().enumerate() {
        if x > xs[best] {
            best = i;
        }
    }
    best
}

struct DecoderOut {
    token: u32,
    last_hidden: Vec<f16>,
}

impl OnnxVL {
    /// Loads a `.mf` / `.mf.gz` bundle or an unpacked model directory.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.is_dir() {
            let read = |name: &str| -> Result<Vec<u8>> {
                let p = path.join(name);
                if !p.is_file() {
                    return Err(missing(name.split('.').next().unwrap_or(name)));
                }
                Ok(std::fs::read(p)?)
            };
            let mut sources = HashMap::new();
            for key in SESSION_KEYS {
                let p = path.join(format!("{key}.onnx"));
                if !p.is_file() {
                    return Err(missing(key));
                }
                sources.insert(key, Source::File(p));
            }
            return Self::assemble(
                sources,
                &read("tokenizer.json")?,
                &read("initial_kv_cache.npy")?,
                &read("config.json")?,
            );
        }
        if !path.is_file() {
            return Err(Error::InvalidModel(
                "Model path is invalid or file does not exist.".into(),
            ));
        }
        let bytes = moonfile::read(path)?;
        let mut components: HashMap<&str, &[u8]> = HashMap::new();
        for (name, content) in moonfile::entries(&bytes)? {
            let base = name.rsplit('/').next().unwrap_or(name);
            let key = base.split('.').next().unwrap_or(base);
            components.insert(key, content);
        }
        let get = |k: &str| components.get(k).copied().ok_or_else(|| missing(k));
        let mut sources = HashMap::new();
        for key in SESSION_KEYS {
            sources.insert(key, Source::Memory(get(key)?));
        }
        Self::assemble(
            sources,
            get("tokenizer")?,
            get("initial_kv_cache")?,
            get("config")?,
        )
    }

    fn assemble(
        sources: HashMap<&'static str, Source<'_>>,
        tokenizer: &[u8],
        initial_kv: &[u8],
        config: &[u8],
    ) -> Result<Self> {
        let config: OnnxConfig = serde_json::from_slice(config)?;
        let tokenizer = Tokenizer::from_bytes(tokenizer)?;
        let npy = npyz::NpyFile::new(initial_kv)?;
        let shape: Vec<usize> = npy.shape().iter().map(|&d| d as usize).collect();
        let kv_shape: [usize; 6] = shape.as_slice().try_into().map_err(|_| {
            Error::InvalidModel(format!("initial_kv_cache must be 6-D, got {shape:?}"))
        })?;
        let initial_kv_cache = npy.into_vec::<f16>()?;

        // Session creation dominates load time; build all of them concurrently.
        let env = environment()?;
        let env = &env;
        let mut sessions: HashMap<&str, Session> = std::thread::scope(|s| {
            let handles: Vec<_> = sources
                .iter()
                .map(|(&key, src)| (key, s.spawn(move || build_session(env, src))))
                .collect();
            handles
                .into_iter()
                .map(|(key, h)| {
                    let session = h
                        .join()
                        .map_err(|_| Error::InvalidModel(format!("failed to load {key}")))??;
                    Ok((key, session))
                })
                .collect::<Result<_>>()
        })?;
        let mut take = |k: &str| sessions.remove(k).map(Mutex::new).ok_or_else(|| missing(k));
        Ok(Self {
            vision_encoder: take("vision_encoder")?,
            vision_projection: take("vision_projection")?,
            text_encoder: take("text_encoder")?,
            text_decoder: take("text_decoder")?,
            size_encoder: take("size_encoder")?,
            size_decoder: take("size_decoder")?,
            coord_encoder: take("coord_encoder")?,
            coord_decoder: take("coord_decoder")?,
            tokenizer,
            initial_kv_cache,
            kv_shape,
            config,
        })
    }

    fn encode_tokens(&self, ids: &[u32]) -> Result<Vec<f16>> {
        let ids: Vec<i64> = ids.iter().map(|&t| t as i64).collect();
        let mut sess = self.text_encoder.lock();
        let out = sess.run(ort::inputs![
            "input_ids" => TensorRef::from_array_view(([1, ids.len()], ids.as_slice()))?
        ])?;
        Ok(out["input_embeds"].try_extract_tensor::<f16>()?.1.to_vec())
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode(text, true)?.get_ids().to_vec())
    }

    fn run_decoder(
        &self,
        embeds: &[f16],
        kv: &mut KvBuffer,
        pos: usize,
    ) -> Result<(DecoderOut, usize)> {
        let dim = self.model_dim();
        let seq = self.seq_len(embeds)?;
        let (kv_shape, kv_data) = kv.prefix(pos);
        let mut sess = self.text_decoder.lock();
        let out = sess.run(ort::inputs![
            "input_embeds" => TensorRef::from_array_view(([1, seq, dim], embeds))?,
            "kv_cache" => TensorRef::from_array_view((kv_shape, kv_data))?,
        ])?;
        let (_, logits) = out["logits"].try_extract_tensor::<f16>()?;
        let token = argmax(logits) as u32;
        let (_, hidden) = out["hidden"].try_extract_tensor::<f16>()?;
        let last_hidden = hidden[hidden.len() - dim..].to_vec();
        let (_, update) = out["new_kv_cache"].try_extract_tensor::<f16>()?;
        kv.write(pos, seq, update);
        Ok((DecoderOut { token, last_hidden }, seq))
    }

    /// Number of embeddings in `embeds` given the model width.
    fn seq_len(&self, embeds: &[f16]) -> Result<usize> {
        let dim = self.model_dim();
        if dim == 0 || embeds.is_empty() || !embeds.len().is_multiple_of(dim) {
            return Err(Error::InvalidModel("embedding size mismatch".into()));
        }
        Ok(embeds.len() / dim)
    }

    /// Text model width, derived from the KV cache (`heads * head_dim`).
    fn model_dim(&self) -> usize {
        self.kv_shape[3] * self.kv_shape[5]
    }

    fn run_vector(&self, session: &Mutex<Session>, input: &[f16]) -> Result<Vec<f16>> {
        let mut sess = session.lock();
        let out = sess.run(ort::inputs![
            "input" => TensorRef::from_array_view(([input.len()], input))?
        ])?;
        Ok(out["output"].try_extract_tensor::<f16>()?.1.to_vec())
    }

    fn encode_raw(&self, image: &image::DynamicImage) -> Result<OnnxEncodedImage> {
        let (patches, n, template) = create_patches(image)?;
        self.encode_patches(&patches, n, template)
    }

    /// Vision encoder → projection → text prefill for `[n, 3, 378, 378]` patches.
    fn encode_patches(
        &self,
        patches: &[f16],
        n: usize,
        template: (usize, usize),
    ) -> Result<OnnxEncodedImage> {
        let s = preprocess::PATCH_SIZE;
        let (feat, seq, enc_dim) = {
            let mut sess = self.vision_encoder.lock();
            let out = sess.run(ort::inputs![
                "input" => TensorRef::from_array_view(([n, 3, s, s], patches))?
            ])?;
            let (shape, data) = out["output"].try_extract_tensor::<f16>()?;
            (data.to_vec(), shape[1] as usize, shape[2] as usize)
        };
        let global = &feat[..seq * enc_dim];
        let grid = (seq as f64).sqrt() as usize;
        let local: Cow<'_, [f16]> = if template == (1, 1) {
            Cow::Borrowed(global)
        } else {
            let (rows, cols) = template;
            let (gh, gw) = (rows * grid, cols * grid);
            let mut full = vec![f16::ZERO; gh * gw * enc_dim];
            for r in 0..rows {
                for c in 0..cols {
                    // Local crops start at index 1 (index 0 is the global patch).
                    // `onnx_vl.py` indexes from 0, feeding the global patch as the
                    // first tile and dropping the last crop; this follows the
                    // reference torch model (`outputs[1:]`) instead.
                    let patch = &feat[(1 + r * cols + c) * seq * enc_dim..][..seq * enc_dim];
                    for y in 0..grid {
                        let dst = ((r * grid + y) * gw + c * grid) * enc_dim;
                        full[dst..dst + grid * enc_dim]
                            .copy_from_slice(&patch[y * grid * enc_dim..][..grid * enc_dim]);
                    }
                }
            }
            Cow::Owned(adaptive_avg_pool2d_numpy(&full, gh, gw, enc_dim, grid, grid))
        };
        let mut joined = Vec::with_capacity(seq * 2 * enc_dim);
        for (g, l) in global.chunks_exact(enc_dim).zip(local.chunks_exact(enc_dim)) {
            joined.extend_from_slice(g);
            joined.extend_from_slice(l);
        }

        let embeds = {
            let mut sess = self.vision_projection.lock();
            let out = sess.run(ort::inputs![
                "input" => TensorRef::from_array_view(([1, seq, 2 * enc_dim], joined.as_slice()))?
            ])?;
            out["output"].try_extract_tensor::<f16>()?.1.to_vec()
        };

        let dim = self.model_dim();
        let img_seq = embeds.len() / dim;
        let init_seq = self.kv_shape[4];
        let hd = self.kv_shape[5];
        let rows: usize = self.kv_shape[..4].iter().product();
        let pos = img_seq + init_seq;
        let mut sess = self.text_decoder.lock();
        let out = sess.run(ort::inputs![
            "input_embeds" => TensorRef::from_array_view(([1, img_seq, dim], embeds.as_slice()))?,
            "kv_cache" => TensorRef::from_array_view((self.kv_shape, self.initial_kv_cache.as_slice()))?,
        ])?;
        let (_, update) = out["new_kv_cache"].try_extract_tensor::<f16>()?;
        let mut kv_cache = Vec::with_capacity(rows * pos * hd);
        for (init, new) in self
            .initial_kv_cache
            .chunks_exact(init_seq * hd)
            .zip(update.chunks_exact(img_seq * hd))
        {
            kv_cache.extend_from_slice(init);
            kv_cache.extend_from_slice(new);
        }
        Ok(OnnxEncodedImage {
            pos,
            kv_cache,
            rows,
            head_dim: hd,
            kv_shape: self.kv_shape,
        })
    }

    fn onnx_image<'a>(&self, image: ImageInput<'a>) -> Result<Cow<'a, OnnxEncodedImage>> {
        match image {
            ImageInput::Image(img) => Ok(Cow::Owned(self.encode_raw(img)?)),
            ImageInput::Encoded(EncodedImage::Onnx(enc)) => Ok(Cow::Borrowed(enc)),
            ImageInput::Encoded(EncodedImage::Torch(_)) => Err(Error::EncodedImageMismatch),
        }
    }

    fn stream<'a>(
        &'a self,
        embeds: Vec<f16>,
        image: ImageInput<'_>,
        settings: Option<SamplingSettings>,
    ) -> Result<TokenStream<'a>> {
        let max_tokens = settings
            .and_then(|s| s.max_tokens)
            .unwrap_or(DEFAULT_MAX_TOKENS);
        let enc = self.onnx_image(image)?;
        Ok(Box::new(OnnxTextStream {
            vl: self,
            kv: KvBuffer::from_encoded(&enc),
            pos: enc.pos,
            embeds,
            generated: 0,
            max_tokens,
            pending_token: None,
            done: false,
        }))
    }

    fn locate(
        &self,
        image: ImageInput<'_>,
        object: &str,
        template: Option<&PrefixSuffix>,
        name: &str,
        include_size: bool,
    ) -> Result<(Vec<Region>, Vec<Point>)> {
        let t = template
            .ok_or_else(|| Error::Unsupported(format!("Model does not support '{name}'.")))?;
        let mut prompt = t.prefix.clone();
        prompt.extend(self.tokenize(&format!(" {object}"))?);
        prompt.extend_from_slice(&t.suffix);
        let mut hidden = self.encode_tokens(&prompt)?;
        let enc = self.onnx_image(image)?;
        let mut kv = KvBuffer::from_encoded(&enc);
        let mut pos = enc.pos;
        drop(enc);

        let mut regions = Vec::new();
        let mut points = Vec::new();
        while regions.len() + points.len() < MAX_OBJECTS {
            let (out, n) = self.run_decoder(&hidden, &mut kv, pos)?;
            pos += n;
            if out.token == self.config.special_tokens.eos {
                break;
            }
            let x_logits = self.run_vector(&self.coord_decoder, &out.last_hidden)?;
            let x = argmax(&x_logits) as f64 / x_logits.len() as f64;
            hidden = self.run_vector(&self.coord_encoder, &[f16::from_f64(x)])?;

            let (out, n) = self.run_decoder(&hidden, &mut kv, pos)?;
            pos += n;
            let y_logits = self.run_vector(&self.coord_decoder, &out.last_hidden)?;
            let y = argmax(&y_logits) as f64 / y_logits.len() as f64;
            hidden = self.run_vector(&self.coord_encoder, &[f16::from_f64(y)])?;

            if !include_size {
                points.push(Point { x, y });
                continue;
            }
            let (out, n) = self.run_decoder(&hidden, &mut kv, pos)?;
            pos += n;
            let size = self.run_vector(&self.size_decoder, &out.last_hidden)?;
            let bins = size.len() / 2;
            let w = argmax(&size[..bins]) as f64 / bins as f64;
            let h = argmax(&size[bins..]) as f64 / bins as f64;
            hidden = self.run_vector(&self.size_encoder, &[f16::from_f64(w), f16::from_f64(h)])?;
            regions.push(Region {
                x_min: x - w / 2.0,
                y_min: y - h / 2.0,
                x_max: x + w / 2.0,
                y_max: y + h / 2.0,
            });
        }
        Ok((regions, points))
    }
}

/// Greedy token stream over the ONNX text decoder (`OnnxVL._generate`).
struct OnnxTextStream<'a> {
    vl: &'a OnnxVL,
    kv: KvBuffer,
    pos: usize,
    embeds: Vec<f16>,
    generated: usize,
    max_tokens: usize,
    pending_token: Option<u32>,
    done: bool,
}

impl OnnxTextStream<'_> {
    fn step(&mut self) -> Result<Option<String>> {
        if let Some(token) = self.pending_token.take() {
            self.embeds = self.vl.encode_tokens(&[token])?;
        }
        if self.generated >= self.max_tokens {
            return Ok(None);
        }
        let (out, n) = self.vl.run_decoder(&self.embeds, &mut self.kv, self.pos)?;
        self.pos += n;
        if out.token == self.vl.config.special_tokens.eos {
            return Ok(None);
        }
        self.generated += 1;
        self.pending_token = Some(out.token);
        Ok(Some(self.vl.tokenizer.decode(&[out.token], true)?))
    }
}

impl Iterator for OnnxTextStream<'_> {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let item = self.step().transpose();
        if !matches!(item, Some(Ok(_))) {
            self.done = true;
        }
        item
    }
}

impl Vlm for OnnxVL {
    fn encode_image(&self, image: ImageInput<'_>) -> Result<EncodedImage> {
        Ok(EncodedImage::Onnx(self.onnx_image(image)?.into_owned()))
    }

    fn caption_stream<'a>(
        &'a self,
        image: ImageInput<'_>,
        length: CaptionLength,
        settings: Option<SamplingSettings>,
    ) -> Result<TokenStream<'a>> {
        let templates = self
            .config
            .templates
            .caption
            .as_ref()
            .ok_or_else(|| Error::Unsupported("Model does not support captioning.".into()))?;
        let prompt = templates.get(length.as_str()).ok_or_else(|| {
            Error::Unsupported(format!(
                "Model does not support caption length '{}'.",
                length.as_str()
            ))
        })?;
        let embeds = self.encode_tokens(prompt)?;
        self.stream(embeds, image, settings)
    }

    fn query_stream<'a>(
        &'a self,
        image: ImageInput<'_>,
        question: &str,
        settings: Option<SamplingSettings>,
    ) -> Result<TokenStream<'a>> {
        let t = self
            .config
            .templates
            .query
            .as_ref()
            .ok_or_else(|| Error::Unsupported("Model does not support querying.".into()))?;
        let mut prompt = t.prefix.clone();
        prompt.extend(self.tokenize(question)?);
        prompt.extend_from_slice(&t.suffix);
        let embeds = self.encode_tokens(&prompt)?;
        self.stream(embeds, image, settings)
    }

    fn detect(&self, image: ImageInput<'_>, object: &str) -> Result<DetectOutput> {
        let t = self.config.templates.detect.as_ref();
        let (objects, _) = self.locate(image, object, t, "detect", true)?;
        Ok(DetectOutput { objects })
    }

    fn point(&self, image: ImageInput<'_>, object: &str) -> Result<PointOutput> {
        let t = self.config.templates.point.as_ref();
        let (_, points) = self.locate(image, object, t, "point", false)?;
        Ok(PointOutput { points })
    }
}
