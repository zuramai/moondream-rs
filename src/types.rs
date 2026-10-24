//! Public types shared by every backend: inputs, outputs, and the [`Vlm`] trait.

use image::DynamicImage;
use serde::Serialize;

use crate::Result;

/// Caption length variant; each maps to a prompt template in the model config.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CaptionLength {
    /// Short caption.
    Short,
    /// Normal caption (default).
    #[default]
    Normal,
    /// Long caption (not supported by every model).
    Long,
}

impl CaptionLength {
    /// Template key used in model configs.
    pub fn as_str(&self) -> &'static str {
        match self {
            CaptionLength::Short => "short",
            CaptionLength::Normal => "normal",
            CaptionLength::Long => "long",
        }
    }
}

/// Text generation settings. `None` fields use backend defaults.
///
/// The ONNX backend decodes greedily and ignores `temperature` and `top_p`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SamplingSettings {
    /// Maximum number of generated tokens.
    pub max_tokens: Option<usize>,
    /// Sampling temperature; `0.0` decodes greedily.
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f32>,
}

/// Output of [`Vlm::caption`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CaptionOutput {
    /// Generated caption.
    pub caption: String,
}

/// Output of [`Vlm::query`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QueryOutput {
    /// Generated answer.
    pub answer: String,
}

/// Bounding box in normalized image coordinates.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Region {
    /// Left edge.
    pub x_min: f64,
    /// Top edge.
    pub y_min: f64,
    /// Right edge.
    pub x_max: f64,
    /// Bottom edge.
    pub y_max: f64,
}

/// Output of [`Vlm::detect`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DetectOutput {
    /// Detected objects.
    pub objects: Vec<Region>,
}

/// Point in normalized image coordinates.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Point {
    /// Horizontal coordinate.
    pub x: f64,
    /// Vertical coordinate.
    pub y: f64,
}

/// Output of [`Vlm::point`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PointOutput {
    /// Located points.
    pub points: Vec<Point>,
}

/// Image already run through a backend's vision and text prefill stages.
///
/// Only valid for the backend that produced it; not stable across versions.
#[derive(Clone, Debug)]
pub enum EncodedImage {
    /// Produced by [`crate::OnnxVL`].
    Onnx(crate::onnx::OnnxEncodedImage),
    /// Produced by [`crate::TorchVL`].
    Torch(crate::torch::TorchEncodedImage),
}

/// Image argument accepted by [`Vlm`] methods.
#[derive(Clone, Copy, Debug)]
pub enum ImageInput<'a> {
    /// Raw image; encoded on every call.
    Image(&'a DynamicImage),
    /// Previously encoded image; skips the vision stage.
    Encoded(&'a EncodedImage),
}

impl<'a> From<&'a DynamicImage> for ImageInput<'a> {
    fn from(image: &'a DynamicImage) -> Self {
        ImageInput::Image(image)
    }
}

impl<'a> From<&'a EncodedImage> for ImageInput<'a> {
    fn from(image: &'a EncodedImage) -> Self {
        ImageInput::Encoded(image)
    }
}

/// Lazily generated text chunks.
pub type TokenStream<'a> = Box<dyn Iterator<Item = Result<String>> + 'a>;

/// Vision-language model interface.
///
/// `*_stream` methods validate input and encode the image eagerly, so errors
/// surface at the call; text is generated as the iterator is consumed.
pub trait Vlm: Send + Sync {
    /// Encodes an image for reuse across calls.
    fn encode_image(&self, image: ImageInput<'_>) -> Result<EncodedImage>;

    /// Streams a caption.
    fn caption_stream<'a>(
        &'a self,
        image: ImageInput<'_>,
        length: CaptionLength,
        settings: Option<SamplingSettings>,
    ) -> Result<TokenStream<'a>>;

    /// Streams an answer to `question`.
    fn query_stream<'a>(
        &'a self,
        image: ImageInput<'_>,
        question: &str,
        settings: Option<SamplingSettings>,
    ) -> Result<TokenStream<'a>>;

    /// Detects bounding boxes of `object`.
    fn detect(&self, image: ImageInput<'_>, object: &str) -> Result<DetectOutput>;

    /// Locates center points of `object`.
    fn point(&self, image: ImageInput<'_>, object: &str) -> Result<PointOutput>;

    /// Generates a full caption.
    fn caption(
        &self,
        image: ImageInput<'_>,
        length: CaptionLength,
        settings: Option<SamplingSettings>,
    ) -> Result<CaptionOutput> {
        let caption = self
            .caption_stream(image, length, settings)?
            .collect::<Result<String>>()?;
        Ok(CaptionOutput { caption })
    }

    /// Generates a full answer to `question`.
    fn query(
        &self,
        image: ImageInput<'_>,
        question: &str,
        settings: Option<SamplingSettings>,
    ) -> Result<QueryOutput> {
        let answer = self
            .query_stream(image, question, settings)?
            .collect::<Result<String>>()?;
        Ok(QueryOutput { answer })
    }
}
