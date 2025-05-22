use half::f16;

#[derive(Debug)]
pub struct EncodedImage {
    pub pos: usize,
    pub kv_cache: ndarray::ArrayD<f16>
}

pub enum CaptionLength {
    Short,
    Normal,
}