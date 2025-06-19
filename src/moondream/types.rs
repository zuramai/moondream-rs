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

#[derive(Debug)]
pub struct Object {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub x_center: f32,
    pub y_center: f32,
}

impl Object {
    pub fn new(x_center: f32, y_center: f32, w: f32, h: f32) -> Self {
        Self {
            x: x_center - w / 2.0,
            y: y_center - h / 2.0,
            w,
            h,
            x_center: x_center,
            y_center: y_center,
        }
    }
    pub fn scale(&self, scale_w: f32, scale_h: f32) -> Self {
        Self {
            x: self.x * scale_w,
            y: self.y * scale_h,
            w: self.w * scale_w,
            h: self.h * scale_h,
            x_center: self.x_center * scale_w,
            y_center: self.y_center * scale_h,
        }
    }
}