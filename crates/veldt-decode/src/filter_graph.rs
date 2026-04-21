/// Specification for the ffmpeg filter graph applied during decode.
#[derive(Debug, Clone)]
pub struct FilterSpec {
    /// Target width.
    pub width: u32,
    /// Target height.
    pub height: u32,
    /// If true, output f32 RGB normalized by mean/std. If false, output u8 RGB.
    pub normalize: bool,
    /// Per-channel mean for normalization (RGB order).
    pub mean: [f32; 3],
    /// Per-channel std for normalization (RGB order).
    pub std: [f32; 3],
}

impl Default for FilterSpec {
    fn default() -> Self {
        FilterSpec {
            width: 224,
            height: 224,
            normalize: false,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }
}
