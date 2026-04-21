use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum VeldtError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("MP4 parse error: {source} in {path}")]
    Mp4Parse {
        path: PathBuf,
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("VKF index error: {0}")]
    VkfIndex(String),

    #[error("Decode error: {0}")]
    Decode(String),

    #[error("Parquet error: {0}")]
    Parquet(String),

    #[error("Dataset error: {0}")]
    Dataset(String),

    #[error("Sample index {index} out of range (dataset has {len} samples)")]
    IndexOutOfRange { index: usize, len: usize },

    #[error("Frame {frame} not found in episode {episode}")]
    FrameNotFound { episode: u32, frame: u64 },

    #[error("Unsupported codec: {0}")]
    UnsupportedCodec(String),

    #[error("Missing sidecar index for {0} — run `veldt index` first")]
    MissingSidecar(PathBuf),

    #[error("Cache error: {0}")]
    Cache(String),
}

pub type Result<T> = std::result::Result<T, VeldtError>;
