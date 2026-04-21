use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;

/// Specifies which frames to extract from an episode.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ClipSpec {
    pub start_frame: u64,
    pub num_frames: u32,
    pub stride: u32,
    pub resolution: (u32, u32), // (height, width)
}

/// Owned frame data — either u8 [T, C, H, W] or f32 [T, C, H, W].
#[derive(Debug, Clone)]
pub enum FrameBuffer {
    U8 {
        data: Vec<u8>,
        shape: [usize; 4], // [T, C, H, W]
    },
    F32 {
        data: Vec<f32>,
        shape: [usize; 4], // [T, C, H, W]
    },
}

impl FrameBuffer {
    /// Number of frames in this buffer.
    pub fn num_frames(&self) -> usize {
        match self {
            FrameBuffer::U8 { shape, .. } => shape[0],
            FrameBuffer::F32 { shape, .. } => shape[0],
        }
    }

    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        match self {
            FrameBuffer::U8 { shape, .. } => shape.iter().product(),
            FrameBuffer::F32 { shape, .. } => shape.iter().product(),
        }
    }

    /// Shape as a slice.
    pub fn shape(&self) -> &[usize; 4] {
        match self {
            FrameBuffer::U8 { shape, .. } => shape,
            FrameBuffer::F32 { shape, .. } => shape,
        }
    }
}

/// A single column of tabular data (actions, states, etc.).
#[derive(Debug, Clone)]
pub struct TabularColumn {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// A single training sample returned by a reader.
#[derive(Debug, Clone)]
pub struct Sample {
    pub frames: FrameBuffer,
    pub tabular: HashMap<String, TabularColumn>,
    pub episode_index: u32,
    pub frame_index: u64,
}

/// Opaque fetch unit ID. Reader-defined, scheduler-opaque.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct FetchUnitId(pub u64);

/// A single fetch unit: the atomic I/O operation.
/// For MP4: a keyframe group (byte range within a file).
#[derive(Debug, Clone)]
pub struct FetchUnit {
    pub id: FetchUnitId,
    pub source: FetchSource,
    pub estimated_bytes: u64,
    /// Number of decoded frames this unit will produce.
    pub num_decoded_frames: u32,
}

/// Where the bytes for a fetch unit live.
#[derive(Debug, Clone)]
pub enum FetchSource {
    LocalFile {
        path: PathBuf,
        byte_offset: u64,
        byte_len: u64,
    },
    // Phase 2: Remote { url: String, byte_offset: u64, byte_len: u64 },
}

/// Output of DatasetReader::plan_fetch.
#[derive(Debug, Clone)]
pub struct FetchPlan {
    /// All unique fetch units needed for this epoch.
    pub units: Vec<FetchUnit>,

    /// For each request index, which FetchUnitIds are needed.
    pub request_to_units: Vec<Vec<FetchUnitId>>,

    /// Future-use queues: for each FetchUnitId, the sorted batch indices that need it.
    /// Built as a side effect of plan generation.
    pub unit_batches: HashMap<FetchUnitId, VecDeque<usize>>,
}

/// Dataset-level metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMeta {
    pub fps: f64,
    pub num_episodes: usize,
    pub num_frames: usize,
    pub camera_keys: Vec<String>,
    pub action_dim: usize,
    pub state_dim: usize,
}

/// A batch of samples ready for transfer to Python.
#[derive(Debug, Clone)]
pub struct Batch {
    /// Video frames: shape [B, T, C, H, W].
    pub frames: FrameBuffer,
    /// Tabular columns: each shaped [B, T, dim].
    pub tabular: HashMap<String, TabularColumn>,
    /// Number of samples in this batch.
    pub batch_size: usize,
}

/// Decoded frames for a single fetch unit, held in the cache.
#[derive(Debug, Clone)]
pub struct DecodedFrames {
    pub data: Vec<u8>,
    pub shape: [usize; 4], // [num_frames, C, H, W]
    pub unit_id: FetchUnitId,
}

impl DecodedFrames {
    /// Number of decoded frames in this unit.
    pub fn num_frames(&self) -> usize {
        self.shape[0]
    }
}

/// Statistics exposed by the engine / cache.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
    pub refetches: u64,
    pub current_frames: usize,
    pub buffer_size: usize,
}
