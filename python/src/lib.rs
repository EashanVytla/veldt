use std::path::PathBuf;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use veldt_core::{ClipSpec, DatasetReader, FrameBuffer};
use veldt_decode::FilterSpec;
use veldt_engine::{Engine, EngineConfig};
use veldt_reader_lerobot::LeRobotReader;

/// Configuration for the veldt Loader.
#[pyclass]
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub num_decode_threads: usize,
    #[pyo3(get, set)]
    pub buffer_size: usize,
    #[pyo3(get, set)]
    pub prefetch_depth: usize,
    #[pyo3(get, set)]
    pub width: u32,
    #[pyo3(get, set)]
    pub height: u32,
    #[pyo3(get, set)]
    pub normalize: bool,
    #[pyo3(get, set)]
    pub mean: (f32, f32, f32),
    #[pyo3(get, set)]
    pub std: (f32, f32, f32),
}

#[pymethods]
impl LoaderConfig {
    #[new]
    #[pyo3(signature = (*, batch_size=32, num_decode_threads=0, buffer_size=10_000, prefetch_depth=64, width=224, height=224, normalize=false, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))]
    fn new(
        batch_size: usize,
        num_decode_threads: usize,
        buffer_size: usize,
        prefetch_depth: usize,
        width: u32,
        height: u32,
        normalize: bool,
        mean: (f32, f32, f32),
        std: (f32, f32, f32),
    ) -> Self {
        LoaderConfig {
            batch_size,
            num_decode_threads,
            buffer_size,
            prefetch_depth,
            width,
            height,
            normalize,
            mean,
            std,
        }
    }
}

impl LoaderConfig {
    fn to_engine_config(&self) -> EngineConfig {
        EngineConfig {
            num_decode_threads: self.num_decode_threads,
            buffer_size: self.buffer_size,
            prefetch_depth: self.prefetch_depth,
            batch_size: self.batch_size,
            filter: FilterSpec {
                width: self.width,
                height: self.height,
                normalize: self.normalize,
                mean: [self.mean.0, self.mean.1, self.mean.2],
                std: [self.std.0, self.std.1, self.std.2],
            },
        }
    }
}

/// A batch of training samples returned by the Loader.
#[pyclass]
pub struct Batch {
    inner: veldt_core::Batch,
}

#[pymethods]
impl Batch {
    /// Get the video frames as a numpy array.
    /// Shape: [B, C, H, W] (u8) or [B, C, H, W] (f32 if normalized).
    #[getter]
    fn frames<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        match &self.inner.frames {
            FrameBuffer::U8 { data, shape } => {
                let arr = PyArray1::from_slice(py, data);
                let reshaped = arr
                    .reshape_with_order(shape.to_vec(), numpy::npyffi::NPY_ORDER::NPY_CORDER)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(reshaped.into_any().unbind())
            }
            FrameBuffer::F32 { data, shape } => {
                let arr = PyArray1::from_slice(py, data);
                let reshaped = arr
                    .reshape_with_order(shape.to_vec(), numpy::npyffi::NPY_ORDER::NPY_CORDER)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(reshaped.into_any().unbind())
            }
        }
    }

    /// Number of samples in this batch.
    #[getter]
    fn batch_size(&self) -> usize {
        self.inner.batch_size
    }

    /// Get a tabular column by name as a numpy array.
    fn tabular<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Option<PyObject>> {
        match self.inner.tabular.get(key) {
            Some(col) => {
                let arr = PyArray1::from_slice(py, &col.data);
                let reshaped = arr
                    .reshape_with_order(col.shape.clone(), numpy::npyffi::NPY_ORDER::NPY_CORDER)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(Some(reshaped.into_any().unbind()))
            }
            None => Ok(None),
        }
    }

    /// List available tabular column names.
    fn keys(&self) -> Vec<String> {
        self.inner.tabular.keys().cloned().collect()
    }
}

/// Seek-optimized video data loader for ML training.
#[pyclass]
pub struct Loader {
    engine: Engine,
    dataset_len: usize,
    num_frames_per_clip: u32,
    stride: u32,
}

#[pymethods]
impl Loader {
    /// Create a new Loader for a LeRobot v3 dataset.
    #[new]
    #[pyo3(signature = (dataset_path, config=None))]
    fn new(dataset_path: &str, config: Option<LoaderConfig>) -> PyResult<Self> {
        let config = config.unwrap_or_else(|| LoaderConfig::new(
            32, 0, 10_000, 64, 224, 224, false,
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ));

        let root = PathBuf::from(dataset_path);
        let reader = LeRobotReader::open(&root)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let dataset_len = reader.len();
        let engine_config = config.to_engine_config();

        let engine = Engine::new(Box::new(reader), engine_config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Loader {
            engine,
            dataset_len,
            num_frames_per_clip: 16,
            stride: 1,
        })
    }

    /// Set up a new epoch with shuffled sample indices.
    ///
    /// Uses the given seed to generate a deterministic shuffle order.
    #[pyo3(signature = (epoch, seed=42))]
    fn set_epoch(&mut self, epoch: i64, seed: u64) -> PyResult<()> {
        let mut indices: Vec<usize> = (0..self.dataset_len).collect();

        // Simple deterministic shuffle using seed + epoch
        let combined_seed = seed.wrapping_add(epoch as u64);
        let mut rng_state = combined_seed;
        for i in (1..indices.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng_state >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }

        let h = self.engine.metadata().fps as u32; // placeholder
        let _ = h;

        let requests: Vec<(usize, ClipSpec)> = indices
            .into_iter()
            .map(|idx| {
                (
                    idx,
                    ClipSpec {
                        start_frame: 0,
                        num_frames: self.num_frames_per_clip,
                        stride: self.stride,
                        resolution: (224, 224), // TODO: from config
                    },
                )
            })
            .collect();

        self.engine
            .set_epoch(requests)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<Batch>> {
        match self.engine.next_batch() {
            Ok(Some(batch)) => Ok(Some(Batch { inner: batch })),
            Ok(None) => Ok(None),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    fn __len__(&self) -> usize {
        self.engine.num_batches()
    }

    /// Return cache statistics as a dict.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.engine.stats();
        let dict = PyDict::new(py);
        dict.set_item("cache_hits", s.cache_hits)?;
        dict.set_item("cache_misses", s.cache_misses)?;
        dict.set_item("evictions", s.evictions)?;
        dict.set_item("refetches", s.refetches)?;
        dict.set_item("current_frames", s.current_frames)?;
        dict.set_item("buffer_size", s.buffer_size)?;
        Ok(dict)
    }

    /// Return dataset metadata as a dict.
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let m = self.engine.metadata();
        let dict = PyDict::new(py);
        dict.set_item("fps", m.fps)?;
        dict.set_item("num_episodes", m.num_episodes)?;
        dict.set_item("num_frames", m.num_frames)?;
        dict.set_item("camera_keys", m.camera_keys.clone())?;
        dict.set_item("action_dim", m.action_dim)?;
        dict.set_item("state_dim", m.state_dim)?;
        Ok(dict)
    }
}

/// The veldt Python module.
#[pymodule]
fn _veldt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LoaderConfig>()?;
    m.add_class::<Loader>()?;
    m.add_class::<Batch>()?;
    Ok(())
}
