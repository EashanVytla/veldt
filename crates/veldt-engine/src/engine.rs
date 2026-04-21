use std::collections::{HashMap, VecDeque};

use veldt_cache::BeladyCache;
use veldt_core::*;
use veldt_decode::{DecodePool, DecodeRequest, FilterSpec};
use veldt_scheduler::{BeladyScheduler, PrefetchBuckets};

/// Configuration for the Engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Number of decode threads (0 = auto-detect CPU cores).
    pub num_decode_threads: usize,
    /// Maximum decoded frames to hold in cache.
    pub buffer_size: usize,
    /// How many batches ahead to prefetch.
    pub prefetch_depth: usize,
    /// Batch size for training.
    pub batch_size: usize,
    /// Filter spec for decode (resize, normalize).
    pub filter: FilterSpec,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            num_decode_threads: 0,
            buffer_size: 10_000,
            prefetch_depth: 64,
            batch_size: 32,
            filter: FilterSpec::default(),
        }
    }
}

/// The main pipeline orchestrator.
///
/// Wires together: reader → plan → scheduler → prefetch → decode → cache → batch.
///
/// Usage:
/// ```ignore
/// let engine = Engine::new(reader, config)?;
/// engine.set_epoch(shuffle_order, epoch)?;
/// while let Some(batch) = engine.next_batch()? {
///     // train on batch
/// }
/// ```
pub struct Engine {
    reader: Box<dyn DatasetReader>,
    config: EngineConfig,
    decode_pool: DecodePool,
    cache: BeladyCache,
    scheduler: BeladyScheduler,

    // Per-epoch state (set by set_epoch)
    epoch_state: Option<EpochState>,
}

struct EpochState {
    requests: Vec<(usize, ClipSpec)>,
    plan: FetchPlan,
    buckets: PrefetchBuckets,
    num_batches: usize,
    current_batch: usize,
    /// Map from FetchUnitId to FetchUnit for lookup during prefetch.
    unit_map: HashMap<FetchUnitId, FetchUnit>,
    /// Remaining future-use queues per unit (mutable copy from plan).
    unit_queues: HashMap<FetchUnitId, VecDeque<usize>>,
}

impl Engine {
    /// Create a new engine with the given reader and configuration.
    pub fn new(reader: Box<dyn DatasetReader>, config: EngineConfig) -> Result<Self> {
        let decode_pool = DecodePool::new(config.num_decode_threads)?;
        let cache = BeladyCache::new(config.buffer_size);
        let scheduler = BeladyScheduler::new(config.prefetch_depth);

        Ok(Engine {
            reader,
            config,
            decode_pool,
            cache,
            scheduler,
            epoch_state: None,
        })
    }

    /// Set up a new epoch with the given shuffle order.
    ///
    /// `requests` is the full shuffle order: `(sample_index, ClipSpec)` for every
    /// sample in the epoch. The engine builds the fetch plan, future-use queues,
    /// and prefetch buckets.
    pub fn set_epoch(&mut self, requests: Vec<(usize, ClipSpec)>) -> Result<()> {
        let batch_size = self.config.batch_size;
        let num_batches = (requests.len() + batch_size - 1) / batch_size;

        // Build the fetch plan via the reader
        let plan = self.reader.plan_fetch(&requests, batch_size)?;

        // Build prefetch buckets from the plan
        let buckets = self.scheduler.build_prefetch_buckets(&plan, num_batches);

        // Build unit lookup map
        let unit_map: HashMap<FetchUnitId, FetchUnit> =
            plan.units.iter().map(|u| (u.id, u.clone())).collect();

        // Copy future-use queues (we'll mutate these as batches are consumed)
        let unit_queues = plan.unit_batches.clone();

        // Reset cache for new epoch
        self.cache = BeladyCache::new(self.config.buffer_size);

        self.epoch_state = Some(EpochState {
            requests,
            plan,
            buckets,
            num_batches,
            current_batch: 0,
            unit_map,
            unit_queues,
        });

        // Prefetch initial window
        self.prefetch_window()?;

        Ok(())
    }

    /// Get the next batch, or None if the epoch is complete.
    pub fn next_batch(&mut self) -> Result<Option<Batch>> {
        match &self.epoch_state {
            Some(s) if s.current_batch < s.num_batches => {}
            _ => return Ok(None),
        };

        let state = self.epoch_state.as_ref().unwrap();
        let batch_idx = state.current_batch;
        let batch_size = self.config.batch_size;
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(state.requests.len());
        let batch_requests: Vec<(usize, ClipSpec)> = state.requests[start..end].to_vec();

        // Collect the units needed for this batch
        let batch_unit_ids: Vec<Vec<FetchUnitId>> = (start..end)
            .map(|i| state.plan.request_to_units[i].clone())
            .collect();

        // Ensure all needed units are in cache (decode if missing)
        let mut all_needed_units: Vec<FetchUnitId> = batch_unit_ids
            .iter()
            .flat_map(|ids| ids.iter().copied())
            .collect();
        all_needed_units.sort_unstable();
        all_needed_units.dedup();

        self.ensure_cached(&all_needed_units)?;

        // Assemble batch frames
        // For each sample in the batch, we need to extract the right frames
        // from the cached decoded data
        let filter = &self.config.filter;
        let h = filter.height as usize;
        let w = filter.width as usize;
        let actual_batch_size = end - start;

        let mut batch_tabular: HashMap<String, Vec<f32>> = HashMap::new();
        let mut all_frame_data: Vec<u8> = Vec::new();
        let mut max_frames = 0u32;

        for (_i, (sample_idx, spec)) in batch_requests.iter().enumerate() {
            // Read tabular data via the reader
            let sample = self.reader.get(*sample_idx, spec)?;

            // Merge tabular data
            for (key, col) in &sample.tabular {
                batch_tabular
                    .entry(key.clone())
                    .or_default()
                    .extend_from_slice(&col.data);
            }

            max_frames = max_frames.max(spec.num_frames);

            // For video frames: in a full implementation, we'd extract the specific
            // frames from the cached keyframe groups. For now, use the reader's frames.
            match &sample.frames {
                FrameBuffer::U8 { data, .. } => all_frame_data.extend_from_slice(data),
                FrameBuffer::F32 { data, .. } => {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            data.as_ptr() as *const u8,
                            data.len() * std::mem::size_of::<f32>(),
                        )
                    };
                    all_frame_data.extend_from_slice(bytes);
                }
            }
        }

        // Build batch tabular columns
        let mut tabular_result = HashMap::new();
        for (key, data) in batch_tabular {
            let per_sample = data.len() / actual_batch_size;
            tabular_result.insert(
                key,
                TabularColumn {
                    data,
                    shape: vec![actual_batch_size, per_sample],
                },
            );
        }

        // Shape: [B*T, C, H, W] where T = max_frames per sample
        let total_frames = all_frame_data.len() / (3 * h * w);
        let frames = if filter.normalize {
            let num_floats = all_frame_data.len() / std::mem::size_of::<f32>();
            let float_data: Vec<f32> = unsafe {
                std::slice::from_raw_parts(
                    all_frame_data.as_ptr() as *const f32,
                    num_floats,
                )
            }
            .to_vec();
            let total_f32_frames = float_data.len() / (3 * h * w);
            FrameBuffer::F32 {
                data: float_data,
                shape: [total_f32_frames, 3, h, w],
            }
        } else {
            FrameBuffer::U8 {
                data: all_frame_data,
                shape: [total_frames, 3, h, w],
            }
        };

        // Consume batch: update cache, evict empty queues
        let state = self.epoch_state.as_mut().unwrap();
        let evicted = self.cache.consume_batch(batch_idx, &all_needed_units);

        // Re-insert evicted units with remaining future uses into prefetch buckets
        for ev in evicted {
            if let Some(next_batch) = ev.next_needed_batch {
                state.buckets.reinsert(ev.unit_id, next_batch);
            }
        }

        state.current_batch += 1;

        // Trigger prefetch for new window
        self.prefetch_window()?;

        Ok(Some(Batch {
            frames,
            tabular: tabular_result,
            batch_size: actual_batch_size,
        }))
    }

    /// Ensure all units are in the cache, decoding any that are missing.
    fn ensure_cached(&mut self, unit_ids: &[FetchUnitId]) -> Result<()> {
        let state = self.epoch_state.as_ref().ok_or_else(|| {
            VeldtError::Dataset("no epoch set".to_string())
        })?;

        // Collect owned data to avoid borrow conflicts
        let mut to_decode: Vec<(FetchUnitId, DecodeRequest, VecDeque<usize>)> = Vec::new();
        for &uid in unit_ids {
            if !self.cache.contains(&uid) {
                self.cache.record_miss();
                if let Some(unit) = state.unit_map.get(&uid) {
                    let path = match &unit.source {
                        FetchSource::LocalFile { path, .. } => path.clone(),
                    };
                    let req = DecodeRequest {
                        path,
                        start_frame: 0,
                        num_frames: unit.num_decoded_frames,
                        stride: 1,
                        filter: self.config.filter.clone(),
                    };
                    let queue = state.unit_queues.get(&uid).cloned().unwrap_or_default();
                    to_decode.push((uid, req, queue));
                }
            }
        }

        if to_decode.is_empty() {
            return Ok(());
        }

        // Build decode requests (owned, no borrows on self)
        let decode_requests: Vec<DecodeRequest> =
            to_decode.iter().map(|(_, req, _)| req.clone()).collect();

        // Decode in parallel
        let results = self.decode_pool.decode_batch(decode_requests);

        // Insert decoded frames into cache
        for (i, result) in results.into_iter().enumerate() {
            let (uid, _, queue) = &to_decode[i];
            match result {
                Ok(clip) => {
                    let frames = DecodedFrames {
                        data: clip.data,
                        shape: clip.shape,
                        unit_id: *uid,
                    };
                    let evicted = self.cache.insert(frames, queue.clone());

                    // Handle evicted units
                    if let Some(state) = self.epoch_state.as_mut() {
                        for ev in evicted {
                            if let Some(next_batch) = ev.next_needed_batch {
                                state.buckets.reinsert(ev.unit_id, next_batch);
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("decode failed for unit {:?}: {}", uid, e);
                }
            }
        }

        Ok(())
    }

    /// Prefetch units in the current lookahead window.
    fn prefetch_window(&mut self) -> Result<()> {
        let state = self.epoch_state.as_ref().ok_or_else(|| {
            VeldtError::Dataset("no epoch set".to_string())
        })?;

        let current = state.current_batch;
        let to_prefetch = self.scheduler.units_to_prefetch(&state.buckets, current);

        if !to_prefetch.is_empty() {
            // Filter to units not already cached
            let needed: Vec<FetchUnitId> = to_prefetch
                .iter()
                .filter(|uid| !self.cache.contains(uid))
                .copied()
                .collect();

            if !needed.is_empty() {
                self.ensure_cached(&needed)?;
            }
        }

        Ok(())
    }

    /// Get current cache statistics.
    pub fn stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Get the underlying reader's metadata.
    pub fn metadata(&self) -> &DatasetMeta {
        self.reader.metadata()
    }

    /// Number of batches in the current epoch.
    pub fn num_batches(&self) -> usize {
        self.epoch_state
            .as_ref()
            .map(|s| s.num_batches)
            .unwrap_or(0)
    }

    /// Current batch index (0-based).
    pub fn current_batch(&self) -> usize {
        self.epoch_state
            .as_ref()
            .map(|s| s.current_batch)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::path::PathBuf;

    /// A minimal mock reader for testing the engine without real files.
    struct MockReader {
        meta: DatasetMeta,
        num_samples: usize,
    }

    impl MockReader {
        fn new(num_samples: usize) -> Self {
            MockReader {
                meta: DatasetMeta {
                    fps: 30.0,
                    num_episodes: 1,
                    num_frames: num_samples,
                    camera_keys: vec!["cam".to_string()],
                    action_dim: 4,
                    state_dim: 4,
                },
                num_samples,
            }
        }
    }

    impl DatasetReader for MockReader {
        fn len(&self) -> usize {
            self.num_samples
        }

        fn get(&self, index: usize, spec: &ClipSpec) -> Result<Sample> {
            if index >= self.num_samples {
                return Err(VeldtError::IndexOutOfRange {
                    index,
                    len: self.num_samples,
                });
            }
            let nf = spec.num_frames as usize;
            let (h, w) = spec.resolution;
            let h = h as usize;
            let w = w as usize;

            let mut tabular = HashMap::new();
            tabular.insert(
                "action".to_string(),
                TabularColumn {
                    data: vec![0.0f32; nf * 4],
                    shape: vec![nf, 4],
                },
            );

            Ok(Sample {
                frames: FrameBuffer::U8 {
                    data: vec![128u8; nf * 3 * h * w],
                    shape: [nf, 3, h, w],
                },
                tabular,
                episode_index: 0,
                frame_index: index as u64,
            })
        }

        fn plan_fetch(
            &self,
            requests: &[(usize, ClipSpec)],
            batch_size: usize,
        ) -> Result<FetchPlan> {
            // Simple mock: each sample maps to one unit (unit_id = sample_index / 30)
            let mut all_units: HashMap<FetchUnitId, FetchUnit> = HashMap::new();
            let mut request_to_units = Vec::with_capacity(requests.len());
            let mut unit_batch_sets: HashMap<FetchUnitId, Vec<usize>> = HashMap::new();

            for (req_idx, (sample_idx, _spec)) in requests.iter().enumerate() {
                let batch_idx = req_idx / batch_size;
                let unit_id = FetchUnitId((*sample_idx / 30) as u64);

                request_to_units.push(vec![unit_id]);
                unit_batch_sets.entry(unit_id).or_default().push(batch_idx);

                all_units.entry(unit_id).or_insert_with(|| FetchUnit {
                    id: unit_id,
                    source: FetchSource::LocalFile {
                        path: PathBuf::from("mock.mp4"),
                        byte_offset: 0,
                        byte_len: 1000,
                    },
                    estimated_bytes: 1000,
                    num_decoded_frames: 30,
                });
            }

            let mut unit_batches: HashMap<FetchUnitId, VecDeque<usize>> = HashMap::new();
            for (uid, mut batches) in unit_batch_sets {
                batches.sort_unstable();
                batches.dedup();
                unit_batches.insert(uid, VecDeque::from(batches));
            }

            Ok(FetchPlan {
                units: all_units.into_values().collect(),
                request_to_units,
                unit_batches,
            })
        }

        fn metadata(&self) -> &DatasetMeta {
            &self.meta
        }
    }

    #[test]
    fn test_engine_creation() {
        let reader = Box::new(MockReader::new(90));
        let config = EngineConfig {
            num_decode_threads: 1,
            buffer_size: 1000,
            prefetch_depth: 2,
            batch_size: 10,
            filter: FilterSpec {
                width: 16,
                height: 16,
                normalize: false,
                ..Default::default()
            },
        };
        let engine = Engine::new(reader, config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_engine_set_epoch() {
        let reader = Box::new(MockReader::new(90));
        let config = EngineConfig {
            num_decode_threads: 1,
            buffer_size: 1000,
            prefetch_depth: 2,
            batch_size: 10,
            filter: FilterSpec {
                width: 16,
                height: 16,
                normalize: false,
                ..Default::default()
            },
        };
        let mut engine = Engine::new(reader, config).unwrap();

        let spec = ClipSpec {
            start_frame: 0,
            num_frames: 5,
            stride: 1,
            resolution: (16, 16),
        };
        let requests: Vec<(usize, ClipSpec)> = (0..30).map(|i| (i, spec)).collect();

        engine.set_epoch(requests).unwrap();
        assert_eq!(engine.num_batches(), 3); // 30 / 10 = 3
        assert_eq!(engine.current_batch(), 0);
    }

    #[test]
    fn test_engine_iterate_batches() {
        let reader = Box::new(MockReader::new(90));
        let config = EngineConfig {
            num_decode_threads: 1,
            buffer_size: 1000,
            prefetch_depth: 2,
            batch_size: 10,
            filter: FilterSpec {
                width: 16,
                height: 16,
                normalize: false,
                ..Default::default()
            },
        };
        let mut engine = Engine::new(reader, config).unwrap();

        let spec = ClipSpec {
            start_frame: 0,
            num_frames: 5,
            stride: 1,
            resolution: (16, 16),
        };
        let requests: Vec<(usize, ClipSpec)> = (0..30).map(|i| (i, spec)).collect();

        engine.set_epoch(requests).unwrap();

        let mut batch_count = 0;
        while let Some(batch) = engine.next_batch().unwrap() {
            assert_eq!(batch.batch_size, 10);
            assert!(batch.tabular.contains_key("action"));
            batch_count += 1;
        }
        assert_eq!(batch_count, 3);

        // After exhausting, next_batch returns None
        assert!(engine.next_batch().unwrap().is_none());
    }

    #[test]
    fn test_engine_partial_last_batch() {
        let reader = Box::new(MockReader::new(90));
        let config = EngineConfig {
            num_decode_threads: 1,
            buffer_size: 1000,
            prefetch_depth: 2,
            batch_size: 8,
            filter: FilterSpec {
                width: 16,
                height: 16,
                normalize: false,
                ..Default::default()
            },
        };
        let mut engine = Engine::new(reader, config).unwrap();

        let spec = ClipSpec {
            start_frame: 0,
            num_frames: 3,
            stride: 1,
            resolution: (16, 16),
        };
        // 25 samples, batch_size=8 → batches of 8, 8, 8, 1
        let requests: Vec<(usize, ClipSpec)> = (0..25).map(|i| (i, spec)).collect();

        engine.set_epoch(requests).unwrap();
        assert_eq!(engine.num_batches(), 4); // ceil(25/8) = 4

        let mut sizes = Vec::new();
        while let Some(batch) = engine.next_batch().unwrap() {
            sizes.push(batch.batch_size);
        }
        assert_eq!(sizes, vec![8, 8, 8, 1]);
    }

    #[test]
    fn test_engine_stats() {
        let reader = Box::new(MockReader::new(90));
        let config = EngineConfig {
            num_decode_threads: 1,
            buffer_size: 1000,
            prefetch_depth: 2,
            batch_size: 10,
            filter: FilterSpec {
                width: 16,
                height: 16,
                normalize: false,
                ..Default::default()
            },
        };
        let mut engine = Engine::new(reader, config).unwrap();

        let spec = ClipSpec {
            start_frame: 0,
            num_frames: 5,
            stride: 1,
            resolution: (16, 16),
        };
        let requests: Vec<(usize, ClipSpec)> = (0..20).map(|i| (i, spec)).collect();

        engine.set_epoch(requests).unwrap();

        // Iterate all batches
        while engine.next_batch().unwrap().is_some() {}

        let stats = engine.stats();
        assert_eq!(stats.buffer_size, 1000);
    }

    #[test]
    fn test_engine_no_epoch() {
        let reader = Box::new(MockReader::new(90));
        let config = EngineConfig::default();
        let mut engine = Engine::new(reader, config).unwrap();

        // No epoch set, next_batch returns None
        assert!(engine.next_batch().unwrap().is_none());
    }
}
