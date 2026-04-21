use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

use veldt_core::*;
use veldt_index::VkfIndex;

use crate::metadata::{EpisodeInfo, InfoJson};
use crate::tabular;

/// LeRobot v3 dataset reader.
///
/// Reads datasets in the LeRobot v3 format (Parquet + MP4 with concatenated episodes).
/// Requires VKF sidecar indices to be pre-built for all video files.
pub struct LeRobotReader {
    root: PathBuf,
    info: InfoJson,
    meta: DatasetMeta,
    episodes: Vec<EpisodeInfo>,
    /// Per video file (keyed by relative path): VKF index
    vkf_indices: HashMap<PathBuf, VkfIndex>,
    /// For each global sample index: (episode_index, frame_within_episode)
    sample_map: Vec<(usize, u64)>,
}

impl LeRobotReader {
    /// Open a LeRobot v3 dataset. VKF sidecars must already exist for all video files.
    pub fn open(root: &Path) -> Result<Self> {
        let info = crate::metadata::load_info(root)?;
        let meta = crate::metadata::build_dataset_meta(&info);
        let episodes = crate::metadata::load_episodes(root, &info)?;

        // Build sample map: global index -> (episode, frame_within_episode)
        let total_frames = info.total_frames;
        let mut sample_map = Vec::with_capacity(total_frames);
        for ep in &episodes {
            for frame in 0..ep.length {
                sample_map.push((ep.episode_index as usize, frame));
            }
        }

        // Load VKF indices for all video files
        let video_keys: Vec<String> = info
            .features
            .iter()
            .filter(|(_, f)| f.dtype == "video")
            .map(|(k, _)| k.clone())
            .collect();

        let mut vkf_indices = HashMap::new();
        for ep in &episodes {
            for vk in &video_keys {
                if let Some(vi) = ep.video_info.get(vk) {
                    let video_rel = video_relative_path(&info, vk, vi.chunk_index, vi.file_index);
                    if vkf_indices.contains_key(&video_rel) {
                        continue;
                    }
                    let mp4_path = root.join(&video_rel);
                    let vkf_path = VkfIndex::sidecar_path(&mp4_path);
                    if !vkf_path.exists() {
                        return Err(VeldtError::MissingSidecar(mp4_path));
                    }
                    let idx = VkfIndex::read_from(&vkf_path)?;
                    vkf_indices.insert(video_rel, idx);
                }
            }
        }

        Ok(LeRobotReader {
            root: root.to_path_buf(),
            info,
            meta,
            episodes,
            vkf_indices,
            sample_map,
        })
    }

    /// Resolve a global sample index to its episode info.
    fn episode_for_sample(&self, index: usize) -> Result<&EpisodeInfo> {
        if index >= self.sample_map.len() {
            return Err(VeldtError::IndexOutOfRange {
                index,
                len: self.sample_map.len(),
            });
        }
        let (ep_idx, _) = self.sample_map[index];
        self.episodes
            .iter()
            .find(|e| e.episode_index == ep_idx as u64)
            .ok_or_else(|| VeldtError::Dataset(format!("episode {} not found", ep_idx)))
    }

    /// For a sample + clip spec, determine which video frame indices (in the MP4 file)
    /// need to be decoded, and which keyframe groups contain them.
    fn resolve_video_frames(
        &self,
        ep: &EpisodeInfo,
        video_key: &str,
        spec: &ClipSpec,
    ) -> Result<ResolvedVideoFrames> {
        let vi = ep.video_info.get(video_key).ok_or_else(|| {
            VeldtError::Dataset(format!("no video info for key {}", video_key))
        })?;

        let video_rel = video_relative_path(&self.info, video_key, vi.chunk_index, vi.file_index);
        let vkf = self.vkf_indices.get(&video_rel).ok_or_else(|| {
            VeldtError::MissingSidecar(self.root.join(&video_rel))
        })?;

        let fps = self.info.fps as f64;

        // The episode starts at from_timestamp in the MP4.
        // Frame N within the episode is at from_timestamp + N/fps in the MP4.
        // Convert to MP4 frame index using the VKF PTS table.
        let start_frame_in_ep = spec.start_frame;
        let stride = spec.stride.max(1) as u64;

        let mut mp4_frame_indices = Vec::new();
        for i in 0..spec.num_frames as u64 {
            let frame_in_ep = start_frame_in_ep + i * stride;
            if frame_in_ep >= ep.length {
                break;
            }
            // Timestamp in the MP4 file
            let mp4_time = vi.from_timestamp + frame_in_ep as f64 / fps;
            // Find the closest frame by PTS
            let mp4_frame = pts_to_frame_index(vkf, mp4_time);
            mp4_frame_indices.push(mp4_frame);
        }

        // Find all keyframe groups needed
        let mut keyframe_groups: Vec<usize> = mp4_frame_indices
            .iter()
            .filter_map(|&f| vkf.keyframe_group_for_frame(f))
            .collect();
        keyframe_groups.sort_unstable();
        keyframe_groups.dedup();

        Ok(ResolvedVideoFrames {
            video_rel,
            mp4_frame_indices,
            keyframe_groups,
        })
    }

    /// Build a unique FetchUnitId from video file path + keyframe group index.
    fn fetch_unit_id(video_rel: &Path, kf_group: usize) -> FetchUnitId {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        video_rel.hash(&mut hasher);
        kf_group.hash(&mut hasher);
        FetchUnitId(hasher.finish())
    }
}

#[allow(dead_code)]
struct ResolvedVideoFrames {
    video_rel: PathBuf,
    mp4_frame_indices: Vec<u32>,
    keyframe_groups: Vec<usize>,
}

/// Convert a timestamp (seconds) to the nearest MP4 frame index using the VKF PTS table.
fn pts_to_frame_index(vkf: &VkfIndex, time_secs: f64) -> u32 {
    let target_pts = (time_secs * vkf.timescale as f64) as u64;
    // Binary search for the closest PTS
    match vkf.frame_pts.binary_search(&target_pts) {
        Ok(idx) => idx as u32,
        Err(idx) => {
            if idx == 0 {
                0
            } else if idx >= vkf.frame_pts.len() {
                (vkf.frame_pts.len() - 1) as u32
            } else {
                // Pick the closer of idx-1 and idx
                let diff_before = target_pts - vkf.frame_pts[idx - 1];
                let diff_after = vkf.frame_pts[idx] - target_pts;
                if diff_before <= diff_after {
                    (idx - 1) as u32
                } else {
                    idx as u32
                }
            }
        }
    }
}

/// Build the relative video path from info.json template.
fn video_relative_path(
    info: &InfoJson,
    video_key: &str,
    chunk_index: u64,
    file_index: u64,
) -> PathBuf {
    // Template: "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    if let Some(ref template) = info.video_path {
        let path_str = template
            .replace("{video_key}", video_key)
            .replace(
                &format!("{{chunk_index:03d}}"),
                &format!("{:03}", chunk_index),
            )
            .replace(
                &format!("{{file_index:03d}}"),
                &format!("{:03}", file_index),
            );
        PathBuf::from(path_str)
    } else {
        PathBuf::from(format!(
            "videos/{}/chunk-{:03}/file-{:03}.mp4",
            video_key, chunk_index, file_index
        ))
    }
}

/// Build the relative data parquet path from info.json template.
fn data_relative_path(info: &InfoJson, chunk_index: u64, file_index: u64) -> PathBuf {
    let path_str = info
        .data_path
        .replace(
            &format!("{{chunk_index:03d}}"),
            &format!("{:03}", chunk_index),
        )
        .replace(
            &format!("{{file_index:03d}}"),
            &format!("{:03}", file_index),
        );
    PathBuf::from(path_str)
}

impl DatasetReader for LeRobotReader {
    fn len(&self) -> usize {
        self.sample_map.len()
    }

    fn get(&self, index: usize, spec: &ClipSpec) -> Result<Sample> {
        if index >= self.sample_map.len() {
            return Err(VeldtError::IndexOutOfRange {
                index,
                len: self.sample_map.len(),
            });
        }

        let (ep_idx, frame_in_ep) = self.sample_map[index];
        let ep = self
            .episodes
            .iter()
            .find(|e| e.episode_index == ep_idx as u64)
            .ok_or_else(|| VeldtError::Dataset(format!("episode {} not found", ep_idx)))?;

        // Read tabular data from the data parquet file
        let data_rel = data_relative_path(&self.info, ep.data_chunk_index, ep.data_file_index);
        let data_path = self.root.join(&data_rel);

        let stride = spec.stride.max(1) as u64;
        // Build global indices for the frames we want
        let mut global_indices = Vec::new();
        for i in 0..spec.num_frames as u64 {
            let frame = frame_in_ep + i * stride;
            if frame >= ep.length {
                break;
            }
            global_indices.push(ep.dataset_from_index + frame);
        }

        let tabular = tabular::read_tabular_for_indices(
            &data_path,
            &global_indices,
            &self.info.features,
        )?;

        // For now, return a placeholder FrameBuffer. The actual video decode
        // is done by veldt-decode via the engine, not directly by the reader.
        // The reader's job is plan_fetch (mapping to FetchUnits) and tabular data.
        let num_frames = global_indices.len();
        let (h, w) = spec.resolution;
        let frames = FrameBuffer::U8 {
            data: vec![0u8; num_frames * 3 * h as usize * w as usize],
            shape: [num_frames, 3, h as usize, w as usize],
        };

        Ok(Sample {
            frames,
            tabular,
            episode_index: ep_idx as u32,
            frame_index: frame_in_ep,
        })
    }

    fn plan_fetch(
        &self,
        requests: &[(usize, ClipSpec)],
        batch_size: usize,
    ) -> Result<FetchPlan> {
        // Use the first video key for planning (multi-camera would need multiple fetch units per sample)
        let video_key = self
            .meta
            .camera_keys
            .first()
            .ok_or_else(|| VeldtError::Dataset("no video keys in dataset".into()))?;

        let mut all_units: HashMap<FetchUnitId, FetchUnit> = HashMap::new();
        let mut request_to_units: Vec<Vec<FetchUnitId>> = Vec::with_capacity(requests.len());
        let mut unit_batch_sets: HashMap<FetchUnitId, Vec<usize>> = HashMap::new();

        for (req_idx, (sample_idx, spec)) in requests.iter().enumerate() {
            let batch_idx = req_idx / batch_size;
            let ep = self.episode_for_sample(*sample_idx)?;
            let resolved = self.resolve_video_frames(ep, video_key, spec)?;

            let vkf = self.vkf_indices.get(&resolved.video_rel).ok_or_else(|| {
                VeldtError::MissingSidecar(self.root.join(&resolved.video_rel))
            })?;

            let mut unit_ids = Vec::new();
            for &kf_group in &resolved.keyframe_groups {
                let unit_id = Self::fetch_unit_id(&resolved.video_rel, kf_group);
                unit_ids.push(unit_id);

                // Track which batches need this unit
                unit_batch_sets
                    .entry(unit_id)
                    .or_default()
                    .push(batch_idx);

                // Build FetchUnit if not already seen
                if !all_units.contains_key(&unit_id) {
                    let (byte_offset, byte_len) =
                        vkf.keyframe_byte_range(kf_group).ok_or_else(|| {
                            VeldtError::Dataset(format!(
                                "keyframe group {} out of range",
                                kf_group
                            ))
                        })?;

                    let mp4_path = self.root.join(&resolved.video_rel);
                    let num_decoded_frames = vkf.keyframes[kf_group].num_frames;

                    all_units.insert(
                        unit_id,
                        FetchUnit {
                            id: unit_id,
                            source: FetchSource::LocalFile {
                                path: mp4_path,
                                byte_offset,
                                byte_len,
                            },
                            estimated_bytes: byte_len,
                            num_decoded_frames,
                        },
                    );
                }
            }
            request_to_units.push(unit_ids);
        }

        // Build future-use queues: sort and dedup batch indices for each unit
        let mut unit_batches: HashMap<FetchUnitId, VecDeque<usize>> = HashMap::new();
        for (unit_id, mut batches) in unit_batch_sets {
            batches.sort_unstable();
            batches.dedup();
            unit_batches.insert(unit_id, VecDeque::from(batches));
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

#[cfg(test)]
mod tests {
    use super::*;
    use veldt_index::parse_mp4;

    fn fixture_root() -> PathBuf {
        let manifest = env!("CARGO_MANIFEST_DIR");
        PathBuf::from(manifest)
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("tests/fixtures/mini_lerobot")
    }

    /// Ensure VKF sidecar exists for the test fixture video.
    fn ensure_vkf(root: &Path) {
        let mp4_path = root.join("videos/observation.images.top/chunk-000/file-000.mp4");
        let vkf_path = VkfIndex::sidecar_path(&mp4_path);
        if !vkf_path.exists() {
            let index = parse_mp4(&mp4_path).expect("parse test MP4");
            index.write_to(&vkf_path).expect("write VKF");
        }
    }

    #[test]
    fn test_load_metadata() {
        let root = fixture_root();
        if !root.exists() {
            eprintln!("skipping: fixture not found at {:?}", root);
            return;
        }

        let info = crate::metadata::load_info(&root).unwrap();
        assert_eq!(info.total_episodes, 2);
        assert_eq!(info.total_frames, 90);
        assert_eq!(info.fps, 30);
        assert!(info.features.contains_key("observation.images.top"));
        assert!(info.features.contains_key("action"));
        assert!(info.features.contains_key("observation.state"));

        let meta = crate::metadata::build_dataset_meta(&info);
        assert_eq!(meta.fps, 30.0);
        assert_eq!(meta.num_episodes, 2);
        assert_eq!(meta.num_frames, 90);
        assert_eq!(meta.camera_keys, vec!["observation.images.top"]);
        assert_eq!(meta.action_dim, 4);
        assert_eq!(meta.state_dim, 4);
    }

    #[test]
    fn test_load_episodes() {
        let root = fixture_root();
        if !root.exists() {
            return;
        }

        let info = crate::metadata::load_info(&root).unwrap();
        let episodes = crate::metadata::load_episodes(&root, &info).unwrap();

        assert_eq!(episodes.len(), 2);

        assert_eq!(episodes[0].episode_index, 0);
        assert_eq!(episodes[0].dataset_from_index, 0);
        assert_eq!(episodes[0].dataset_to_index, 45);
        assert_eq!(episodes[0].length, 45);

        assert_eq!(episodes[1].episode_index, 1);
        assert_eq!(episodes[1].dataset_from_index, 45);
        assert_eq!(episodes[1].dataset_to_index, 90);
        assert_eq!(episodes[1].length, 45);

        // Check video info
        let vi = episodes[0]
            .video_info
            .get("observation.images.top")
            .unwrap();
        assert_eq!(vi.chunk_index, 0);
        assert_eq!(vi.file_index, 0);
        assert!((vi.from_timestamp - 0.0).abs() < 0.001);
        assert!((vi.to_timestamp - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_open_reader() {
        let root = fixture_root();
        if !root.exists() {
            return;
        }

        ensure_vkf(&root);
        let reader = LeRobotReader::open(&root).unwrap();

        assert_eq!(reader.len(), 90);
        assert!(!reader.is_empty());
        assert_eq!(reader.metadata().num_episodes, 2);
        assert_eq!(reader.metadata().camera_keys.len(), 1);
    }

    #[test]
    fn test_get_sample() {
        let root = fixture_root();
        if !root.exists() {
            return;
        }

        ensure_vkf(&root);
        let reader = LeRobotReader::open(&root).unwrap();

        let spec = ClipSpec {
            start_frame: 0,
            num_frames: 5,
            stride: 1,
            resolution: (32, 32),
        };

        // Sample from episode 0
        let sample = reader.get(0, &spec).unwrap();
        assert_eq!(sample.episode_index, 0);
        assert_eq!(sample.frame_index, 0);
        assert!(sample.tabular.contains_key("action"));
        assert!(sample.tabular.contains_key("observation.state"));

        let action = &sample.tabular["action"];
        assert_eq!(action.shape, vec![5, 4]); // 5 frames, 4-dim action
        assert_eq!(action.data.len(), 20);

        // Sample from episode 1
        let sample = reader.get(50, &spec).unwrap();
        assert_eq!(sample.episode_index, 1);
        assert_eq!(sample.frame_index, 5); // frame 50 global = frame 5 in episode 1
    }

    #[test]
    fn test_get_sample_out_of_range() {
        let root = fixture_root();
        if !root.exists() {
            return;
        }

        ensure_vkf(&root);
        let reader = LeRobotReader::open(&root).unwrap();

        let spec = ClipSpec {
            start_frame: 0,
            num_frames: 1,
            stride: 1,
            resolution: (32, 32),
        };

        assert!(reader.get(100, &spec).is_err());
    }

    #[test]
    fn test_plan_fetch() {
        let root = fixture_root();
        if !root.exists() {
            return;
        }

        ensure_vkf(&root);
        let reader = LeRobotReader::open(&root).unwrap();

        let spec = ClipSpec {
            start_frame: 0,
            num_frames: 10,
            stride: 1,
            resolution: (32, 32),
        };

        // 6 requests, batch_size = 3 → 2 batches
        let requests: Vec<(usize, ClipSpec)> = vec![
            (0, spec),  // ep 0, frames 0-9 → kf group 0
            (10, spec), // ep 0, frames 10-19 → kf group 0
            (25, spec), // ep 0, frames 25-34 → kf groups 0 + 1
            (45, spec), // ep 1, frames 0-9 → kf group 1 (or wherever ep 1 starts in mp4)
            (55, spec), // ep 1, frames 10-19
            (70, spec), // ep 1, frames 25-34
        ];

        let plan = reader.plan_fetch(&requests, 3).unwrap();

        // Should have some fetch units
        assert!(!plan.units.is_empty());
        // Should have 6 request-to-unit mappings
        assert_eq!(plan.request_to_units.len(), 6);
        // Each request should map to at least 1 unit
        for units in &plan.request_to_units {
            assert!(!units.is_empty());
        }
        // Future-use queues should exist for every unit
        for unit in &plan.units {
            assert!(plan.unit_batches.contains_key(&unit.id));
        }
        // Batch indices should be sorted and deduped
        for queue in plan.unit_batches.values() {
            let v: Vec<_> = queue.iter().collect();
            for i in 1..v.len() {
                assert!(v[i] > v[i - 1], "queue not sorted/deduped");
            }
        }
    }

    #[test]
    fn test_plan_fetch_shared_keyframe_groups() {
        let root = fixture_root();
        if !root.exists() {
            return;
        }

        ensure_vkf(&root);
        let reader = LeRobotReader::open(&root).unwrap();

        // Two samples that should share the same keyframe group
        let spec = ClipSpec {
            start_frame: 0,
            num_frames: 5,
            stride: 1,
            resolution: (32, 32),
        };

        let requests: Vec<(usize, ClipSpec)> = vec![
            (0, spec),  // ep 0, frame 0 → kf group 0
            (10, spec), // ep 0, frame 10 → kf group 0 (same group!)
        ];

        let plan = reader.plan_fetch(&requests, 2).unwrap();

        // Both requests should map to the same unit
        assert_eq!(plan.request_to_units[0], plan.request_to_units[1]);
        // Should have exactly 1 unique unit
        assert_eq!(plan.units.len(), 1);
    }
}
