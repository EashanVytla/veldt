use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;
use veldt_core::{DatasetMeta, Result, VeldtError};

/// Raw info.json as deserialized from disk.
#[derive(Debug, Deserialize)]
pub struct InfoJson {
    pub codebase_version: String,
    pub robot_type: Option<String>,
    pub total_episodes: usize,
    pub total_frames: usize,
    pub total_tasks: usize,
    pub chunks_size: usize,
    pub fps: u32,
    pub splits: HashMap<String, String>,
    pub data_path: String,
    pub video_path: Option<String>,
    pub features: HashMap<String, FeatureInfo>,
}

#[derive(Debug, Deserialize)]
pub struct FeatureInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    #[serde(default)]
    pub names: Option<serde_json::Value>,
    #[serde(default)]
    pub video_info: Option<VideoInfo>,
}

#[derive(Debug, Deserialize)]
pub struct VideoInfo {
    #[serde(rename = "video.fps")]
    pub fps: f64,
    #[serde(rename = "video.codec")]
    pub codec: String,
    #[serde(rename = "video.pix_fmt")]
    pub pix_fmt: String,
}

/// Parsed episode metadata from the episodes parquet files.
#[derive(Debug, Clone)]
pub struct EpisodeInfo {
    pub episode_index: u64,
    pub data_chunk_index: u64,
    pub data_file_index: u64,
    pub dataset_from_index: u64,
    pub dataset_to_index: u64,
    pub length: u64,
    /// Per video key: (chunk_index, file_index, from_timestamp, to_timestamp)
    pub video_info: HashMap<String, VideoEpisodeInfo>,
}

#[derive(Debug, Clone)]
pub struct VideoEpisodeInfo {
    pub chunk_index: u64,
    pub file_index: u64,
    pub from_timestamp: f64,
    pub to_timestamp: f64,
}

/// Load and parse meta/info.json.
pub fn load_info(dataset_root: &Path) -> Result<InfoJson> {
    let info_path = dataset_root.join("meta").join("info.json");
    let content = std::fs::read_to_string(&info_path).map_err(|e| {
        VeldtError::Dataset(format!("failed to read {}: {}", info_path.display(), e))
    })?;
    serde_json::from_str(&content).map_err(|e| {
        VeldtError::Dataset(format!("failed to parse {}: {}", info_path.display(), e))
    })
}

/// Build DatasetMeta from InfoJson.
pub fn build_dataset_meta(info: &InfoJson) -> DatasetMeta {
    let camera_keys: Vec<String> = info
        .features
        .iter()
        .filter(|(_, f)| f.dtype == "video")
        .map(|(k, _)| k.clone())
        .collect();

    let action_dim = info
        .features
        .get("action")
        .map(|f| f.shape.iter().product())
        .unwrap_or(0);

    let state_dim = info
        .features
        .get("observation.state")
        .map(|f| f.shape.iter().product())
        .unwrap_or(0);

    DatasetMeta {
        fps: info.fps as f64,
        num_episodes: info.total_episodes,
        num_frames: info.total_frames,
        camera_keys,
        action_dim,
        state_dim,
    }
}

/// Load episode metadata from all parquet files under meta/episodes/.
pub fn load_episodes(
    dataset_root: &Path,
    info: &InfoJson,
) -> Result<Vec<EpisodeInfo>> {
    let episodes_dir = dataset_root.join("meta").join("episodes");
    if !episodes_dir.exists() {
        return Err(VeldtError::Dataset(format!(
            "episodes directory not found: {}",
            episodes_dir.display()
        )));
    }

    // Collect video keys from features
    let video_keys: Vec<String> = info
        .features
        .iter()
        .filter(|(_, f)| f.dtype == "video")
        .map(|(k, _)| k.clone())
        .collect();

    // Read all parquet files from chunk directories
    let mut episodes = Vec::new();
    let mut chunk_dirs: Vec<_> = std::fs::read_dir(&episodes_dir)
        .map_err(|e| VeldtError::Dataset(format!("read episodes dir: {}", e)))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();
    chunk_dirs.sort_by_key(|e| e.file_name());

    for chunk_dir in chunk_dirs {
        let mut files: Vec<_> = std::fs::read_dir(chunk_dir.path())
            .map_err(|e| VeldtError::Dataset(format!("read chunk dir: {}", e)))?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "parquet")
                    .unwrap_or(false)
            })
            .collect();
        files.sort_by_key(|e| e.file_name());

        for file_entry in files {
            let path = file_entry.path();
            let batch_episodes = read_episodes_parquet(&path, &video_keys)?;
            episodes.extend(batch_episodes);
        }
    }

    // Sort by episode_index
    episodes.sort_by_key(|e| e.episode_index);
    Ok(episodes)
}

/// Read episode metadata from a single parquet file.
fn read_episodes_parquet(
    path: &Path,
    video_keys: &[String],
) -> Result<Vec<EpisodeInfo>> {
    use arrow::array::{Float64Array, Int64Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = std::fs::File::open(path).map_err(|e| {
        VeldtError::Parquet(format!("open {}: {}", path.display(), e))
    })?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
        VeldtError::Parquet(format!("parquet reader for {}: {}", path.display(), e))
    })?;
    let reader = builder.build().map_err(|e| {
        VeldtError::Parquet(format!("build reader for {}: {}", path.display(), e))
    })?;

    let mut episodes = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| {
            VeldtError::Parquet(format!("read batch from {}: {}", path.display(), e))
        })?;

        let episode_idx_col = batch
            .column_by_name("episode_index")
            .ok_or_else(|| VeldtError::Parquet("missing episode_index column".into()))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| VeldtError::Parquet("episode_index not int64".into()))?;

        let data_chunk_col = batch
            .column_by_name("data/chunk_index")
            .ok_or_else(|| VeldtError::Parquet("missing data/chunk_index".into()))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| VeldtError::Parquet("data/chunk_index not int64".into()))?;

        let data_file_col = batch
            .column_by_name("data/file_index")
            .ok_or_else(|| VeldtError::Parquet("missing data/file_index".into()))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| VeldtError::Parquet("data/file_index not int64".into()))?;

        let from_idx_col = batch
            .column_by_name("dataset_from_index")
            .ok_or_else(|| VeldtError::Parquet("missing dataset_from_index".into()))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| VeldtError::Parquet("dataset_from_index not int64".into()))?;

        let to_idx_col = batch
            .column_by_name("dataset_to_index")
            .ok_or_else(|| VeldtError::Parquet("missing dataset_to_index".into()))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| VeldtError::Parquet("dataset_to_index not int64".into()))?;

        let length_col = batch
            .column_by_name("length")
            .ok_or_else(|| VeldtError::Parquet("missing length column".into()))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| VeldtError::Parquet("length not int64".into()))?;

        for row in 0..batch.num_rows() {
            let mut video_info = HashMap::new();
            for vk in video_keys {
                let chunk_col_name = format!("videos/{}/chunk_index", vk);
                let file_col_name = format!("videos/{}/file_index", vk);
                let from_ts_col_name = format!("videos/{}/from_timestamp", vk);
                let to_ts_col_name = format!("videos/{}/to_timestamp", vk);

                let v_chunk = batch
                    .column_by_name(&chunk_col_name)
                    .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                    .map(|c| c.value(row) as u64)
                    .unwrap_or(0);

                let v_file = batch
                    .column_by_name(&file_col_name)
                    .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                    .map(|c| c.value(row) as u64)
                    .unwrap_or(0);

                let from_ts = batch
                    .column_by_name(&from_ts_col_name)
                    .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
                    .map(|c| c.value(row))
                    .unwrap_or(0.0);

                let to_ts = batch
                    .column_by_name(&to_ts_col_name)
                    .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
                    .map(|c| c.value(row))
                    .unwrap_or(0.0);

                video_info.insert(
                    vk.clone(),
                    VideoEpisodeInfo {
                        chunk_index: v_chunk,
                        file_index: v_file,
                        from_timestamp: from_ts,
                        to_timestamp: to_ts,
                    },
                );
            }

            episodes.push(EpisodeInfo {
                episode_index: episode_idx_col.value(row) as u64,
                data_chunk_index: data_chunk_col.value(row) as u64,
                data_file_index: data_file_col.value(row) as u64,
                dataset_from_index: from_idx_col.value(row) as u64,
                dataset_to_index: to_idx_col.value(row) as u64,
                length: length_col.value(row) as u64,
                video_info,
            });
        }
    }

    Ok(episodes)
}

