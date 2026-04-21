mod metadata;
mod reader;
mod tabular;

pub use metadata::{EpisodeInfo, InfoJson, VideoEpisodeInfo};
pub use reader::LeRobotReader;

use std::path::Path;
use veldt_core::{DatasetMeta, Result};

impl InfoJson {
    /// Deserialize info.json from a dataset root directory.
    pub fn deserialize_from(root: &Path) -> Result<Self> {
        metadata::load_info(root)
    }
}

/// Build DatasetMeta from a dataset root (loads info.json).
pub fn build_dataset_meta_from(root: &Path) -> Result<DatasetMeta> {
    let info = metadata::load_info(root)?;
    Ok(metadata::build_dataset_meta(&info))
}
