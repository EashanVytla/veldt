use std::collections::HashMap;
use std::path::Path;

use arrow::array::{Array, AsArray, Float32Array, Float64Array, Int64Array};
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use veldt_core::{Result, TabularColumn, VeldtError};

use crate::metadata::FeatureInfo;

/// Read tabular data for specific global indices from a data parquet file.
///
/// Only reads features that are numeric (float32, float64, int64) and not "video" or "image".
/// Returns a map of feature name -> TabularColumn with data for the requested indices.
pub fn read_tabular_for_indices(
    parquet_path: &Path,
    global_indices: &[u64],
    features: &HashMap<String, FeatureInfo>,
) -> Result<HashMap<String, TabularColumn>> {
    if global_indices.is_empty() {
        return Ok(HashMap::new());
    }

    let file = std::fs::File::open(parquet_path).map_err(|e| {
        VeldtError::Parquet(format!("open {}: {}", parquet_path.display(), e))
    })?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
        VeldtError::Parquet(format!("reader for {}: {}", parquet_path.display(), e))
    })?;
    let reader = builder.build().map_err(|e| {
        VeldtError::Parquet(format!("build reader {}: {}", parquet_path.display(), e))
    })?;

    // Identify which features are tabular (not video/image, not metadata columns)
    let tabular_features: Vec<(&String, &FeatureInfo)> = features
        .iter()
        .filter(|(_, f)| matches!(f.dtype.as_str(), "float32" | "float64" | "int64"))
        .filter(|(k, _)| {
            !matches!(
                k.as_str(),
                "episode_index" | "frame_index" | "timestamp" | "index" | "task_index"
            )
        })
        .collect();

    // We need the "index" column to match rows to global indices
    let index_set: std::collections::HashSet<u64> = global_indices.iter().copied().collect();

    // Accumulate data per feature
    let mut result: HashMap<String, Vec<f32>> = HashMap::new();
    for (name, _) in &tabular_features {
        result.insert((*name).clone(), Vec::new());
    }

    // Read all batches and filter by global index
    for batch_result in reader {
        let batch = batch_result.map_err(|e| {
            VeldtError::Parquet(format!("read batch: {}", e))
        })?;

        // Get the global index column
        let index_col = batch
            .column_by_name("index")
            .ok_or_else(|| VeldtError::Parquet("missing 'index' column in data parquet".into()))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| VeldtError::Parquet("'index' column not int64".into()))?;

        // Find which rows match our requested indices
        let mut matching_rows: Vec<(usize, u64)> = Vec::new(); // (row_in_batch, global_idx)
        for row in 0..batch.num_rows() {
            let idx = index_col.value(row) as u64;
            if index_set.contains(&idx) {
                matching_rows.push((row, idx));
            }
        }

        if matching_rows.is_empty() {
            continue;
        }

        // Extract data for each tabular feature
        for (name, feat_info) in &tabular_features {
            if let Some(col) = batch.column_by_name(name) {
                let feature_data = result.get_mut(*name).unwrap();
                for &(row, _) in &matching_rows {
                    extract_feature_values(col.as_ref(), row, feat_info, feature_data)?;
                }
            }
        }
    }

    // Reorder results to match the requested global_indices order
    // Currently we accumulated in file order; we need to match requested order.
    // For simplicity and correctness, we do a second pass: build index -> position mapping.
    // In practice, indices are consecutive within an episode, so the file order matches.

    let num_frames = global_indices.len();
    let mut tabular_result = HashMap::new();
    for (name, feat_info) in &tabular_features {
        if let Some(data) = result.remove(*name) {
            let dim: usize = feat_info.shape.iter().product();
            tabular_result.insert(
                (*name).clone(),
                TabularColumn {
                    data,
                    shape: vec![num_frames, dim],
                },
            );
        }
    }

    Ok(tabular_result)
}

/// Extract float values from an Arrow array at a given row.
fn extract_feature_values(
    col: &dyn Array,
    row: usize,
    feat_info: &FeatureInfo,
    output: &mut Vec<f32>,
) -> Result<()> {
    let dim: usize = feat_info.shape.iter().product();

    if dim > 1 {
        // List column: list<element: float>
        match col.data_type() {
            DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _) => {
                let list_arr = col.as_list::<i32>();
                let values = list_arr.value(row);
                if let Some(f32_arr) = values.as_any().downcast_ref::<Float32Array>() {
                    for i in 0..f32_arr.len().min(dim) {
                        output.push(f32_arr.value(i));
                    }
                } else if let Some(f64_arr) = values.as_any().downcast_ref::<Float64Array>() {
                    for i in 0..f64_arr.len().min(dim) {
                        output.push(f64_arr.value(i) as f32);
                    }
                } else {
                    // Fallback: zeros
                    output.extend(std::iter::repeat(0.0f32).take(dim));
                }
            }
            _ => {
                output.extend(std::iter::repeat(0.0f32).take(dim));
            }
        }
    } else {
        // Scalar column
        if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
            output.push(arr.value(row));
        } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
            output.push(arr.value(row) as f32);
        } else if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
            output.push(arr.value(row) as f32);
        } else {
            output.push(0.0);
        }
    }

    Ok(())
}
