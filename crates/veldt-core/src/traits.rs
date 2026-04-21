use crate::error::Result;
use crate::types::{ClipSpec, DatasetMeta, FetchPlan, Sample};

/// A dataset reader that knows how to access samples and plan I/O.
///
/// Each format (LeRobot v3, RLDS, HDF5, etc.) implements this trait.
/// Everything above the reader — scheduling, prefetch, caching, batching —
/// is format-agnostic and operates through this interface.
pub trait DatasetReader: Send + Sync {
    /// Total number of samples in the dataset.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a single sample by index. Used for random access fallback.
    fn get(&self, index: usize, spec: &ClipSpec) -> Result<Sample>;

    /// Plan fetches for an epoch's worth of requests.
    ///
    /// Given the full shuffle order as (sample_index, ClipSpec) pairs and a
    /// batch_size, builds a `FetchPlan` containing:
    /// - The set of unique fetch units needed
    /// - A mapping from each request to the units it requires
    /// - Future-use queues (unit → sorted batch indices) for Belady scheduling
    fn plan_fetch(
        &self,
        requests: &[(usize, ClipSpec)],
        batch_size: usize,
    ) -> Result<FetchPlan>;

    /// Optional: hint the reader about upcoming fetches (e.g., madvise, open connections).
    fn prefetch_hint(&self, _plan: &FetchPlan) {}

    /// Return dataset-level metadata.
    fn metadata(&self) -> &DatasetMeta;
}
