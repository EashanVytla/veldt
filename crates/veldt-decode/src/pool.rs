use rayon::prelude::*;

use veldt_core::Result;

use crate::decoder::{decode_clip, DecodeRequest, DecodedClip};

/// A Rayon-based parallel decode pool.
///
/// Each decode runs in its own Rayon task with an independent ffmpeg decoder
/// context (ffmpeg codec contexts are !Send, so each thread creates its own).
pub struct DecodePool {
    pool: rayon::ThreadPool,
}

impl DecodePool {
    /// Create a new decode pool with the specified number of threads.
    /// If `num_threads` is 0, uses the number of physical CPU cores.
    pub fn new(num_threads: usize) -> Result<Self> {
        let num_threads = if num_threads == 0 {
            num_cpus()
        } else {
            num_threads
        };

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| veldt_core::VeldtError::Decode(format!("thread pool: {}", e)))?;

        Ok(DecodePool { pool })
    }

    /// Decode multiple clips in parallel.
    pub fn decode_batch(&self, requests: Vec<DecodeRequest>) -> Vec<Result<DecodedClip>> {
        self.pool.install(|| {
            requests
                .par_iter()
                .map(|req| decode_clip(req))
                .collect()
        })
    }

    /// Decode a single clip (for convenience).
    pub fn decode_one(&self, request: DecodeRequest) -> Result<DecodedClip> {
        self.pool.install(|| decode_clip(&request))
    }

    /// Number of threads in the pool.
    pub fn num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = DecodePool::new(2).unwrap();
        assert_eq!(pool.num_threads(), 2);
    }

    #[test]
    fn test_pool_default_threads() {
        let pool = DecodePool::new(0).unwrap();
        assert!(pool.num_threads() > 0);
    }
}
