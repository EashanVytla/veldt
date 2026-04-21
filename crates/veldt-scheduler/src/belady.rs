use std::collections::{HashMap, VecDeque};
use veldt_core::{ClipSpec, FetchPlan, FetchUnitId};

/// Prefetch buckets: indexed by batch number, each bucket contains the
/// FetchUnitIds that should be prefetched when that batch enters the
/// lookahead window.
#[derive(Debug, Clone)]
pub struct PrefetchBuckets {
    buckets: Vec<Vec<FetchUnitId>>,
    num_batches: usize,
}

impl PrefetchBuckets {
    /// Build prefetch buckets from a FetchPlan's future-use queues.
    ///
    /// Each unit is inserted into the bucket corresponding to its first needed batch
    /// (i.e., `unit_batches[unit_id].front()`).
    pub fn build(plan: &FetchPlan, num_batches: usize) -> Self {
        let mut buckets = vec![Vec::new(); num_batches];

        for (unit_id, queue) in &plan.unit_batches {
            if let Some(&first_batch) = queue.front() {
                if first_batch < num_batches {
                    buckets[first_batch].push(*unit_id);
                }
            }
        }

        PrefetchBuckets {
            buckets,
            num_batches,
        }
    }

    /// Get the units that should be prefetched when `batch_index` enters
    /// the lookahead window. Returns the bucket at `batch_index`, or empty
    /// if out of range.
    pub fn get_bucket(&self, batch_index: usize) -> &[FetchUnitId] {
        if batch_index < self.num_batches {
            &self.buckets[batch_index]
        } else {
            &[]
        }
    }

    /// Re-insert a unit into the bucket for its next needed batch.
    /// Called when a unit is evicted from the cache but still has future uses.
    pub fn reinsert(&mut self, unit_id: FetchUnitId, next_needed_batch: usize) {
        if next_needed_batch < self.num_batches {
            self.buckets[next_needed_batch].push(unit_id);
        }
    }

    pub fn num_batches(&self) -> usize {
        self.num_batches
    }
}

/// The Belady scheduler: builds the data structures that drive prefetch
/// and eviction decisions.
pub struct BeladyScheduler {
    pub prefetch_depth: usize,
}

impl BeladyScheduler {
    pub fn new(prefetch_depth: usize) -> Self {
        BeladyScheduler { prefetch_depth }
    }

    /// Build prefetch buckets from a FetchPlan.
    pub fn build_prefetch_buckets(
        &self,
        plan: &FetchPlan,
        num_batches: usize,
    ) -> PrefetchBuckets {
        PrefetchBuckets::build(plan, num_batches)
    }

    /// Get all units that should be prefetched for the current batch,
    /// considering the lookahead window.
    ///
    /// At batch `current_batch`, we want to ensure all units needed by
    /// batches `[current_batch, current_batch + prefetch_depth]` are
    /// either cached or being fetched.
    pub fn units_to_prefetch(
        &self,
        buckets: &PrefetchBuckets,
        current_batch: usize,
    ) -> Vec<FetchUnitId> {
        let target = current_batch + self.prefetch_depth;
        if target < buckets.num_batches() {
            buckets.get_bucket(target).to_vec()
        } else {
            Vec::new()
        }
    }
}

/// Build future-use queues from a shuffle order and a function that maps
/// each sample to its required FetchUnitIds.
///
/// This is a helper for DatasetReader::plan_fetch implementations.
/// Given the full shuffle order and batch_size, produces
/// `HashMap<FetchUnitId, VecDeque<usize>>` where each queue contains
/// the sorted, deduplicated batch indices that need that unit.
pub fn build_future_use_queues(
    _requests: &[(usize, ClipSpec)],
    batch_size: usize,
    sample_to_units: &[Vec<FetchUnitId>],
) -> HashMap<FetchUnitId, VecDeque<usize>> {
    let mut queues: HashMap<FetchUnitId, VecDeque<usize>> = HashMap::new();

    for (i, unit_ids) in sample_to_units.iter().enumerate() {
        let batch_index = i / batch_size;
        for &unit_id in unit_ids {
            let queue = queues.entry(unit_id).or_default();
            // Deduplicate: only push if this batch isn't already the last entry
            if queue.back() != Some(&batch_index) {
                queue.push_back(batch_index);
            }
        }
    }

    queues
}

#[cfg(test)]
mod tests {
    use super::*;
    use veldt_core::{FetchSource, FetchUnit};
    use std::path::PathBuf;

    fn make_unit(id: u64, num_frames: u32) -> FetchUnit {
        FetchUnit {
            id: FetchUnitId(id),
            source: FetchSource::LocalFile {
                path: PathBuf::from("test.mp4"),
                byte_offset: 0,
                byte_len: 1000,
            },
            estimated_bytes: (num_frames as u64) * 1000,
            num_decoded_frames: num_frames,
        }
    }

    fn make_plan(
        units: Vec<FetchUnit>,
        request_to_units: Vec<Vec<FetchUnitId>>,
        unit_batches: HashMap<FetchUnitId, VecDeque<usize>>,
    ) -> FetchPlan {
        FetchPlan {
            units,
            request_to_units,
            unit_batches,
        }
    }

    #[test]
    fn test_build_future_use_queues_basic() {
        // 6 samples, batch_size=2 => 3 batches
        // Sample 0,1 -> batch 0; Sample 2,3 -> batch 1; Sample 4,5 -> batch 2
        let requests: Vec<(usize, ClipSpec)> = (0..6)
            .map(|i| {
                (
                    i,
                    ClipSpec {
                        start_frame: 0,
                        num_frames: 10,
                        stride: 1,
                        resolution: (224, 224),
                    },
                )
            })
            .collect();

        let a = FetchUnitId(0);
        let b = FetchUnitId(1);
        let c = FetchUnitId(2);

        // Sample 0 needs A, Sample 1 needs A,B, Sample 2 needs B,
        // Sample 3 needs C, Sample 4 needs A, Sample 5 needs C
        let sample_to_units = vec![
            vec![a],
            vec![a, b],
            vec![b],
            vec![c],
            vec![a],
            vec![c],
        ];

        let queues = build_future_use_queues(&requests, 2, &sample_to_units);

        // A: used by samples 0,1 (batch 0) and sample 4 (batch 2)
        assert_eq!(queues[&a], VecDeque::from([0, 2]));
        // B: used by samples 1 (batch 0) and 2 (batch 1)
        assert_eq!(queues[&b], VecDeque::from([0, 1]));
        // C: used by samples 3 (batch 1) and 5 (batch 2)
        assert_eq!(queues[&c], VecDeque::from([1, 2]));
    }

    #[test]
    fn test_build_future_use_queues_dedup() {
        // Two samples in the same batch both need unit A
        let requests: Vec<(usize, ClipSpec)> = (0..2)
            .map(|i| {
                (
                    i,
                    ClipSpec {
                        start_frame: 0,
                        num_frames: 10,
                        stride: 1,
                        resolution: (224, 224),
                    },
                )
            })
            .collect();

        let a = FetchUnitId(0);
        let sample_to_units = vec![vec![a], vec![a]];

        let queues = build_future_use_queues(&requests, 2, &sample_to_units);

        // Both in batch 0, should deduplicate to single entry
        assert_eq!(queues[&a], VecDeque::from([0]));
    }

    #[test]
    fn test_prefetch_buckets_build() {
        let a = FetchUnitId(0);
        let b = FetchUnitId(1);
        let c = FetchUnitId(2);

        let mut unit_batches = HashMap::new();
        unit_batches.insert(a, VecDeque::from([0, 2]));
        unit_batches.insert(b, VecDeque::from([1, 3]));
        unit_batches.insert(c, VecDeque::from([2]));

        let plan = make_plan(
            vec![make_unit(0, 30), make_unit(1, 30), make_unit(2, 30)],
            vec![],
            unit_batches,
        );

        let buckets = PrefetchBuckets::build(&plan, 4);

        // A's first needed batch = 0
        assert!(buckets.get_bucket(0).contains(&a));
        // B's first needed batch = 1
        assert!(buckets.get_bucket(1).contains(&b));
        // C's first needed batch = 2
        assert!(buckets.get_bucket(2).contains(&c));
        // Bucket 3 should be empty (no unit starts there)
        assert!(buckets.get_bucket(3).is_empty());
    }

    #[test]
    fn test_prefetch_buckets_reinsert() {
        let a = FetchUnitId(0);

        let mut unit_batches = HashMap::new();
        unit_batches.insert(a, VecDeque::from([0]));

        let plan = make_plan(vec![make_unit(0, 30)], vec![], unit_batches);

        let mut buckets = PrefetchBuckets::build(&plan, 5);

        // Simulate eviction: A still has a future use at batch 3
        buckets.reinsert(a, 3);

        assert!(buckets.get_bucket(3).contains(&a));
    }

    #[test]
    fn test_units_to_prefetch() {
        let a = FetchUnitId(0);
        let b = FetchUnitId(1);

        let mut unit_batches = HashMap::new();
        unit_batches.insert(a, VecDeque::from([0]));
        unit_batches.insert(b, VecDeque::from([3]));

        let plan = make_plan(
            vec![make_unit(0, 30), make_unit(1, 30)],
            vec![],
            unit_batches,
        );

        let scheduler = BeladyScheduler::new(3); // look 3 batches ahead
        let buckets = PrefetchBuckets::build(&plan, 5);

        // At batch 0, lookahead target = 0 + 3 = 3, so prefetch bucket[3]
        let to_prefetch = scheduler.units_to_prefetch(&buckets, 0);
        assert!(to_prefetch.contains(&b));
        assert!(!to_prefetch.contains(&a)); // A is in bucket 0, already past

        // At batch 2, lookahead target = 2 + 3 = 5, out of range
        let to_prefetch = scheduler.units_to_prefetch(&buckets, 2);
        assert!(to_prefetch.is_empty());
    }

    #[test]
    fn test_empty_plan() {
        let plan = make_plan(vec![], vec![], HashMap::new());
        let buckets = PrefetchBuckets::build(&plan, 0);
        assert_eq!(buckets.num_batches(), 0);

        let scheduler = BeladyScheduler::new(64);
        let to_prefetch = scheduler.units_to_prefetch(&buckets, 0);
        assert!(to_prefetch.is_empty());
    }
}
