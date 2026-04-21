use veldt_core::FetchUnitId;

/// A seek-grouped request within a batch.
#[derive(Debug, Clone)]
pub struct GroupedSeek {
    /// The FetchUnitId to read.
    pub unit_id: FetchUnitId,
    /// Original sample indices within the batch that need this unit.
    pub sample_indices: Vec<usize>,
}

/// Sort and deduplicate seeks within a batch.
///
/// Given a batch of (sample_index, required_unit_ids) pairs, groups them
/// by FetchUnitId so each unit is fetched once. Within each group, sample
/// indices are preserved for later frame extraction.
///
/// The returned groups are sorted by FetchUnitId for deterministic ordering
/// and sequential disk access when units come from the same file.
pub fn group_batch_seeks(
    batch_requests: &[(usize, Vec<FetchUnitId>)],
) -> Vec<GroupedSeek> {
    let mut unit_to_samples: std::collections::BTreeMap<FetchUnitId, Vec<usize>> =
        std::collections::BTreeMap::new();

    for (sample_idx, unit_ids) in batch_requests {
        for &unit_id in unit_ids {
            unit_to_samples
                .entry(unit_id)
                .or_default()
                .push(*sample_idx);
        }
    }

    unit_to_samples
        .into_iter()
        .map(|(unit_id, sample_indices)| GroupedSeek {
            unit_id,
            sample_indices,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_batch_seeks_dedup() {
        let a = FetchUnitId(10);
        let b = FetchUnitId(20);

        // Sample 0 needs A, Sample 1 needs A and B, Sample 2 needs B
        let batch = vec![
            (0, vec![a]),
            (1, vec![a, b]),
            (2, vec![b]),
        ];

        let groups = group_batch_seeks(&batch);

        assert_eq!(groups.len(), 2);

        // Sorted by FetchUnitId, so A (10) comes first
        assert_eq!(groups[0].unit_id, a);
        assert_eq!(groups[0].sample_indices, vec![0, 1]);

        assert_eq!(groups[1].unit_id, b);
        assert_eq!(groups[1].sample_indices, vec![1, 2]);
    }

    #[test]
    fn test_group_batch_seeks_single_unit() {
        let a = FetchUnitId(1);

        let batch = vec![
            (0, vec![a]),
            (1, vec![a]),
            (2, vec![a]),
        ];

        let groups = group_batch_seeks(&batch);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].unit_id, a);
        assert_eq!(groups[0].sample_indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_group_batch_seeks_empty() {
        let groups = group_batch_seeks(&[]);
        assert!(groups.is_empty());
    }

    #[test]
    fn test_group_batch_seeks_no_overlap() {
        let a = FetchUnitId(1);
        let b = FetchUnitId(2);
        let c = FetchUnitId(3);

        let batch = vec![
            (0, vec![a]),
            (1, vec![b]),
            (2, vec![c]),
        ];

        let groups = group_batch_seeks(&batch);

        assert_eq!(groups.len(), 3);
        // Each unit has exactly one sample
        for group in &groups {
            assert_eq!(group.sample_indices.len(), 1);
        }
    }
}
