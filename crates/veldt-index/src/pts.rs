/// PTS (Presentation Timestamp) utilities.
///
/// The core PTS computation is done in `moov::build_pts_map`.
/// This module provides helper functions for PTS lookups.

/// Find which keyframe group a frame belongs to, given sorted keyframe
/// sample indices (0-based).
///
/// Returns the index into the keyframe array.
pub fn keyframe_group_for_frame(
    frame_index: usize,
    keyframe_first_frames: &[usize],
) -> usize {
    match keyframe_first_frames.binary_search(&frame_index) {
        Ok(i) => i,
        Err(i) => i.saturating_sub(1),
    }
}

/// Dummy struct to keep the public API from lib.rs working.
/// Actual PTS data lives in `VkfIndex::frame_pts`.
pub struct PtsMap;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyframe_group_lookup() {
        // Keyframes at samples 0, 30, 60, 90
        let keyframes = vec![0, 30, 60, 90];

        assert_eq!(keyframe_group_for_frame(0, &keyframes), 0);
        assert_eq!(keyframe_group_for_frame(15, &keyframes), 0);
        assert_eq!(keyframe_group_for_frame(29, &keyframes), 0);
        assert_eq!(keyframe_group_for_frame(30, &keyframes), 1);
        assert_eq!(keyframe_group_for_frame(59, &keyframes), 1);
        assert_eq!(keyframe_group_for_frame(60, &keyframes), 2);
        assert_eq!(keyframe_group_for_frame(90, &keyframes), 3);
        assert_eq!(keyframe_group_for_frame(99, &keyframes), 3);
    }
}
