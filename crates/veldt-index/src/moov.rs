use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use veldt_core::{Result, VeldtError};

use crate::vkf::{KeyframeEntry, VkfIndex};

/// Codec identifier stored in the VKF index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecId {
    H264,
    H265,
    VP9,
    AV1,
    Unknown,
}

impl CodecId {
    pub fn to_fourcc(self) -> u32 {
        match self {
            CodecId::H264 => u32::from_be_bytes(*b"avc1"),
            CodecId::H265 => u32::from_be_bytes(*b"hev1"),
            CodecId::VP9 => u32::from_be_bytes(*b"vp09"),
            CodecId::AV1 => u32::from_be_bytes(*b"av01"),
            CodecId::Unknown => 0,
        }
    }

    pub fn from_fourcc(fourcc: u32) -> Self {
        let bytes = fourcc.to_be_bytes();
        match &bytes {
            b"avc1" => CodecId::H264,
            b"hev1" => CodecId::H265,
            b"vp09" => CodecId::VP9,
            b"av01" => CodecId::AV1,
            _ => CodecId::Unknown,
        }
    }
}

/// Parse an MP4 file and build a VkfIndex.
///
/// Reads the moov atom to extract keyframe byte offsets, sample sizes,
/// chunk offsets, sample-to-chunk mapping, and PTS information for
/// the first video track found.
pub fn parse_mp4(path: &Path) -> Result<VkfIndex> {
    let file = File::open(path).map_err(|e| VeldtError::Mp4Parse {
        path: path.to_path_buf(),
        source: Box::new(e),
    })?;
    let size = file
        .metadata()
        .map_err(|e| VeldtError::Mp4Parse {
            path: path.to_path_buf(),
            source: Box::new(e),
        })?
        .len();
    let reader = BufReader::new(file);

    let mp4 = mp4::Mp4Reader::read_header(reader, size).map_err(|e| VeldtError::Mp4Parse {
        path: path.to_path_buf(),
        source: Box::new(e),
    })?;

    // Find the first video track
    let video_track = mp4
        .tracks()
        .values()
        .find(|t| {
            t.track_type()
                .map(|tt| tt == mp4::TrackType::Video)
                .unwrap_or(false)
        })
        .ok_or_else(|| VeldtError::Mp4Parse {
            path: path.to_path_buf(),
            source: "no video track found".into(),
        })?;

    let track_id = video_track.track_id();
    let timescale = video_track.timescale();
    let width = video_track.width();
    let height = video_track.height();
    let sample_count = video_track.sample_count() as usize;

    // Determine codec
    let codec = match video_track.media_type() {
        Ok(mp4::MediaType::H264) => CodecId::H264,
        Ok(mp4::MediaType::H265) => CodecId::H265,
        Ok(mp4::MediaType::VP9) => CodecId::VP9,
        _ => CodecId::Unknown,
    };

    // Access the raw stbl boxes through the trak
    let trak = mp4
        .moov
        .traks
        .iter()
        .find(|t| t.tkhd.track_id == track_id)
        .ok_or_else(|| VeldtError::Mp4Parse {
            path: path.to_path_buf(),
            source: "trak not found".into(),
        })?;

    let stbl = &trak.mdia.minf.stbl;

    // === Extract all data from mp4 types into plain Vecs ===

    // Keyframe sample numbers (1-based) from stss
    // If stss is absent, all samples are keyframes
    let keyframe_samples: Vec<u32> = match &stbl.stss {
        Some(stss) => stss.entries.clone(),
        None => (1..=(sample_count as u32)).collect(),
    };

    // Sample sizes from stsz
    let sample_sizes: Vec<u64> = if stbl.stsz.sample_size > 0 {
        vec![stbl.stsz.sample_size as u64; sample_count]
    } else {
        stbl.stsz.sample_sizes.iter().map(|&s| s as u64).collect()
    };

    // Chunk offsets from stco or co64
    let chunk_offsets: Vec<u64> = if let Some(ref stco) = stbl.stco {
        stco.entries.iter().map(|&o| o as u64).collect()
    } else if let Some(ref co64) = stbl.co64 {
        co64.entries.clone()
    } else {
        return Err(VeldtError::VkfIndex(
            "neither stco nor co64 found".to_string(),
        ));
    };

    // Sample-to-chunk entries: (first_chunk, samples_per_chunk)
    let stsc_entries: Vec<(u32, u32)> = stbl
        .stsc
        .entries
        .iter()
        .map(|e| (e.first_chunk, e.samples_per_chunk))
        .collect();

    // Time-to-sample entries: (sample_count, sample_delta)
    let stts_entries: Vec<(u32, u32)> = stbl
        .stts
        .entries
        .iter()
        .map(|e| (e.sample_count, e.sample_delta))
        .collect();

    // Composition time offset entries (optional): (sample_count, sample_offset)
    let ctts_entries: Option<Vec<(u32, i32)>> = stbl.ctts.as_ref().map(|ctts| {
        ctts.entries
            .iter()
            .map(|e| (e.sample_count, e.sample_offset))
            .collect()
    });

    // === Build derived data from plain Vecs ===

    // Build per-sample byte offsets
    let sample_offsets =
        build_sample_offsets(&chunk_offsets, &stsc_entries, &sample_sizes, sample_count);

    // Build PTS map
    let frame_pts = build_pts_map(&stts_entries, &ctts_entries, sample_count);

    // Build keyframe entries
    let num_keyframes = keyframe_samples.len();
    let mut keyframes = Vec::with_capacity(num_keyframes);

    for (i, &kf_sample_1based) in keyframe_samples.iter().enumerate() {
        let kf_sample = (kf_sample_1based - 1) as usize; // 0-based

        let byte_offset = sample_offsets[kf_sample];

        // Byte range: from this keyframe to the start of the next (or end of last sample)
        let next_kf_sample = if i + 1 < num_keyframes {
            (keyframe_samples[i + 1] - 1) as usize
        } else {
            sample_count
        };

        let byte_end = if next_kf_sample < sample_count {
            sample_offsets[next_kf_sample]
        } else {
            let last = sample_count - 1;
            sample_offsets[last] + sample_sizes[last]
        };

        let byte_len = byte_end - byte_offset;
        let pts = frame_pts[kf_sample];
        let group_frames = (next_kf_sample - kf_sample) as u32;

        keyframes.push(KeyframeEntry {
            byte_offset,
            byte_len,
            pts,
            num_frames: group_frames,
        });
    }

    Ok(VkfIndex {
        version: 1,
        codec_fourcc: codec.to_fourcc(),
        width,
        height,
        timescale,
        num_keyframes: num_keyframes as u32,
        num_frames: sample_count as u32,
        keyframes,
        frame_pts,
    })
}

/// Build per-sample byte offsets from chunk offsets, stsc, and sample sizes.
fn build_sample_offsets(
    chunk_offsets: &[u64],
    stsc_entries: &[(u32, u32)], // (first_chunk_1based, samples_per_chunk)
    sample_sizes: &[u64],
    sample_count: usize,
) -> Vec<u64> {
    let mut offsets = vec![0u64; sample_count];
    let mut sample_idx = 0usize;

    for chunk_idx in 0..chunk_offsets.len() {
        let chunk_num = (chunk_idx + 1) as u32;
        let samples_in_chunk = stsc_samples_per_chunk(stsc_entries, chunk_num);

        let mut offset = chunk_offsets[chunk_idx];
        for _ in 0..samples_in_chunk {
            if sample_idx >= sample_count {
                break;
            }
            offsets[sample_idx] = offset;
            offset += sample_sizes[sample_idx];
            sample_idx += 1;
        }
    }

    offsets
}

/// Given stsc entries and a 1-based chunk number, return samples_per_chunk.
fn stsc_samples_per_chunk(entries: &[(u32, u32)], chunk_num: u32) -> u32 {
    let mut result = entries[0].1;
    for &(first_chunk, samples_per_chunk) in entries {
        if first_chunk <= chunk_num {
            result = samples_per_chunk;
        } else {
            break;
        }
    }
    result
}

/// Build PTS array from stts + optional ctts entries.
fn build_pts_map(
    stts_entries: &[(u32, u32)],
    ctts_entries: &Option<Vec<(u32, i32)>>,
    sample_count: usize,
) -> Vec<u64> {
    // Decode timestamps from stts
    let mut dts = Vec::with_capacity(sample_count);
    let mut current_dts: u64 = 0;
    for &(count, delta) in stts_entries {
        for _ in 0..count {
            dts.push(current_dts);
            current_dts += delta as u64;
        }
    }
    dts.resize(sample_count, current_dts);

    // Apply composition time offsets from ctts
    if let Some(ctts) = ctts_entries {
        let mut sample_idx = 0;
        for &(count, offset) in ctts {
            for _ in 0..count {
                if sample_idx < dts.len() {
                    dts[sample_idx] = (dts[sample_idx] as i64 + offset as i64) as u64;
                }
                sample_idx += 1;
            }
        }
    }

    dts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_id_roundtrip() {
        for codec in [CodecId::H264, CodecId::H265, CodecId::VP9, CodecId::AV1] {
            assert_eq!(CodecId::from_fourcc(codec.to_fourcc()), codec);
        }
    }

    #[test]
    fn test_stsc_samples_per_chunk() {
        // Single entry covering all chunks
        let entries = vec![(1u32, 10u32)];
        assert_eq!(stsc_samples_per_chunk(&entries, 1), 10);
        assert_eq!(stsc_samples_per_chunk(&entries, 5), 10);

        // Two entries: chunks 1-3 have 5 samples, chunks 4+ have 3
        let entries = vec![(1, 5), (4, 3)];
        assert_eq!(stsc_samples_per_chunk(&entries, 1), 5);
        assert_eq!(stsc_samples_per_chunk(&entries, 3), 5);
        assert_eq!(stsc_samples_per_chunk(&entries, 4), 3);
        assert_eq!(stsc_samples_per_chunk(&entries, 10), 3);
    }

    #[test]
    fn test_build_sample_offsets() {
        // 2 chunks, 3 samples per chunk, sizes: [100, 200, 150, 300, 250, 100]
        let chunk_offsets = vec![1000u64, 2000];
        let stsc_entries = vec![(1u32, 3u32)];
        let sample_sizes = vec![100u64, 200, 150, 300, 250, 100];

        let offsets = build_sample_offsets(&chunk_offsets, &stsc_entries, &sample_sizes, 6);

        assert_eq!(offsets[0], 1000); // chunk 1, sample 0
        assert_eq!(offsets[1], 1100); // chunk 1, sample 1
        assert_eq!(offsets[2], 1300); // chunk 1, sample 2
        assert_eq!(offsets[3], 2000); // chunk 2, sample 3
        assert_eq!(offsets[4], 2300); // chunk 2, sample 4
        assert_eq!(offsets[5], 2550); // chunk 2, sample 5
    }

    #[test]
    fn test_build_pts_map_simple() {
        // Constant frame rate: 30fps, timescale 30000 → delta = 1000
        let stts = vec![(90u32, 1000u32)];
        let pts = build_pts_map(&stts, &None, 90);

        assert_eq!(pts.len(), 90);
        assert_eq!(pts[0], 0);
        assert_eq!(pts[1], 1000);
        assert_eq!(pts[29], 29000);
        assert_eq!(pts[89], 89000);
    }

    #[test]
    fn test_build_pts_map_with_ctts() {
        let stts = vec![(4u32, 1000u32)];
        // ctts: B-frames with reordering
        let ctts = Some(vec![
            (1u32, 0i32),
            (1, 2000),
            (1, 1000),
            (1, 0),
        ]);
        let pts = build_pts_map(&stts, &ctts, 4);

        assert_eq!(pts[0], 0);     // dts=0, ctts=0
        assert_eq!(pts[1], 3000);  // dts=1000, ctts=2000
        assert_eq!(pts[2], 3000);  // dts=2000, ctts=1000
        assert_eq!(pts[3], 3000);  // dts=3000, ctts=0
    }

    fn fixture_path() -> std::path::PathBuf {
        let manifest = env!("CARGO_MANIFEST_DIR");
        std::path::PathBuf::from(manifest)
            .parent().unwrap()
            .parent().unwrap()
            .join("tests/fixtures/mini_lerobot/test_video.mp4")
    }

    #[test]
    fn test_parse_real_mp4() {
        let path = fixture_path();
        if !path.exists() {
            eprintln!("skipping: fixture not found at {:?}", path);
            return;
        }

        let index = parse_mp4(&path).unwrap();

        // 3 seconds at 30fps = 90 frames
        assert_eq!(index.num_frames, 90);
        // -g 30 means keyframes at 0, 30, 60 → 3 keyframe groups
        assert_eq!(index.num_keyframes, 3);
        assert_eq!(index.width, 96);
        assert_eq!(index.height, 96);
        assert_eq!(index.codec_fourcc, CodecId::H264.to_fourcc());

        // Keyframe groups should have 30 frames each
        for kf in &index.keyframes {
            assert_eq!(kf.num_frames, 30);
            assert!(kf.byte_len > 0);
        }

        // PTS should be monotonically increasing
        assert_eq!(index.frame_pts.len(), 90);
        for i in 1..index.frame_pts.len() {
            assert!(index.frame_pts[i] > index.frame_pts[i - 1]);
        }
    }

    #[test]
    fn test_parse_and_vkf_roundtrip() {
        let path = fixture_path();
        if !path.exists() {
            return;
        }

        let index = parse_mp4(&path).unwrap();

        let dir = std::env::temp_dir().join("veldt_test_moov_roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let vkf_path = dir.join("test.vkf");

        index.write_to(&vkf_path).unwrap();
        let loaded = crate::vkf::VkfIndex::read_from(&vkf_path).unwrap();

        assert_eq!(loaded.num_frames, index.num_frames);
        assert_eq!(loaded.num_keyframes, index.num_keyframes);
        assert_eq!(loaded.frame_pts, index.frame_pts);

        std::fs::remove_dir_all(&dir).ok();
    }
}
