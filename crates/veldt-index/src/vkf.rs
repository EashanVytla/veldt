use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use veldt_core::{Result, VeldtError};

const VKF_MAGIC: &[u8; 4] = b"VKF1";
const VKF_VERSION: u16 = 1;

/// A single keyframe group entry in the VKF index.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct KeyframeEntry {
    /// Absolute byte offset of the keyframe group in the MP4 file.
    pub byte_offset: u64,
    /// Byte length from this keyframe to the next (or EOF).
    pub byte_len: u64,
    /// Presentation timestamp in timescale units.
    pub pts: u64,
    /// Number of frames in this keyframe group.
    pub num_frames: u32,
}

/// The VKF sidecar index: precomputed keyframe information for an MP4 file.
///
/// Binary format:
/// ```text
/// Header (32 bytes):
///   magic:          [u8; 4]   = b"VKF1"
///   version:        u16       = 1
///   flags:          u16       = 0
///   num_keyframes:  u32
///   num_frames:     u32
///   timescale:      u32
///   codec_fourcc:   u32       // e.g., b"avc1"
///   width:          u16
///   height:         u16
///   reserved:       [u8; 4]
///
/// Keyframe table (num_keyframes * 28 bytes each):
///   byte_offset:    u64
///   byte_len:       u64
///   pts:            u64
///   num_frames:     u32
///
/// Frame PTS table (num_frames * 8 bytes each):
///   pts:            u64
///
/// Footer:
///   crc32:          u32       // over header + both tables
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VkfIndex {
    pub version: u16,
    pub codec_fourcc: u32,
    pub width: u16,
    pub height: u16,
    pub timescale: u32,
    pub num_keyframes: u32,
    pub num_frames: u32,
    pub keyframes: Vec<KeyframeEntry>,
    pub frame_pts: Vec<u64>,
}

impl VkfIndex {
    /// Write this index to a binary VKF file.
    pub fn write_to(&self, path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);
        let mut hasher = Crc32Hasher::new();

        // Header (32 bytes)
        write_and_hash(&mut w, &mut hasher, VKF_MAGIC)?;
        write_and_hash(&mut w, &mut hasher, &VKF_VERSION.to_le_bytes())?;
        write_and_hash(&mut w, &mut hasher, &0u16.to_le_bytes())?; // flags
        write_and_hash(&mut w, &mut hasher, &self.num_keyframes.to_le_bytes())?;
        write_and_hash(&mut w, &mut hasher, &self.num_frames.to_le_bytes())?;
        write_and_hash(&mut w, &mut hasher, &self.timescale.to_le_bytes())?;
        write_and_hash(&mut w, &mut hasher, &self.codec_fourcc.to_le_bytes())?;
        write_and_hash(&mut w, &mut hasher, &self.width.to_le_bytes())?;
        write_and_hash(&mut w, &mut hasher, &self.height.to_le_bytes())?;
        write_and_hash(&mut w, &mut hasher, &[0u8; 4])?; // reserved

        // Keyframe table
        for kf in &self.keyframes {
            write_and_hash(&mut w, &mut hasher, &kf.byte_offset.to_le_bytes())?;
            write_and_hash(&mut w, &mut hasher, &kf.byte_len.to_le_bytes())?;
            write_and_hash(&mut w, &mut hasher, &kf.pts.to_le_bytes())?;
            write_and_hash(&mut w, &mut hasher, &kf.num_frames.to_le_bytes())?;
        }

        // Frame PTS table
        for &pts in &self.frame_pts {
            write_and_hash(&mut w, &mut hasher, &pts.to_le_bytes())?;
        }

        // CRC32 footer
        let crc = hasher.finalize();
        w.write_all(&crc.to_le_bytes())?;
        w.flush()?;

        Ok(())
    }

    /// Read a VKF index from a binary file.
    pub fn read_from(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut r = BufReader::new(file);
        let mut hasher = Crc32Hasher::new();

        // Header
        let mut magic = [0u8; 4];
        read_and_hash(&mut r, &mut hasher, &mut magic)?;
        if &magic != VKF_MAGIC {
            return Err(VeldtError::VkfIndex(format!(
                "invalid magic: {:?}",
                magic
            )));
        }

        let version = read_u16_and_hash(&mut r, &mut hasher)?;
        if version != VKF_VERSION {
            return Err(VeldtError::VkfIndex(format!(
                "unsupported version: {}",
                version
            )));
        }

        let _flags = read_u16_and_hash(&mut r, &mut hasher)?;
        let num_keyframes = read_u32_and_hash(&mut r, &mut hasher)?;
        let num_frames = read_u32_and_hash(&mut r, &mut hasher)?;
        let timescale = read_u32_and_hash(&mut r, &mut hasher)?;
        let codec_fourcc = read_u32_and_hash(&mut r, &mut hasher)?;
        let width = read_u16_and_hash(&mut r, &mut hasher)?;
        let height = read_u16_and_hash(&mut r, &mut hasher)?;
        let mut reserved = [0u8; 4];
        read_and_hash(&mut r, &mut hasher, &mut reserved)?;

        // Keyframe table
        let mut keyframes = Vec::with_capacity(num_keyframes as usize);
        for _ in 0..num_keyframes {
            let byte_offset = read_u64_and_hash(&mut r, &mut hasher)?;
            let byte_len = read_u64_and_hash(&mut r, &mut hasher)?;
            let pts = read_u64_and_hash(&mut r, &mut hasher)?;
            let num_kf_frames = read_u32_and_hash(&mut r, &mut hasher)?;
            keyframes.push(KeyframeEntry {
                byte_offset,
                byte_len,
                pts,
                num_frames: num_kf_frames,
            });
        }

        // Frame PTS table
        let mut frame_pts = Vec::with_capacity(num_frames as usize);
        for _ in 0..num_frames {
            frame_pts.push(read_u64_and_hash(&mut r, &mut hasher)?);
        }

        // CRC32 footer
        let expected_crc = hasher.finalize();
        let mut crc_bytes = [0u8; 4];
        r.read_exact(&mut crc_bytes)?;
        let stored_crc = u32::from_le_bytes(crc_bytes);

        if stored_crc != expected_crc {
            return Err(VeldtError::VkfIndex(format!(
                "CRC mismatch: stored={:#x}, computed={:#x}",
                stored_crc, expected_crc
            )));
        }

        Ok(VkfIndex {
            version,
            codec_fourcc,
            width,
            height,
            timescale,
            num_keyframes,
            num_frames,
            keyframes,
            frame_pts,
        })
    }

    /// Get the VKF sidecar path for an MP4 file (same name, .vkf extension).
    pub fn sidecar_path(mp4_path: &Path) -> std::path::PathBuf {
        mp4_path.with_extension("vkf")
    }

    /// Find which keyframe group contains the given 0-based frame index.
    /// Returns the keyframe group index.
    pub fn keyframe_group_for_frame(&self, frame_index: u32) -> Option<usize> {
        if frame_index >= self.num_frames {
            return None;
        }

        // Compute cumulative frame counts to find the right group
        let mut cumulative = 0u32;
        for (i, kf) in self.keyframes.iter().enumerate() {
            if frame_index < cumulative + kf.num_frames {
                return Some(i);
            }
            cumulative += kf.num_frames;
        }

        // Should not reach here if frame_index < num_frames
        Some(self.keyframes.len() - 1)
    }

    /// Get the byte range for a keyframe group.
    pub fn keyframe_byte_range(&self, group_index: usize) -> Option<(u64, u64)> {
        self.keyframes
            .get(group_index)
            .map(|kf| (kf.byte_offset, kf.byte_len))
    }

    /// Get the first frame index of a keyframe group.
    pub fn keyframe_first_frame(&self, group_index: usize) -> u32 {
        self.keyframes[..group_index]
            .iter()
            .map(|kf| kf.num_frames)
            .sum()
    }
}

// --- CRC32 helper (IEEE polynomial, no external dep) ---

struct Crc32Hasher {
    crc: u32,
}

impl Crc32Hasher {
    fn new() -> Self {
        Crc32Hasher { crc: 0xFFFF_FFFF }
    }

    fn update(&mut self, data: &[u8]) {
        for &byte in data {
            let index = ((self.crc ^ byte as u32) & 0xFF) as usize;
            self.crc = CRC32_TABLE[index] ^ (self.crc >> 8);
        }
    }

    fn finalize(self) -> u32 {
        self.crc ^ 0xFFFF_FFFF
    }
}

fn write_and_hash(
    w: &mut impl Write,
    hasher: &mut Crc32Hasher,
    data: &[u8],
) -> Result<()> {
    w.write_all(data)?;
    hasher.update(data);
    Ok(())
}

fn read_and_hash(
    r: &mut impl Read,
    hasher: &mut Crc32Hasher,
    buf: &mut [u8],
) -> Result<()> {
    r.read_exact(buf)?;
    hasher.update(buf);
    Ok(())
}

fn read_u16_and_hash(r: &mut impl Read, hasher: &mut Crc32Hasher) -> Result<u16> {
    let mut buf = [0u8; 2];
    read_and_hash(r, hasher, &mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32_and_hash(r: &mut impl Read, hasher: &mut Crc32Hasher) -> Result<u32> {
    let mut buf = [0u8; 4];
    read_and_hash(r, hasher, &mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_and_hash(r: &mut impl Read, hasher: &mut Crc32Hasher) -> Result<u64> {
    let mut buf = [0u8; 8];
    read_and_hash(r, hasher, &mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

// CRC32 lookup table (IEEE polynomial 0xEDB88320)
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = 0xEDB8_8320 ^ (crc >> 1);
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    use super::*;
    fn make_test_index() -> VkfIndex {
        VkfIndex {
            version: 1,
            codec_fourcc: u32::from_be_bytes(*b"avc1"),
            width: 640,
            height: 480,
            timescale: 30000,
            num_keyframes: 3,
            num_frames: 90,
            keyframes: vec![
                KeyframeEntry {
                    byte_offset: 1000,
                    byte_len: 5000,
                    pts: 0,
                    num_frames: 30,
                },
                KeyframeEntry {
                    byte_offset: 6000,
                    byte_len: 4500,
                    pts: 30000,
                    num_frames: 30,
                },
                KeyframeEntry {
                    byte_offset: 10500,
                    byte_len: 4000,
                    pts: 60000,
                    num_frames: 30,
                },
            ],
            frame_pts: (0..90).map(|i| i as u64 * 1000).collect(),
        }
    }

    #[test]
    fn test_vkf_roundtrip() {
        let index = make_test_index();
        let dir = std::env::temp_dir().join("veldt_test_vkf");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.vkf");

        index.write_to(&path).unwrap();
        let loaded = VkfIndex::read_from(&path).unwrap();

        assert_eq!(loaded.version, index.version);
        assert_eq!(loaded.codec_fourcc, index.codec_fourcc);
        assert_eq!(loaded.width, index.width);
        assert_eq!(loaded.height, index.height);
        assert_eq!(loaded.timescale, index.timescale);
        assert_eq!(loaded.num_keyframes, index.num_keyframes);
        assert_eq!(loaded.num_frames, index.num_frames);
        assert_eq!(loaded.keyframes.len(), index.keyframes.len());
        assert_eq!(loaded.frame_pts, index.frame_pts);

        for (a, b) in loaded.keyframes.iter().zip(index.keyframes.iter()) {
            assert_eq!(a.byte_offset, b.byte_offset);
            assert_eq!(a.byte_len, b.byte_len);
            assert_eq!(a.pts, b.pts);
            assert_eq!(a.num_frames, b.num_frames);
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_vkf_crc_corruption() {
        let index = make_test_index();
        let dir = std::env::temp_dir().join("veldt_test_vkf_corrupt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corrupt.vkf");

        index.write_to(&path).unwrap();

        // Corrupt one byte in the middle
        let mut data = std::fs::read(&path).unwrap();
        data[20] ^= 0xFF;
        std::fs::write(&path, &data).unwrap();

        let result = VkfIndex::read_from(&path);
        assert!(result.is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_keyframe_group_for_frame() {
        let index = make_test_index();

        // Group 0: frames 0-29
        assert_eq!(index.keyframe_group_for_frame(0), Some(0));
        assert_eq!(index.keyframe_group_for_frame(15), Some(0));
        assert_eq!(index.keyframe_group_for_frame(29), Some(0));

        // Group 1: frames 30-59
        assert_eq!(index.keyframe_group_for_frame(30), Some(1));
        assert_eq!(index.keyframe_group_for_frame(45), Some(1));

        // Group 2: frames 60-89
        assert_eq!(index.keyframe_group_for_frame(60), Some(2));
        assert_eq!(index.keyframe_group_for_frame(89), Some(2));

        // Out of range
        assert_eq!(index.keyframe_group_for_frame(90), None);
    }

    #[test]
    fn test_keyframe_first_frame() {
        let index = make_test_index();

        assert_eq!(index.keyframe_first_frame(0), 0);
        assert_eq!(index.keyframe_first_frame(1), 30);
        assert_eq!(index.keyframe_first_frame(2), 60);
    }

    #[test]
    fn test_sidecar_path() {
        assert_eq!(
            VkfIndex::sidecar_path(Path::new("/data/video.mp4")),
            Path::new("/data/video.vkf")
        );
    }

    #[test]
    fn test_crc32_known_value() {
        let mut hasher = Crc32Hasher::new();
        hasher.update(b"123456789");
        // Known CRC32 of "123456789" is 0xCBF43926
        assert_eq!(hasher.finalize(), 0xCBF4_3926);
    }
}
