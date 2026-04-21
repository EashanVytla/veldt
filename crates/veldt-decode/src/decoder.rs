extern crate ffmpeg_next as ffmpeg;

use ffmpeg::format::{input, Pixel};
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context as ScalerContext, flag::Flags};
use ffmpeg::util::frame::video::Video;

use veldt_core::{Result, VeldtError};

use crate::filter_graph::FilterSpec;

/// A request to decode a clip from a video file.
#[derive(Debug, Clone)]
pub struct DecodeRequest {
    /// Path to the MP4 file.
    pub path: std::path::PathBuf,
    /// First frame to extract (0-based).
    pub start_frame: u64,
    /// Number of frames to extract.
    pub num_frames: u32,
    /// Stride between extracted frames (1 = every frame).
    pub stride: u32,
    /// Filter spec (resize, normalize).
    pub filter: FilterSpec,
}

/// Decoded clip: contiguous frame data.
#[derive(Debug, Clone)]
pub struct DecodedClip {
    /// Raw pixel data. If normalized: f32 in [T, C, H, W] layout.
    /// If not normalized: u8 in [T, H, W, C] layout (RGB24).
    pub data: Vec<u8>,
    /// Shape of the data.
    pub shape: [usize; 4],
    /// Whether the data is f32 (normalized) or u8.
    pub is_float: bool,
}

/// Decode a clip from a video file using ffmpeg.
///
/// Opens the file, seeks to the region around `start_frame`, decodes frames,
/// applies resize via swscale, and returns the requested frames.
pub fn decode_clip(request: &DecodeRequest) -> Result<DecodedClip> {
    ffmpeg::init().map_err(|e| VeldtError::Decode(format!("ffmpeg init: {}", e)))?;

    let path_str = request.path.to_str().ok_or_else(|| {
        VeldtError::Decode(format!("invalid path: {:?}", request.path))
    })?;

    let mut ictx = input(&path_str)
        .map_err(|e| VeldtError::Decode(format!("open {}: {}", path_str, e)))?;

    // Find video stream
    let stream = ictx
        .streams()
        .best(Type::Video)
        .ok_or_else(|| VeldtError::Decode("no video stream".to_string()))?;

    let video_stream_index = stream.index();
    let time_base = stream.time_base();

    // Set up decoder
    let context_decoder =
        ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| VeldtError::Decode(format!("decoder context: {}", e)))?;
    let mut decoder = context_decoder
        .decoder()
        .video()
        .map_err(|e| VeldtError::Decode(format!("video decoder: {}", e)))?;

    // Set up scaler: source format -> RGB24 at target resolution
    let mut scaler = ScalerContext::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        Pixel::RGB24,
        request.filter.width,
        request.filter.height,
        Flags::BILINEAR,
    )
    .map_err(|e| VeldtError::Decode(format!("scaler: {}", e)))?;

    // Seek to just before the target frame
    // We seek by timestamp. Estimate PTS from frame index using stream time_base.
    if request.start_frame > 0 {
        let fps = estimate_fps(&ictx, video_stream_index);
        if fps > 0.0 {
            let target_ts = (request.start_frame as f64 / fps
                * time_base.1 as f64
                / time_base.0 as f64) as i64;
            // Seek backward to nearest keyframe
            ictx.seek(target_ts, ..target_ts)
                .map_err(|e| VeldtError::Decode(format!("seek: {}", e)))?;
        }
    }

    // Decode frames
    let target_start = request.start_frame as usize;
    let stride = request.stride.max(1) as usize;
    let target_count = request.num_frames as usize;
    let target_end = target_start + (target_count - 1) * stride + 1;

    let w = request.filter.width as usize;
    let h = request.filter.height as usize;

    let mut collected_frames: Vec<Vec<u8>> = Vec::with_capacity(target_count);
    let mut frame_counter: usize = 0;
    let mut decoded_frame = Video::empty();
    let mut rgb_frame = Video::empty();

    // Process packets
    let mut done = false;
    for (stream, packet) in ictx.packets() {
        if done {
            break;
        }
        if stream.index() != video_stream_index {
            continue;
        }

        decoder
            .send_packet(&packet)
            .map_err(|e| VeldtError::Decode(format!("send_packet: {}", e)))?;

        while decoder.receive_frame(&mut decoded_frame).is_ok() {
            if frame_counter >= target_start && frame_counter < target_end {
                if (frame_counter - target_start) % stride == 0 {
                    scaler
                        .run(&decoded_frame, &mut rgb_frame)
                        .map_err(|e| VeldtError::Decode(format!("scale: {}", e)))?;

                    // Copy RGB data
                    let data = rgb_frame.data(0);
                    let expected_size = h * w * 3;
                    let row_bytes = rgb_frame.stride(0) as usize;

                    if row_bytes == w * 3 {
                        collected_frames.push(data[..expected_size].to_vec());
                    } else {
                        // Handle stride padding
                        let mut frame_data = Vec::with_capacity(expected_size);
                        for row in 0..h {
                            let start = row * row_bytes;
                            frame_data.extend_from_slice(&data[start..start + w * 3]);
                        }
                        collected_frames.push(frame_data);
                    }
                }
            }

            frame_counter += 1;
            if collected_frames.len() >= target_count {
                done = true;
                break;
            }
        }
    }

    // Flush decoder
    if !done {
        decoder
            .send_eof()
            .map_err(|e| VeldtError::Decode(format!("send_eof: {}", e)))?;

        while decoder.receive_frame(&mut decoded_frame).is_ok() {
            if frame_counter >= target_start && frame_counter < target_end {
                if (frame_counter - target_start) % stride == 0 {
                    scaler
                        .run(&decoded_frame, &mut rgb_frame)
                        .map_err(|e| VeldtError::Decode(format!("scale: {}", e)))?;

                    let data = rgb_frame.data(0);
                    let expected_size = h * w * 3;
                    let row_bytes = rgb_frame.stride(0) as usize;

                    if row_bytes == w * 3 {
                        collected_frames.push(data[..expected_size].to_vec());
                    } else {
                        let mut frame_data = Vec::with_capacity(expected_size);
                        for row in 0..h {
                            let start = row * row_bytes;
                            frame_data.extend_from_slice(&data[start..start + w * 3]);
                        }
                        collected_frames.push(frame_data);
                    }
                }
            }
            frame_counter += 1;
            if collected_frames.len() >= target_count {
                break;
            }
        }
    }

    let actual_count = collected_frames.len();
    if actual_count == 0 {
        return Err(VeldtError::Decode(format!(
            "no frames decoded (wanted {} starting at frame {})",
            target_count, target_start
        )));
    }

    // Assemble into contiguous buffer
    if request.filter.normalize {
        // Convert u8 RGB [T, H, W, 3] -> f32 CHW [T, 3, H, W] with normalization
        let num_pixels = actual_count * 3 * h * w;
        let mut float_data = vec![0.0f32; num_pixels];

        for (t, frame) in collected_frames.iter().enumerate() {
            for y in 0..h {
                for x in 0..w {
                    let src_idx = y * w * 3 + x * 3;
                    for c in 0..3 {
                        let val = frame[src_idx + c] as f32 / 255.0;
                        let normalized =
                            (val - request.filter.mean[c]) / request.filter.std[c];
                        // Target layout: [T, C, H, W]
                        let dst_idx = t * (3 * h * w) + c * (h * w) + y * w + x;
                        float_data[dst_idx] = normalized;
                    }
                }
            }
        }

        let byte_data = unsafe {
            let ptr = float_data.as_ptr() as *const u8;
            let len = float_data.len() * std::mem::size_of::<f32>();
            std::slice::from_raw_parts(ptr, len).to_vec()
        };

        Ok(DecodedClip {
            data: byte_data,
            shape: [actual_count, 3, h, w],
            is_float: true,
        })
    } else {
        // u8 layout: [T, H, W, C]
        let total_size = actual_count * h * w * 3;
        let mut contiguous = Vec::with_capacity(total_size);
        for frame in &collected_frames {
            contiguous.extend_from_slice(frame);
        }

        Ok(DecodedClip {
            data: contiguous,
            shape: [actual_count, h, w, 3],
            is_float: false,
        })
    }
}

/// Estimate FPS from the stream metadata.
fn estimate_fps(ictx: &ffmpeg::format::context::Input, stream_index: usize) -> f64 {
    let stream = ictx.stream(stream_index).unwrap();
    let rate = stream.avg_frame_rate();
    if rate.1 > 0 {
        rate.0 as f64 / rate.1 as f64
    } else {
        let tb = stream.time_base();
        if tb.0 > 0 {
            tb.1 as f64 / tb.0 as f64
        } else {
            30.0 // fallback
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_path() -> std::path::PathBuf {
        let manifest = env!("CARGO_MANIFEST_DIR");
        std::path::PathBuf::from(manifest)
            .parent().unwrap()  // crates/
            .parent().unwrap()  // veldt/
            .join("tests/fixtures/mini_lerobot/test_video.mp4")
    }

    #[test]
    fn test_filter_spec_default() {
        let spec = FilterSpec::default();
        assert_eq!(spec.width, 224);
        assert_eq!(spec.height, 224);
        assert!(!spec.normalize);
    }

    #[test]
    fn test_decode_first_frames() {
        let path = fixture_path();
        if !path.exists() {
            eprintln!("skipping test: fixture not found at {:?}", path);
            return;
        }

        let request = DecodeRequest {
            path: path.clone(),
            start_frame: 0,
            num_frames: 10,
            stride: 1,
            filter: FilterSpec {
                width: 48,
                height: 48,
                normalize: false,
                ..Default::default()
            },
        };

        let clip = decode_clip(&request).unwrap();

        assert_eq!(clip.shape[0], 10); // 10 frames
        assert_eq!(clip.shape[1], 48); // H
        assert_eq!(clip.shape[2], 48); // W
        assert_eq!(clip.shape[3], 3);  // C (RGB)
        assert!(!clip.is_float);
        assert_eq!(clip.data.len(), 10 * 48 * 48 * 3);
    }

    #[test]
    fn test_decode_with_seek() {
        let path = fixture_path();
        if !path.exists() {
            return;
        }

        let request = DecodeRequest {
            path,
            start_frame: 40,
            num_frames: 5,
            stride: 1,
            filter: FilterSpec {
                width: 32,
                height: 32,
                normalize: false,
                ..Default::default()
            },
        };

        let clip = decode_clip(&request).unwrap();
        assert_eq!(clip.shape[0], 5);
        assert_eq!(clip.shape[1], 32);
        assert_eq!(clip.shape[2], 32);
    }

    #[test]
    fn test_decode_with_stride() {
        let path = fixture_path();
        if !path.exists() {
            return;
        }

        let request = DecodeRequest {
            path,
            start_frame: 0,
            num_frames: 5,
            stride: 3, // every 3rd frame
            filter: FilterSpec {
                width: 32,
                height: 32,
                normalize: false,
                ..Default::default()
            },
        };

        let clip = decode_clip(&request).unwrap();
        assert_eq!(clip.shape[0], 5); // 5 frames extracted
    }

    #[test]
    fn test_decode_normalized() {
        let path = fixture_path();
        if !path.exists() {
            return;
        }

        let request = DecodeRequest {
            path,
            start_frame: 0,
            num_frames: 3,
            stride: 1,
            filter: FilterSpec {
                width: 32,
                height: 32,
                normalize: true,
                mean: [0.485, 0.456, 0.406],
                std: [0.229, 0.224, 0.225],
            },
        };

        let clip = decode_clip(&request).unwrap();
        assert!(clip.is_float);
        assert_eq!(clip.shape, [3, 3, 32, 32]); // [T, C, H, W]

        // f32 data: 3 * 3 * 32 * 32 * 4 bytes
        assert_eq!(clip.data.len(), 3 * 3 * 32 * 32 * 4);

        // Verify values are in reasonable normalized range
        let float_data: &[f32] = unsafe {
            std::slice::from_raw_parts(
                clip.data.as_ptr() as *const f32,
                clip.data.len() / 4,
            )
        };
        for &val in float_data {
            assert!(val > -10.0 && val < 10.0, "normalized value out of range: {}", val);
        }
    }
}
