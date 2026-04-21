use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use veldt_index::{parse_mp4, VkfIndex};

#[derive(Parser)]
#[command(name = "veldt", about = "Seek-optimized video data loader for ML training")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build VKF sidecar indices for all MP4 files in a dataset.
    Index {
        /// Path to the dataset root directory.
        #[arg()]
        path: PathBuf,
    },
    /// Verify VKF sidecars match their MP4 files.
    Verify {
        /// Path to the dataset root directory.
        #[arg()]
        path: PathBuf,
    },
    /// Print dataset info and recommended configuration.
    Info {
        /// Path to the dataset root directory.
        #[arg()]
        path: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Index { path } => cmd_index(&path),
        Commands::Verify { path } => cmd_verify(&path),
        Commands::Info { path } => cmd_info(&path),
    };

    if let Err(e) = result {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}

/// Walk a directory tree and find all .mp4 files.
fn find_mp4_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    walk_dir(root, &mut files);
    files.sort();
    files
}

fn walk_dir(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_dir(&path, files);
        } else if path.extension().map(|e| e == "mp4").unwrap_or(false) {
            files.push(path);
        }
    }
}

fn cmd_index(root: &Path) -> veldt_core::Result<()> {
    let mp4_files = find_mp4_files(root);
    if mp4_files.is_empty() {
        println!("No MP4 files found under {}", root.display());
        return Ok(());
    }

    println!("Found {} MP4 file(s), building VKF indices...", mp4_files.len());

    let mut indexed = 0;
    let mut skipped = 0;
    for mp4_path in &mp4_files {
        let vkf_path = VkfIndex::sidecar_path(mp4_path);
        if vkf_path.exists() {
            skipped += 1;
            continue;
        }

        match parse_mp4(mp4_path) {
            Ok(index) => {
                index.write_to(&vkf_path)?;
                println!(
                    "  {} -> {} ({} keyframes, {} frames)",
                    mp4_path.display(),
                    vkf_path.display(),
                    index.num_keyframes,
                    index.num_frames,
                );
                indexed += 1;
            }
            Err(e) => {
                eprintln!("  SKIP {}: {}", mp4_path.display(), e);
            }
        }
    }

    println!(
        "Done: {} indexed, {} already had sidecars",
        indexed, skipped
    );
    Ok(())
}

fn cmd_verify(root: &Path) -> veldt_core::Result<()> {
    let mp4_files = find_mp4_files(root);
    if mp4_files.is_empty() {
        println!("No MP4 files found under {}", root.display());
        return Ok(());
    }

    let mut ok_count = 0;
    let mut mismatch_count = 0;
    let mut missing_count = 0;

    for mp4_path in &mp4_files {
        let vkf_path = VkfIndex::sidecar_path(mp4_path);
        if !vkf_path.exists() {
            println!("  MISSING {}", vkf_path.display());
            missing_count += 1;
            continue;
        }

        let vkf = match VkfIndex::read_from(&vkf_path) {
            Ok(v) => v,
            Err(e) => {
                println!("  CORRUPT {}: {}", vkf_path.display(), e);
                mismatch_count += 1;
                continue;
            }
        };

        let mp4_index = match parse_mp4(mp4_path) {
            Ok(v) => v,
            Err(e) => {
                println!("  PARSE ERROR {}: {}", mp4_path.display(), e);
                mismatch_count += 1;
                continue;
            }
        };

        if vkf.num_frames != mp4_index.num_frames
            || vkf.num_keyframes != mp4_index.num_keyframes
            || vkf.codec_fourcc != mp4_index.codec_fourcc
        {
            println!(
                "  MISMATCH {}: vkf has {}/{} kf/frames, mp4 has {}/{}",
                mp4_path.display(),
                vkf.num_keyframes,
                vkf.num_frames,
                mp4_index.num_keyframes,
                mp4_index.num_frames,
            );
            mismatch_count += 1;
        } else {
            println!("  OK {}", mp4_path.display());
            ok_count += 1;
        }
    }

    println!(
        "Verified: {} ok, {} mismatch, {} missing",
        ok_count, mismatch_count, missing_count
    );

    if mismatch_count > 0 || missing_count > 0 {
        println!("Run `veldt index {}` to rebuild indices.", root.display());
    }

    Ok(())
}

fn cmd_info(root: &Path) -> veldt_core::Result<()> {
    let info_path = root.join("meta").join("info.json");
    if !info_path.exists() {
        println!("No meta/info.json found at {}", root.display());
        println!("This may not be a LeRobot v3 dataset.");
        return Ok(());
    }

    let info = veldt_reader_lerobot::InfoJson::deserialize_from(root)?;
    let meta = veldt_reader_lerobot::build_dataset_meta_from(root)?;

    println!("Dataset: {}", root.display());
    println!("  Format:     LeRobot {}", info.codebase_version);
    if let Some(ref robot) = info.robot_type {
        println!("  Robot:      {}", robot);
    }
    println!("  Episodes:   {}", info.total_episodes);
    println!("  Frames:     {}", info.total_frames);
    println!("  FPS:        {}", info.fps);
    println!("  Cameras:    {:?}", meta.camera_keys);
    println!("  Action dim: {}", meta.action_dim);
    println!("  State dim:  {}", meta.state_dim);

    // Count VKF files
    let mp4_files = find_mp4_files(root);
    let vkf_count = mp4_files
        .iter()
        .filter(|p| VkfIndex::sidecar_path(p).exists())
        .count();
    println!(
        "  Videos:     {} MP4 ({} indexed)",
        mp4_files.len(),
        vkf_count
    );

    if vkf_count < mp4_files.len() {
        println!("\n  Run `veldt index {}` to build missing sidecars.", root.display());
    }

    Ok(())
}
