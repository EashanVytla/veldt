# veldt

### A seek-optimized data loader for robot learning, built in Rust.

---

## Overview

veldt reads existing robot learning datasets (starting with LeRobot v3) and serves training batches to PyTorch at GPU speed — no format conversion, no frame extraction, no data duplication. A Belady-optimal prefetch scheduler uses the known epoch shuffle order to eliminate the P99 tail-latency stalls that starve GPUs on video-heavy datasets.

Drop-in Python API via PyO3. Tokio for async I/O, Rayon for parallel decode, DLPack for zero-copy tensor transfer.

**Modes** (in implementation order):

1. **Local** — Reads on-disk data faster than PyTorch DataLoader.
2. **Streaming + cache** — Trains while streaming; subsequent epochs run at local speed.
3. **Streaming only** — No local persistence, for storage-constrained environments.

---

## The Problem

Robot learning datasets pair MP4 video with tabular control data (Parquet). Training requires shuffled random access across episodes. Both dominant loading approaches break down on this access pattern.

### Local: PyTorch DataLoader + LeRobot

N worker processes each seek into MP4 files, decode frames via PyAV/ffmpeg, pickle-serialize tensors, and pass them to the main process. MP4 inter-frame compression (I/P/B frames) means random seeks must decode from the nearest keyframe forward — potentially 30+ wasted frames. Cold seeks are catastrophic.

Benchmarks on `lerobot/aloha_sim_insertion_human` (480×640, 50fps, 50 episodes, 4 workers):

| Metric | Value |
|---|---|
| Throughput | 71.3 samples/sec |
| P50 / P99 batch latency | 3.6 ms / 21,680 ms |
| Mean CPU | 5.0% |

The 6,000x P50→P99 spread confirms the bottleneck is I/O latency from cold keyframe seeks, not CPU decode. At batch level, 1% of batches already stall for 20+ seconds on this small dataset (~1.5 GB video) where much of the data fits in the OS page cache. On production datasets (DROID: 350 hours, OXE: 903M timesteps), the page cache covers a negligible fraction. The cold-seek rate per sample rises sharply, pushing both P50 and P99 batch latency upward.

Additional overhead: per-worker CPython interpreters with high memory cost, and the GIL forcing process-level parallelism instead of lightweight threads.

### Streaming: HuggingFace IterableDataset

Fetches Parquet rows and video segments over HTTP on demand.

Same dataset, streaming mode:

| Metric | Value |
|---|---|
| Throughput | 4.7 samples/sec |
| P50 / P99 batch latency | 13,195 ms / 19,845 ms |

15x slower than local, no parallelism. Parallelism is structurally unavailable: PyTorch `IterableDataset` assigns one worker per shard, and LeRobot v3 consolidates episodes into a single data Parquet per chunk (`num_shards=1`). Attempting `num_workers > 1`:

```
Too many dataloader workers: 3 (max is dataset.num_shards=1).
```

### Root Cause

MP4 has good random seek at the container level — the moov atom indexes exact byte offsets per keyframe. The problem is one layer below: H.264/AV1 keyframes (I-frames) occur every ~30 frames; all other frames (P/B) encode only diffs and can't be decoded independently. Seeking to an arbitrary frame means jumping to the nearest keyframe (fast), then decoding every frame forward to the target (wasteful). With keyframes every ~30 frames, ~97% of randomly sampled targets require decoding 10-25 throwaway frames first. Pre-extracting frames (WebDataset-style) eliminates this at the cost of 10-50x storage blowup.

---

## How veldt Solves This

### Key Insight

Training doesn't sample individual frames at random. Each sample needs a **temporal window** (~50 consecutive frames for action chunking). The true pattern is random-episode, sequential-window — and sequential reads are exactly what codecs optimize for. The expensive part is just the initial seek per window.

### Pipeline

```
PLAN → PREFETCH → DECODE → SLICE + TRANSFORM → DLPACK → GPU
```

**Plan (Belady scheduler).** At epoch start, veldt receives the full shuffle order and maps each sample to its required keyframe group (~30 frames from one keyframe to the next). This produces `(keyframe_group, first_needed_batch, last_needed_batch)` triples for the entire epoch. Because shuffle order is known in advance, Belady's optimal eviction is implementable — items are evicted immediately after their last consumer and prefetched just before their first.

**Prefetch (Tokio).** Async tasks read MP4 byte ranges ahead of training per the Belady schedule. Multiple episodes fetched concurrently. I/O overlaps with GPU computation.

**Decode (Rayon).** Compressed segments are decoded by a Rayon thread pool via `ffmpeg-next` (optional NVDEC). True thread-level parallelism — no GIL, no process forking. Decoded frames enter a bounded cache with Belady eviction.

**Slice + Transform.** Frame windows are sliced from cached keyframe groups. Resize and normalize run in Rust. Stochastic augmentation (random crop, color jitter) is left to user-side PyTorch on GPU. Cache sits post-decode/pre-transform, so augmentation stays stochastic across epochs.

**DLPack transfer.** Batched tensors go to PyTorch via `pyo3-dlpack` / `torch.from_dlpack()`. Zero-copy within a single process — no inter-process communication needed since Rust threads replace worker processes entirely. With NVDEC, frames decode directly to GPU memory.

**Tabular path.** Actions/states read from Parquet via `arrow-rs`, packed into tensors, transferred through the same DLPack path. Arrow is an internal detail; users see only `torch.Tensor`.

### Batch-Aware Seek Grouping

Samples in each batch are sorted by (episode, timestamp) before issuing seeks. Same-episode samples share keyframe groups, reducing seeks from ~32 to ~20 for a typical batch.

### Memory Behavior

Cache is bounded by concurrent working set, not dataset size. For 32 samples across ~20 episodes:

```
~40 keyframe groups × 30 frames × 921,600 bytes ≈ 1.1 GB
```

Belady eviction releases groups immediately after their last consumer. Zero additional storage beyond the original dataset.

---

## Operating Modes

### Mode 1: Local (MVP)

Dataset on local disk. Tokio handles async disk I/O, Rayon handles parallel decode, Belady drives eviction.

**Target:** P99 from 21.8s → sub-100ms. Throughput from 71 → 500+ samples/sec. GPU utilization 85-95%. This mode alone justifies the project — the bottleneck is structural, not a tuning problem.

### Mode 2: Streaming + Local Cache

First epoch streams compressed MP4 segments from HuggingFace Hub via async HTTP while training proceeds. Segments cached to disk as they arrive. Subsequent epochs run at local speed. Unlike a blocking download, training starts in seconds — critical for large datasets (DROID: 350 hours, OXE: 903M timesteps).

Cache stores original compressed byte ranges — no decompression, no inflation.

### Mode 3: Streaming Without Cache

Same as Mode 2 but segments are not persisted. Re-fetches each epoch. For shared clusters with limited SSD, quick experiments, or CI/CD pipelines.

**Cache strategy configuration:**

- `PostDecode` — Decoded frames in RAM. Stochastic augmentation each epoch. Default for training.
- `PostTransform` — Final tensors. Fastest, but freezes augmentation. For evaluation.
- `CompressedOnly` — Compressed segments on disk (Mode 2). Re-decode on access.
- `None` — No caching. Re-fetch and re-decode (Mode 3).

---

## Modular Format Support

veldt defines a trait-based reader interface rather than hard-coding formats:

```rust
pub trait DatasetReader: Send + Sync {
    fn len(&self) -> usize;
    async fn get(&self, index: usize) -> Result<Sample>;
    fn prefetch_hint(&self, indices: &[usize]) {}
}
```

Everything below the reader (scheduling, prefetch, decode, caching, batching, DLPack) is format-agnostic. The trait is also exposed as a Python protocol via PyO3 for users who don't write Rust.

**Built-in reader priority:**

1. LeRobot v3 (Parquet + MP4)
2. RLDS/TFRecord (Open X-Embodiment)
3. HDF5 (robomimic)
4. Zarr (Diffusion Policy)

---

## Python API

```python
import veldt

loader = veldt.Loader(
    path="lerobot/aloha_sim_insertion_human",
    batch_size=32,
    resize=(224, 224),
    normalize=True,
    num_decode_threads=8,      # Rayon pool size
    cache_budget_gb=2.0,       # Decoded frame cache bound
    mode="local",              # "local", "stream_cached", "stream"
)

for epoch in range(num_epochs):
    loader.set_epoch(epoch)    # Triggers Belady schedule rebuild
    for batch in loader:
        # batch.frames  → torch.Tensor [B, T, C, H, W] (via DLPack)
        # batch.actions → torch.Tensor [B, T, action_dim]
        # batch.state   → torch.Tensor [B, T, state_dim]

        frames = my_augmentation(batch.frames)  # User transforms on GPU
        loss = model(frames, batch.actions)
        loss.backward()
```

---

## Technology Stack

| Component | Crate / Tool | Role |
|---|---|---|
| Async I/O & prefetch | `tokio` | Disk reads, HTTP streaming, prefetch scheduling |
| Parallel decode | `rayon` | Multi-threaded keyframe group decoding |
| Video decode | `ffmpeg-next` | MP4/H.264/H.265/AV1 frame decoding |
| Hardware decode | NVDEC via `cudarc` | Optional GPU-accelerated video decode |
| Parquet reading | `arrow-rs` | Read tabular data (actions, states) |
| Python bridge | `pyo3` | Expose Rust API to Python |
| Tensor transfer | `pyo3-dlpack` | Zero-copy Rust → PyTorch via DLPack |
| Keyframe index | Custom | MP4 moov atom parsing, sidecar index file |

---

## Development Roadmap

### Phase 1: Local Mode (MVP)

- LeRobot v3 reader (Parquet + MP4)
- Keyframe index builder (moov atom parser)
- Belady epoch scheduler
- Rayon parallel decode pool
- DLPack → PyTorch transfer
- Basic Python API (`veldt.Loader`)
- Benchmark suite vs PyTorch DataLoader

### Phase 2: Streaming + Cache

- Tokio HTTP range-request fetcher
- Disk-backed compressed segment cache
- Epoch-0 hybrid mode (stream while training)
- Cache management (eviction, integrity)

### Phase 3: Streaming Without Cache

- Memory-only segment buffer
- Adaptive prefetch depth (bandwidth-aware)
- Graceful degradation under network pressure

### Phase 4: Ecosystem

- RLDS/TFRecord, HDF5, Zarr readers
- Python-defined reader protocol
- NVDEC hardware decode integration
- Multi-node distributed shard coordination

---

## Benchmarks

All data from `lerobot/aloha_sim_insertion_human` (50 episodes, 25K frames, 480×640, 50fps, AV1). Batch size 64, 100 batches, 10 warmup.

### Local (4 workers)

| Metric | Value |
|---|---|
| Total time (6,400 samples) | 89.7 s |
| Throughput | 71.3 samples/sec |
| P50 / P95 / P99 batch latency | 3.6 / 163 / 21,680 ms |
| Mean / Max CPU | 5.0% / 42.3% |

### Streaming (0 workers, buffer 1000)

| Metric | Value |
|---|---|
| Total time (6,400 samples) | 1,352.7 s |
| Throughput | 4.7 samples/sec |
| P50 / P95 / P99 batch latency | 13,195 / 17,315 / 19,845 ms |
| Mean / Max CPU | 5.8% / 67.1% |

### pusht (small synthetic, 96×96 images)

| Mode | Throughput | P99 Batch Latency |
|---|---|---|
| Local (4 workers) | 20.9 samples/sec | 22,116 ms |
| Streaming (0 workers) | 3,906 samples/sec | 17 ms |

Streaming wins on small datasets that fit in buffer. The relationship inverts on real video data.

---

## Design Decisions

**Why no new format?** Adoption. Nobody converts terabytes for a new tool. MP4 is already efficient storage — the problem is the reader.

**Why Belady over LRU?** LRU assumes temporal locality. Shuffled training has the opposite: recently-used frames won't recur until next epoch. Belady evicts optimally because we know the full shuffle order.

**Why cache post-decode, not post-transform?** Stochastic augmentation must differ each epoch. Caching post-transform freezes it. Exception: `PostTransform` for deterministic evaluation.

**Why DLPack over Arrow?** Arrow is for columnar data. DLPack is for dense N-d tensors, natively supported by PyTorch, and works on GPU memory. Arrow is used internally for Parquet; the Python API returns only `torch.Tensor` via DLPack.

**Why Tokio + Rayon?** Different bottlenecks. Tokio: async I/O and prefetch scheduling (latency hiding). Rayon: parallel CPU decode (throughput). Together they keep the GPU fed continuously.

---

## Target Audience

- Robot learning researchers training VLA models (PI0, Octo, OpenVLA, Diffusion Policy) on video datasets
- Teams scaling from sim (pusht) to real-world data (ALOHA, DROID, Bridge) and hitting data loading walls
- LeRobot users with low GPU utilization during training
- Rust developers looking to contribute to robotics ML# veldt
