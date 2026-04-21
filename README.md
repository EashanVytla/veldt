# veldt

### A seek-optimized data loader for video-heavy ML training, built in Rust.

---

## Overview

veldt reads existing datasets (starting with LeRobot v3) and serves training batches to PyTorch at GPU speed — no format conversion, no frame extraction, no data duplication. An epoch-aware, Belady-optimal scheduler uses the known shuffle order to eliminate the P99 tail-latency stalls that starve GPUs on video-heavy datasets.

Drop-in Python API via PyO3. Tokio for async I/O through OpenDAL (local + S3 + GCS + HF Hub behind one interface), Rayon for parallel decode, ffmpeg filter graphs for fused decode-and-transform, DLPack for zero-copy tensor transfer.

**Modes** (in implementation order):

1. **Local** — Reads on-disk data faster than PyTorch DataLoader.
2. **Streaming + cache** — Trains while streaming; subsequent epochs run at local speed.
3. **Streaming only** — No local persistence, for storage-constrained environments.

**Scope.** Built first for robot learning (LeRobot, PI0, ALOHA, DROID). The pipeline is format-agnostic; video generation training (Open-Sora, HunyuanVideo) has the same underlying access pattern — random clip sampling over compressed video — and is a first-class Phase 4 target.

---

## The Problem

Video-heavy training pairs MP4 video with tabular control or label data (Parquet). Training requires shuffled random access. Both dominant loading approaches break down on this access pattern.

### Local: PyTorch DataLoader + LeRobot

N worker processes each seek into MP4 files, decode frames via torchcodec/PyAV/ffmpeg, pickle-serialize tensors, and pass them to the main process. MP4 inter-frame compression (I/P/B frames) means random seeks must decode from the nearest keyframe forward — potentially 30+ wasted frames. Cold seeks are catastrophic.

Benchmarks on `lerobot/aloha_sim_insertion_human` (480×640, 50fps, 50 episodes, 4 workers):

| Metric | Value |
|---|---|
| Throughput | 71.3 samples/sec |
| P50 / P99 batch latency | 3.6 ms / 21,680 ms |
| Mean CPU | 5.0% (workers idle, waiting on I/O — not CPU-bound) |

The 6,000× P50→P99 spread confirms the bottleneck is I/O latency from cold keyframe seeks. At batch level, 1% of batches already stall for 20+ seconds on this small dataset (~1.5 GB video) where much of the data fits in the OS page cache. On production datasets (DROID: 8.7 TB, OXE: 903M timesteps), the page cache covers a negligible fraction. The cold-seek rate per sample rises sharply, pushing both P50 and P99 upward.

### Streaming: HuggingFace IterableDataset

Fetches Parquet rows and video segments over HTTP on demand.

Same dataset, streaming mode:

| Metric | Value |
|---|---|
| Throughput | 4.7 samples/sec |
| P50 / P99 batch latency | 13,195 ms / 19,845 ms |

15× slower than local. The root cause is a chain of primitive-level decisions in the HF / LeRobot streaming path:

| # | What it does | Why it's slow |
|---|---|---|
| 1 | `IterableDataset` with `num_shards=1` (v3 consolidates episodes per chunk) | PyTorch forces one worker per shard → single-threaded |
| 2 | Buffer-shuffle draws from first N examples, replaces as consumed | **Shuffle order is not known at epoch start** — blocks any ahead-of-time planning |
| 3 | Synchronous HTTP fetch per sample via HF Hub | Each sample blocks on network RTT; no prefetch |
| 4 | Per-sample seek via torchcodec/PyAV | Each call re-validates decoder state; cold seeks hit the keyframe scan |
| 5 | `set_format(type='torch')` path broken (LeRobot #1282, closed "not planned") | Python-object deepcopy per sample defeats the torch-tensor fast path |
| 6 | Filename resolution + redirects through HF Hub | Per-sample DNS + redirect + TLS handshake unless connection is held |
| 7 | No inter-sample seek grouping | Random 32 samples = 32 independent HTTP range requests |

Attempting `num_workers > 1`:

```
Too many dataloader workers: 3 (max is dataset.num_shards=1).
```

Row 2 is the architectural lynchpin: buffer-shuffle silently prevents any epoch-level scheduling. Until the full shuffle order is knowable at epoch start, Belady-style planning is impossible.

### Root Cause

MP4 has good random seek at the container level — the moov atom indexes exact byte offsets per keyframe. The problem is one layer below: H.264/AV1 keyframes (I-frames) occur every ~30 frames; all other frames (P/B) encode only diffs and can't be decoded independently. Seeking to an arbitrary frame means jumping to the nearest keyframe (fast), then decoding every frame forward to the target (wasteful). With keyframes every ~30 frames, ~97% of randomly sampled targets require decoding 10-25 throwaway frames first. Pre-extracting frames (WebDataset-style) eliminates this at the cost of 10-50× storage blowup.

---

## Why Now

This isn't a tuning problem reported in isolation — it's an ecosystem-wide signal that's being treated as codec or hardware limitation.

- **huggingface/lerobot#1623** — SmolVLA training dataloader wait (~1s) exceeds backprop time (~0.7s) on a many-core server. AV1 decode measured 5× slower than H.264. Reporter abandoned video datasets for pre-extracted images despite the storage cost.
- **huggingface/lerobot#2282** — 2× regression between 0.3.3 and 0.3.4 traced to the dataloading path. GPU utilization oscillates 0–100% — canonical data starvation.
- **LeRobot v3 defaults to AV1** (better compression, much slower decode). The tax is paid by every new v3 dataset.
- **torchcodec** (PyTorch's own decoder) lists "approximate seeking mode" as its top-priority fix because seek accuracy requires an initial linear scan. veldt's keyframe index *is* approximate seeking, available today. On CPU random access, torchcodec is 3.3× slower than decord (meta-pytorch/torchcodec#426).

The common prescribed fixes are "use more cores" or "switch decoder." Neither addresses the structural issue: no existing loader plans across samples and across videos using known epoch-level ordering.

---

## How veldt Solves This

### Key Insight

Training doesn't sample individual frames at random. Each sample needs a **clip** — a range of consecutive frames (~50 for robot action chunking, 16–300 for video generation). Sequential reads within a clip are exactly what codecs optimize for. The expensive part is just the initial seek.

### Pipeline

```
PLAN → PREFETCH → DECODE (fused with filter graph) → DLPACK → GPU
```

**Plan (Belady scheduler).** At epoch start, veldt receives the full shuffle order and asks each reader to translate sample indices into a format-aware `FetchPlan` (for MP4: keyframe groups). The scheduler then produces `(fetch_unit, first_needed_batch, last_needed_batch)` triples for the entire epoch. Because the shuffle order is known in advance, Belady's optimal eviction is implementable — items are evicted immediately after their last consumer and prefetched just before their first.

**Prefetch (Tokio + OpenDAL).** Async tasks issue byte-range reads ahead of training per the Belady schedule. One code path handles local disk, S3, GCS, Azure, and HF Hub — OpenDAL is the backend. Multiple episodes fetch concurrently. I/O overlaps with GPU computation.

**Decode (Rayon + ffmpeg filter graph).** Compressed segments decode in a Rayon thread pool via `ffmpeg-next`, optionally through NVDEC. Resize, color conversion, and normalization fuse into the same ffmpeg filter graph — on the NVDEC path, output lands in CUDA memory directly with no host→device copy. Stochastic augmentation (random crop, color jitter) is deferred to user-side PyTorch on GPU, so the cache sits pre-augmentation.

**DLPack transfer.** Batched tensors go to PyTorch via `pyo3-dlpack` / `torch.from_dlpack()`. Single-process, zero-copy.

**Tabular path.** Actions/states read from Parquet via `arrow-rs`, packed into tensors, transferred through the same DLPack path.

### Architecture: stateless server + stateful client

"Server" in veldt means **precomputed sidecar index files** sitting alongside the source media — not a daemon. Indices record keyframe byte offsets, PTS→frame maps, codec metadata, and optional norm stats. `veldt index <path>` produces them once; every subsequent trainer reads them without coordination. S3/GCS/HF Hub are already the HTTP range-read server.

The stateful layer lives on the client:

| Layer | Where | Lifetime |
|---|---|---|
| Sidecar indices (`*.vkf`) | Alongside source media (local or remote bucket) | Regenerated on dataset version bump |
| Compressed segment cache | Client disk (Mode 2) | LRU / TTL per trainer |
| Decoded frame cache | Client RAM | Epoch-scoped, Belady-evicted |

Missing sidecars trigger an inline build with a warning, so users never hit a hard error from a cold dataset.

### Batch-Aware Seek Grouping

Samples in each batch are sorted by (episode, timestamp) before issuing seeks. Same-episode samples share keyframe groups, reducing seeks from ~32 to ~20 for a typical batch.

### Memory Behavior

Cache is bounded by concurrent working set, not dataset size. For 32 samples across ~20 episodes:

```
~40 keyframe groups × 30 frames × 921,600 bytes ≈ 1.1 GB
```

Belady eviction releases groups immediately after their last consumer. No additional on-disk storage in Mode 1. Mode 2 stores compressed byte ranges only, bounded by an explicit budget.

---

## Operating Modes

### Mode 1: Local (MVP)

Dataset on local disk. Tokio + OpenDAL handle async I/O; Rayon handles parallel decode; Belady drives eviction.

**Target:** P99 from 21.8s → sub-100ms. Throughput from 71 → 500+ samples/sec. GPU utilization 85-95%. This mode alone justifies the project — the bottleneck is structural, not a tuning problem.

### Mode 2: Streaming + Local Cache

First epoch streams compressed MP4 segments over HTTP range reads while training proceeds. Segments cache to disk as they arrive. Subsequent epochs run at local speed. Unlike a blocking download, training starts in seconds — critical for DROID (8.7 TB) and OXE-scale datasets.

Cache stores original compressed byte ranges — no decompression, no inflation.

### Mode 3: Streaming Without Cache

Same as Mode 2 but segments are not persisted. Re-fetches each epoch. For shared clusters with limited SSD, quick experiments, or CI/CD pipelines.

### Cache Strategy

Three independent knobs — one per cache tier, not one aggregate budget:

| Knob | Default | Controls |
|---|---|---|
| `compressed_cache_gb` | 0 (Mode 1) / 10 (Mode 2) | Disk-backed compressed segments |
| `decoded_cache_gb` | 2 | RAM decoded-frame cache (Belady-evicted) |
| `prefetch_depth` | 64 | Belady lookahead in samples |

Caching mode selects the eviction policy:

- `PostDecode` — Decoded frames in RAM. Stochastic augmentation each epoch. Default for training.
- `PostTransform` — Final tensors. Fastest, but freezes augmentation. For evaluation.
- `CompressedOnly` — Compressed segments on disk (Mode 2). Re-decode on access.
- `None` — No caching. Re-fetch and re-decode (Mode 3).

---

## Format Tricks

Each supported format exposes a native primitive; the reader's job is to exploit it. The `FetchPlan` the Belady scheduler operates on is format-aware at the reader level and opaque above it.

| Format | Primitive | Cold-access cost | veldt's move |
|---|---|---|---|
| MP4 H.264/HEVC | moov atom → byte offset; decode from nearest I-frame | Decode 10-25 P/B frames before target | Sidecar keyframe index; batch seeks by GOP; precomputed PTS map |
| MP4 AV1 | Same primitives, ~5× slower decode | Very expensive | Hardware decode mandatory; longer GOP tolerance; warn if no NVDEC/VideoToolbox |
| WebM / VP9 | SeekHead + cluster index | One cluster read | Cluster-aligned batching |
| Parquet | Row groups + column chunks + column statistics | Whole row group materialized | Column projection (drop unused cameras); row-group-aware prefetch |
| Arrow IPC | Memory-mapped random access | None if resident | mmap directly; no intermediate cache |
| HDF5 | Chunk index + per-chunk compression | Chunk decompress | Chunk-aligned access; parallel chunk fetch |
| Zarr | One chunk per file + consolidated metadata | Per-chunk HTTP GET | Async chunk fetching; consolidated-metadata read at open |
| RLDS / TFRecord | Sequential + sidecar index | Full record read | Build index at convert time; cache shard order |
| MCAP (Foxglove) | Chunk Index + Message Index | One chunk read | Message-index sub-chunk precision; map topics → tensor columns |
| WebDataset (tar) | Sequential within shard | Whole shard on random | Shuffle shards, not samples; prefetch 1-2 shards |

---

## Modular Format Support

veldt defines a trait-based reader interface rather than hard-coding formats:

```rust
pub trait DatasetReader: Send + Sync {
    fn len(&self) -> usize;

    // Return a specific clip range, not just a sample index.
    async fn get(&self, index: usize, spec: ClipSpec) -> Result<Sample>;

    // Epoch-start planning: reader decides how to group primitive fetches.
    // FetchPlan is format-aware at the reader; opaque to the scheduler.
    fn plan_fetch(&self, requests: &[(usize, ClipSpec)]) -> FetchPlan;

    // Optional: open persistent connections, madvise mmaps, etc.
    fn prefetch_hint(&self, plan: &FetchPlan) {}
}

pub struct ClipSpec {
    pub start_frame: u64,
    pub num_frames: u32,      // variable-length for video gen; fixed for robot learning
    pub stride: u32,
    pub resolution: (u32, u32),
}
```

Everything above the reader — Belady scheduling, prefetch, decoding, caching, batching, DLPack — is format-agnostic. The trait is exposed as a Python protocol via PyO3 for users who don't write Rust.

**Built-in reader priority:**

1. LeRobot v3 (Parquet + MP4)
2. RLDS/TFRecord (Open X-Embodiment)
3. HDF5 (robomimic)
4. Zarr (Diffusion Policy)
5. MCAP (Foxglove) — robotics multimodal logs
6. CSV + MP4 clips (Open-Sora, HunyuanVideo, video generation)

---

## Python API

```python
import veldt

loader = veldt.Loader(
    path="lerobot/aloha_sim_insertion_human",
    batch_size=32,
    resize=(224, 224),
    normalize=True,
    num_decode_threads=None,       # None → num_physical_cores
    compressed_cache_gb=10,        # Mode 2 disk-backed segments
    decoded_cache_gb=2,            # RAM decoded-frame cache
    prefetch_depth=64,             # Belady lookahead in samples
    mode="local",                  # "local", "stream_cached", "stream"
)

for epoch in range(num_epochs):
    loader.set_epoch(epoch)        # Triggers Belady schedule rebuild
    for batch in loader:
        # batch.frames  → torch.Tensor [B, T, C, H, W] (via DLPack)
        # batch.actions → torch.Tensor [B, T, action_dim]
        # batch.state   → torch.Tensor [B, T, state_dim]

        frames = my_augmentation(batch.frames)  # User transforms on GPU
        loss = model(frames, batch.actions)
        loss.backward()
```

### Ergonomics

- **Format auto-detection.** `veldt.Loader(path=...)` inspects the path and picks the reader; no explicit `reader=` argument unless overriding.
- **Dataset presets.** `veldt.Loader.from_preset("aloha" | "droid" | "pusht" | "libero")` sets codec, camera names, normalization stats, clip length.
- **Observability.** `loader.stats()` returns cache hit rates, queue depths, decode fps, GPU-idle fraction.
- **Calibration.** `loader.calibrate()` runs a short probe at init and tunes decode threads + prefetch depth.
- **Explain mode.** `loader.explain(epoch=0)` prints the Belady schedule as human-readable pseudo-code.
- **Graceful fallback.** Missing keyframe index, unsupported codec, or bad timestamps → slow-path (torchcodec) with a one-line warning. Never a hard error.
- **Env overrides.** Every knob has a `VELDT_*` equivalent for script-based sweeps.
- **Checkpoint/resume.** `loader.state_dict() / load_state_dict()` matching `IterableDataset` semantics.
- **PyTorch `Dataset` conformance.** `veldt.Dataset` wraps the reader as a `torch.utils.data.Dataset` for training loops that can't swap the outer DataLoader.

### CLI

- `veldt index <path>` — precompute sidecar indices (keyframe offsets, PTS maps).
- `veldt bench <path>` — probe a dataset and print a recommended config block.
- `veldt verify <path>` — check sidecar validity, flag corrupt media.

---

## Technology Stack

| Component | Crate / Tool | Role |
|---|---|---|
| Unified I/O | `opendal` (+ `tokio` runtime) | Local disk, S3, GCS, Azure, HF Hub behind one interface |
| Parallel decode | `rayon` | Multi-threaded keyframe group decoding |
| Video decode + transform | `ffmpeg-next` | MP4/H.264/H.265/AV1 decode fused with filter graph (scale, format, normalize) |
| Hardware decode | NVDEC via `cudarc` | Optional GPU-accelerated decode; output in CUDA memory |
| Parquet reading | `arrow-rs` | Tabular data (actions, states) |
| Python bridge | `pyo3` | Expose Rust API to Python |
| Tensor transfer | `pyo3-dlpack` | Zero-copy Rust → PyTorch via DLPack |
| Sidecar index | Custom (`*.vkf`) | Keyframe byte offsets, PTS map, codec metadata |

**OpenDAL gotcha.** The default `RangeReader` discards its internal stream on every `seek`, triggering a fresh S3 GET. Wrap with a BufferReader (the pattern contributed back by Greptime) so consecutive reads within a keyframe group reuse one HTTP connection.

---

## Development Roadmap

### Phase 1: Local Mode (MVP)

- LeRobot v3 reader (Parquet + MP4)
- Keyframe index builder (moov atom parser) + `veldt index` CLI
- Belady epoch scheduler
- Rayon parallel decode pool with ffmpeg filter graph fusion
- DLPack → PyTorch transfer
- Python API with three-tier cache budgets
- Benchmark suite vs PyTorch DataLoader + torchcodec (CPU and CUDA)

### Phase 1.5: PI0 Validation

- Integrate veldt into `openpi` / `openpi_pytorch` training loop
- Run PI0 fine-tuning on LIBERO end-to-end
- Report GPU utilization and wall-clock delta vs default LeRobot loader
- Land as `examples/openpi_pi0_libero.py`

### Phase 2: Streaming + Cache

- OpenDAL-backed HTTP range fetcher with BufferReader wrapper
- Disk-backed compressed segment cache
- Epoch-0 hybrid mode (stream while training)
- Cache management (eviction, integrity)

### Phase 3: Streaming Without Cache

- Memory-only segment buffer
- Adaptive prefetch depth (bandwidth-aware)
- Graceful degradation under network pressure

### Phase 4: Ecosystem

- RLDS/TFRecord, HDF5, Zarr, MCAP readers
- Open-Sora / HunyuanVideo reader (CSV + MP4 with bucket batching)
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
| Mean / Max CPU | 5.0% / 42.3% (workers idle on I/O) |

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

### Precompute Once, Train Many

All veldt numbers assume `veldt index` has been run once against the dataset. The index is a few hundred KB per episode, lives alongside the source media, and is shared across every trainer hitting that bucket. Cold indexing takes seconds for a typical LeRobot dataset — and it's the only piece of work that has to happen before training can start.

### Stress-Test Matrix

Integration with external repos is an exit criterion for Phase 4:

| Repo | Format | Purpose |
|---|---|---|
| `Physical-Intelligence/openpi` | LeRobot v3 | PI0 flagship (Phase 1.5) |
| `huggingface/lerobot` SmolVLA | LeRobot v3 | Reproduce and fix the #1623 bottleneck |
| `NVlabs/diffusion_policy` | Zarr | Non-video; tests reader abstraction |
| `hpcaitech/Open-Sora` | CSV + MP4 clips | Video generation, variable-length clips |
| `Tencent/HunyuanVideo` | CSV + MP4 clips | Video generation at scale |
| `ARISE-Initiative/robomimic` | HDF5 | Non-MP4; tests format breadth |

---

## Related Work

- **torchcodec** (Meta PyTorch) — the current default video decoder inside LeRobot. Releases the GIL, supports CUDA decode. Operates per-`VideoDecoder`; no cross-video or epoch-level coordination. veldt's edge is the scheduler, not the decoder.
- **Vidformer** (Dominik Winecki, OSU) — Rust + ffmpeg + OpenDAL stack for interactive annotation and Video-on-Demand rendering. Solves the "instant first-frame of an edit script" problem. Adjacent stack, different goal (rendering vs training). veldt may reuse Vidformer's SIR / filter-graph engine in Phase 4 rather than reinventing.
- **NVIDIA DALI** — GPU-accelerated preprocessing for deep learning. Broad support, but no epoch-aware planning and no Belady-style eviction.
- **WebDataset** — tar-based sequential access. Sidesteps the seek problem by forcing sequential reads, at the cost of shuffle quality and 10-50× storage inflation.
- **MCAP / Foxglove** — indexed container format for robotics logs, with native chunk and message indices. Complementary format; veldt treats it as a first-class reader target.

---

## Design Decisions

**Why no new format?** Adoption. Nobody converts terabytes for a new tool. MP4 is already efficient storage — the problem is the reader.

**Why Belady over LRU?** LRU assumes temporal locality. Shuffled training has the opposite: recently-used frames won't recur until next epoch. Belady evicts optimally because we know the full shuffle order.

**Why cache post-decode, not post-transform?** Stochastic augmentation must differ each epoch. Caching post-transform freezes it. Exception: `PostTransform` for deterministic evaluation.

**Why DLPack over Arrow?** Arrow is for columnar data. DLPack is for dense N-d tensors, natively supported by PyTorch, and works on GPU memory. Arrow is used internally for Parquet; the Python API returns only `torch.Tensor` via DLPack.

**Why Tokio + Rayon + OpenDAL?** Tokio: async I/O and prefetch scheduling (latency hiding). Rayon: parallel CPU decode (throughput). OpenDAL: one code path for local disk and object storage. Together they keep the GPU fed continuously.

**Why fuse resize/normalize into the ffmpeg filter graph?** Avoids one CPU→GPU copy per sample on NVDEC paths; avoids extra memcpy on CPU paths. The transform spec compiles to the same filter graph regardless of backend.

**Why stateless server?** S3/GCS already do the server job. Running a daemon in front adds ops burden without improving range-read performance. Sidecar files in the bucket give every trainer the shared work without coordination.

**Why not lead with "no GIL, no process forking"?** torchcodec also releases the GIL. The real edge is cross-video, cross-batch, epoch-aware coordination — the scheduler, not the decoder.

---

## Target Audience

- Robot learning researchers training VLA models (PI0, Octo, OpenVLA, Diffusion Policy) on video datasets
- Video generation teams (Open-Sora, HunyuanVideo-class) hitting CPU decode bottlenecks on petabyte-scale data
- Teams scaling from sim (pusht) to real-world data (ALOHA, DROID, Bridge) and hitting data loading walls
- LeRobot users with low GPU utilization during training
- Rust developers looking to contribute to ML systems