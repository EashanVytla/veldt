# Benchmarks

This directory contains two small benchmarks for measuring dataset input performance in the LeRobot data pipeline:

- `bench_local.py`: benchmarks `LeRobotDataset` loaded through a standard PyTorch `DataLoader`. This is the local or cached path.
- `bench_streaming.py`: benchmarks `StreamingLeRobotDataset` streamed from the Hugging Face Hub. This is the streaming path.
- `run_compare.py`: runs the local benchmark first, then the streaming benchmark, writes both results to JSON, and generates comparison plots.

Both benchmarks:

- load a dataset split defined by `--repo-id` and `--episodes`
- run a short warmup
- measure per-batch timing over a fixed number of batches
- report throughput, latency, CPU usage, and how often the loop is waiting on data

## What Each Benchmark Measures

### `bench_local.py`

Use this to measure the performance of the local/cached pipeline, including:

- dataset access through `LeRobotDataset`
- PyTorch `DataLoader` batching
- multi-worker loading and prefetch behavior

This is the benchmark to use when you want to evaluate steady-state training input performance once data is available locally or cached.

### `bench_streaming.py`

Use this to measure the performance of the streaming pipeline, including:

- dataset access through `StreamingLeRobotDataset`
- streaming from the Hub
- buffering and batch delivery in an iterable dataset setup

This is the benchmark to use when you want to understand the overhead of remote streaming and how much the training loop is gated by data arrival.

Note: this benchmark always uses `num_workers=0`. The code documents that streaming with multiple workers is not supported here because of video decoding instability.

## Requirements

These scripts depend on:

- `torch`
- `numpy`
- `psutil`
- `lerobot`

They assume those dependencies are already installed in your environment.

## How To Run

Run the scripts from the repo root:

```bash
python3 benchmarks/bench_local.py
python3 benchmarks/bench_streaming.py
python3 benchmarks/run_compare.py
```

You can override any of the CLI arguments shown below.

## Arguments

### `bench_local.py`

```bash
python3 benchmarks/bench_local.py \
  --repo-id lerobot/pusht \
  --episodes 0 1 2 3 4 \
  --batch-size 32 \
  --num-workers 4 \
  --num-batches 100 \
  --warmup-batches 10 \
  --prefetch-factor 2
```

Arguments:

- `--repo-id`: dataset repo id to load. Default: `lerobot/pusht`
- `--episodes`: one or more episode indices to benchmark. Default: `0 1 2 3 4`
- `--batch-size`: number of samples per batch. Default: `32`
- `--num-workers`: PyTorch `DataLoader` worker count. Default: `4`
- `--num-batches`: number of measured benchmark batches. Default: `100`
- `--warmup-batches`: number of unmeasured warmup batches to run first. Default: `10`
- `--prefetch-factor`: PyTorch prefetch factor per worker when `num_workers > 0`. Default: `2`

Example:

```bash
python3 benchmarks/bench_local.py --repo-id lerobot/pusht --episodes 0 1 2 --batch-size 64 --num-workers 8
```

### `bench_streaming.py`

```bash
python3 benchmarks/bench_streaming.py \
  --repo-id lerobot/pusht \
  --episodes 0 1 2 3 4 \
  --batch-size 32 \
  --num-batches 100 \
  --warmup-batches 10 \
  --buffer-size 1000
```

Arguments:

- `--repo-id`: dataset repo id to stream. Default: `lerobot/pusht`
- `--episodes`: one or more episode indices to benchmark. Default: `0 1 2 3 4`
- `--batch-size`: number of samples per batch. Default: `32`
- `--num-batches`: number of measured benchmark batches. Default: `100`
- `--warmup-batches`: number of unmeasured warmup batches to run first. Default: `10`
- `--buffer-size`: streaming shuffle/buffer size passed to `StreamingLeRobotDataset`. Default: `1000`

Example:

```bash
python3 benchmarks/bench_streaming.py --repo-id lerobot/pusht --episodes 0 1 2 --batch-size 64 --buffer-size 2000
```

### `run_compare.py`

```bash
python3 benchmarks/run_compare.py \
  --repo-id lerobot/pusht \
  --episodes 0 1 2 3 4 \
  --batch-size 32 \
  --num-batches 100 \
  --warmup-batches 10 \
  --local-num-workers 4 \
  --local-prefetch-factor 2 \
  --stream-buffer-size 1000 \
  --output-dir benchmarks/results
```

Arguments:

- `--repo-id`: dataset repo id used for both benchmarks. Default: `lerobot/pusht`
- `--episodes`: one or more episode indices used for both benchmarks. Default: `0 1 2 3 4`
- `--batch-size`: batch size used for both benchmarks. Default: `32`
- `--num-batches`: number of measured batches for both benchmarks. Default: `100`
- `--warmup-batches`: number of warmup batches for both benchmarks. Default: `10`
- `--local-num-workers`: worker count for the local benchmark. Default: `4`
- `--local-prefetch-factor`: prefetch factor for the local benchmark. Default: `2`
- `--stream-buffer-size`: buffer size for the streaming benchmark. Default: `1000`
- `--output-dir`: parent directory for generated artifacts. Default: `benchmarks/results`
- `--run-name`: optional explicit subdirectory name. If omitted, the script uses a timestamped directory

Example:

```bash
python3 benchmarks/run_compare.py --repo-id lerobot/pusht --episodes 0 1 2 --batch-size 64 --run-name pusht-compare
```

## Output

Both scripts print a summary like:

```text
=== LeRobot Local Benchmark ===
Dataset:          lerobot/pusht (5 episodes)
Batch size:       32
Num workers:      4
Prefetch factor:  2
Batches:          100

--- Results ---
Throughput:       1234.5 samples/sec | 38.6 batches/sec
Latency (ms):     mean=25.9  p50=22.1  p95=41.7  p99=57.3
CPU utilization:  mean=182.4%  max=241.7%
Pipeline stall:   91.3% of iteration time spent waiting for data
```

The streaming benchmark prints the same result fields, but reports `Buffer size` instead of `Prefetch factor`, and `Num workers` is always shown as `0 (required for streaming)`.

## Comparison Runner Output

`run_compare.py` writes a result directory containing:

- `results.json`: structured metrics and the exact benchmark arguments used
- `throughput_samples_per_sec.png`: side-by-side throughput comparison in samples/sec
- `throughput_batches_per_sec.png`: side-by-side throughput comparison in batches/sec
- `latency_comparison.png`: grouped comparison for mean, p50, p95, and p99 latency
- `cpu_comparison.png`: grouped comparison for mean and max CPU utilization
- `stall_ratio_comparison.png`: side-by-side pipeline stall comparison

By default the output goes under `benchmarks/results/<timestamp>/`. If you pass `--run-name`, the output goes under `benchmarks/results/<run-name>/` instead.

## What The Outputs Mean

- `Throughput`: overall data delivery rate. Higher is better. `samples/sec` is the most useful number for comparing end-to-end input capacity across different batch sizes. `batches/sec` is useful when batch size is fixed.
- `Latency (ms)`: time to get one batch. Lower is better. The percentile values show tail behavior:
  - `mean`: average batch time
  - `p50`: typical batch time
  - `p95` and `p99`: slow-batch behavior, which is often what training feels in practice
- `CPU utilization`: average and peak CPU load sampled during the measured section. Higher CPU can be good if it produces more throughput, but it can also indicate expensive decoding or preprocessing.
- `Pipeline stall`: fraction of the benchmark loop spent waiting for the next batch to arrive. Lower is better. A high stall ratio means the consumer is mostly blocked on input rather than doing useful work.

## How To Evaluate The Pipeline

These benchmarks are most useful when comparing configurations, not as absolute pass/fail checks.

Use the results like this:

- Compare `bench_local.py` against `bench_streaming.py` on the same dataset, episodes, batch size, and number of measured batches.
- Prefer configurations with higher throughput and lower `p95`/`p99` latency.
- Treat high `Pipeline stall` as a sign that the input pipeline is the bottleneck.
- Use CPU numbers to distinguish between I/O-bound and CPU-heavy cases:
  - low throughput + low CPU often means waiting on I/O or network
  - low throughput + high CPU often means decoding, transforms, or collation are the bottleneck
- For the local benchmark, vary `--num-workers` and `--prefetch-factor` to find the best steady-state configuration.
- For the streaming benchmark, vary `--buffer-size` to understand the tradeoff between shuffle/buffering behavior and delivery smoothness.

In practice:

- If local throughput is much higher and stall ratio is much lower than streaming, the streaming path is the limiting stage.
- If local throughput is still low, the issue is likely in decoding, collation, transforms, or worker configuration rather than the network.
- If `p50` is good but `p95`/`p99` are bad, the pipeline is unstable and will likely produce training hiccups even if the average looks acceptable.

## Reproducibility Tips

For fair comparisons:

- use the same `--repo-id`, `--episodes`, and `--batch-size`
- keep `--num-batches` large enough to smooth out noise
- run each configuration more than once
- avoid comparing runs from different machines or under very different background CPU/network load
