"""Benchmark StreamingLeRobotDataset (streaming from HuggingFace Hub)."""

import argparse
import multiprocessing as mp
import time

from utils import CPUMonitor, build_result, print_summary


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark StreamingLeRobotDataset (Hub streaming)")
    p.add_argument("--repo-id", default="lerobot/pusht")
    p.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-batches", type=int, default=100)
    p.add_argument("--warmup-batches", type=int, default=10)
    p.add_argument("--buffer-size", type=int, default=1000)
    return p.parse_args()


def run_benchmark(args):
    import torch
    from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

    mp.set_start_method('spawn', force=True)

    print(f"Creating streaming dataset {args.repo_id} (episodes {args.episodes})...")
    dataset = StreamingLeRobotDataset(
        repo_id=args.repo_id,
        episodes=args.episodes,
        streaming=True,
        buffer_size=args.buffer_size,
        shuffle=True,
    )

    # IterableDataset: num_workers=0 required (video decoding segfaults otherwise)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=3,
    )

    # Warmup
    print(f"Warming up ({args.warmup_batches} batches)...")
    loader_iter = iter(loader)
    for _ in range(args.warmup_batches):
        batch = next(loader_iter)

    # Benchmark
    print(f"Benchmarking ({args.num_batches} batches)...")
    cpu_monitor = CPUMonitor()
    cpu_monitor.start()

    batch_latencies = []
    wait_times = []
    total_samples = 0

    wall_start = time.perf_counter()

    for i in range(args.num_batches):
        t_request = time.perf_counter()
        batch = next(loader_iter)
        t_data = time.perf_counter()

        # Minimal processing — access tensor shapes to force materialization
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                _ = v.shape

        t_end = time.perf_counter()

        batch_latencies.append(t_end - t_request)
        wait_times.append(t_data - t_request)
        total_samples += args.batch_size

    wall_end = time.perf_counter()
    cpu_monitor.stop()

    total_time = wall_end - wall_start
    latencies_ms = [latency * 1000 for latency in batch_latencies]
    total_wait = sum(wait_times)

    result = build_result(
        mode="Streaming",
        repo_id=args.repo_id,
        episodes=args.episodes,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        total_samples=total_samples,
        total_time=total_time,
        latencies_ms=latencies_ms,
        cpu_mean=cpu_monitor.mean,
        cpu_max=cpu_monitor.max,
        stall_ratio=total_wait / total_time,
        num_workers=3,
        buffer_size=args.buffer_size,
    )
    print_summary(
        result,
        extra_info={
            "Num workers": "0 (required for streaming)",
            "Buffer size": str(args.buffer_size),
        },
    )
    return result


if __name__ == "__main__":
    run_benchmark(parse_args())
