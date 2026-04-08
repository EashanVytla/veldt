"""Benchmark LeRobotDataset with standard PyTorch DataLoader (local/cached mode)."""

import argparse
import time

from utils import CPUMonitor, build_result, print_summary


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark LeRobotDataset (local/cached)")
    p.add_argument("--repo-id", default="lerobot/pusht")
    p.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-batches", type=int, default=100)
    p.add_argument("--warmup-batches", type=int, default=10)
    p.add_argument("--prefetch-factor", type=int, default=2)
    return p.parse_args()


def run_benchmark(args):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading dataset {args.repo_id} (episodes {args.episodes})...")
    dataset = LeRobotDataset(args.repo_id, episodes=args.episodes)
    print(f"Dataset size: {len(dataset)} samples")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=False,
    )

    # Warmup
    print(f"Warming up ({args.warmup_batches} batches)...")
    loader_iter = iter(loader)
    for _ in range(args.warmup_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
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
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
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
        mode="Local",
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
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    print_summary(
        result,
        extra_info={
            "Num workers": str(args.num_workers),
            "Prefetch factor": str(args.prefetch_factor),
        },
    )
    return result


if __name__ == "__main__":
    run_benchmark(parse_args())
