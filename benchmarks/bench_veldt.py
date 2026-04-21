"""Benchmark veldt data loader against the same dataset/config as the baseline benchmarks."""

import argparse
import time

from utils import CPUMonitor, build_result, print_summary


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark veldt loader")
    p.add_argument("--dataset-path", required=True, help="Path to LeRobot v3 dataset root")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-batches", type=int, default=100)
    p.add_argument("--warmup-batches", type=int, default=10)
    p.add_argument("--buffer-size", type=int, default=10_000)
    p.add_argument("--prefetch-depth", type=int, default=64)
    p.add_argument("--num-decode-threads", type=int, default=0)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--epochs", type=int, default=1, help="Number of epochs to iterate")
    return p.parse_args()


def run_benchmark(args):
    from veldt import Loader, LoaderConfig

    print(f"Loading dataset from {args.dataset_path}...")
    config = LoaderConfig(
        batch_size=args.batch_size,
        num_decode_threads=args.num_decode_threads,
        buffer_size=args.buffer_size,
        prefetch_depth=args.prefetch_depth,
        width=args.width,
        height=args.height,
        normalize=args.normalize,
    )

    loader = Loader(args.dataset_path, config)
    meta = loader.metadata()
    print(f"Dataset: {meta['num_episodes']} episodes, {meta['num_frames']} frames, {meta['fps']} fps")

    # Warmup epoch
    print(f"Warming up ({args.warmup_batches} batches)...")
    loader.set_epoch(epoch=999, seed=0)
    warmup_count = 0
    for batch in loader:
        warmup_count += 1
        if warmup_count >= args.warmup_batches:
            break

    # Benchmark
    total_batches_to_measure = args.num_batches
    print(f"Benchmarking ({total_batches_to_measure} batches across {args.epochs} epoch(s))...")

    cpu_monitor = CPUMonitor()
    cpu_monitor.start()

    batch_latencies = []
    wait_times = []
    total_samples = 0
    batches_measured = 0

    wall_start = time.perf_counter()

    for epoch in range(args.epochs):
        loader.set_epoch(epoch=epoch, seed=42)

        for batch in loader:
            t_start = time.perf_counter()

            # Access the data to force materialization
            frames = batch.frames
            _ = frames.shape
            for key in batch.keys():
                tab = batch.tabular(key)
                if tab is not None:
                    _ = tab.shape

            t_end = time.perf_counter()

            batch_latencies.append(t_end - t_start)
            wait_times.append(t_end - t_start)
            total_samples += batch.batch_size
            batches_measured += 1

            if batches_measured >= total_batches_to_measure:
                break

        if batches_measured >= total_batches_to_measure:
            break

    wall_end = time.perf_counter()
    cpu_monitor.stop()

    total_time = wall_end - wall_start
    latencies_ms = [lat * 1000 for lat in batch_latencies]
    total_wait = sum(wait_times)

    # Get veldt cache stats
    stats = loader.stats()

    result = build_result(
        mode="Veldt",
        repo_id=args.dataset_path,
        episodes=[],
        batch_size=args.batch_size,
        num_batches=batches_measured,
        total_samples=total_samples,
        total_time=total_time,
        latencies_ms=latencies_ms,
        cpu_mean=cpu_monitor.mean,
        cpu_max=cpu_monitor.max,
        stall_ratio=total_wait / total_time if total_time > 0 else 0,
        buffer_size=args.buffer_size,
    )

    print_summary(
        result,
        extra_info={
            "Buffer size": str(args.buffer_size),
            "Prefetch depth": str(args.prefetch_depth),
            "Decode threads": str(args.num_decode_threads or "auto"),
            "Cache hits": str(stats.get("cache_hits", 0)),
            "Cache misses": str(stats.get("cache_misses", 0)),
            "Evictions": str(stats.get("evictions", 0)),
            "Refetches": str(stats.get("refetches", 0)),
        },
    )
    return result


if __name__ == "__main__":
    run_benchmark(parse_args())
