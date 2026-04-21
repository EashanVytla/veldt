"""Run veldt benchmark against baseline results, generate comparison artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bench_veldt import run_benchmark as run_veldt_benchmark
from utils import BenchmarkResult, write_results_json


def parse_args():
    p = argparse.ArgumentParser(
        description="Run veldt benchmark and compare against baseline LeRobot results."
    )
    p.add_argument("--dataset-path", required=True, help="Path to LeRobot v3 dataset root")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-batches", type=int, default=100)
    p.add_argument("--warmup-batches", type=int, default=10)
    p.add_argument("--buffer-size", type=int, default=10_000)
    p.add_argument("--prefetch-depth", type=int, default=64)
    p.add_argument("--num-decode-threads", type=int, default=0)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--height", type=int, default=224)
    p.add_argument(
        "--baseline-results",
        default="benchmarks/results/pusht-compare/results.json",
        help="Path to baseline results JSON (from run_compare.py)",
    )
    p.add_argument("--output-dir", default="benchmarks/results")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def load_baseline(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        print(f"  Baseline results not found at {p}, skipping comparison.")
        return None
    return json.loads(p.read_text())


def print_comparison(veldt: BenchmarkResult, baseline: dict):
    """Print a side-by-side comparison table."""
    local = baseline.get("results", {}).get("local", {})
    streaming = baseline.get("results", {}).get("streaming", {})

    print()
    print("=" * 70)
    print("COMPARISON: veldt vs. PyTorch DataLoader (LeRobot baseline)")
    print("=" * 70)
    print()

    def row(label, veldt_val, local_val, streaming_val=None, unit="", fmt=".1f"):
        v = f"{veldt_val:{fmt}}{unit}"
        l = f"{local_val:{fmt}}{unit}" if local_val is not None else "—"
        s = f"{streaming_val:{fmt}}{unit}" if streaming_val is not None else "—"
        print(f"  {label:<25} {'veldt':<15} {'Local':<15} {'Streaming':<15}")
        print(f"  {'':<25} {v:<15} {l:<15} {s:<15}")
        print()

    print(f"  {'Metric':<25} {'veldt':<15} {'Local':<15} {'Streaming':<15}")
    print(f"  {'-' * 25} {'-' * 14} {'-' * 14} {'-' * 14}")

    def fmt_row(label, vv, lk, sk=None):
        lv = local.get(lk)
        sv = streaming.get(sk or lk) if streaming else None
        print(f"  {label:<25} {vv:<15} {str(lv or '—'):<15} {str(sv or '—'):<15}")

    # Throughput
    fmt_row(
        "Samples/sec",
        f"{veldt.samples_per_sec:.1f}",
        "samples_per_sec",
    )
    fmt_row(
        "Batches/sec",
        f"{veldt.batches_per_sec:.1f}",
        "batches_per_sec",
    )

    # Latency
    fmt_row("Latency mean (ms)", f"{veldt.latency_mean_ms:.1f}", "latency_mean_ms")
    fmt_row("Latency P50 (ms)", f"{veldt.latency_p50_ms:.1f}", "latency_p50_ms")
    fmt_row("Latency P95 (ms)", f"{veldt.latency_p95_ms:.1f}", "latency_p95_ms")
    fmt_row("Latency P99 (ms)", f"{veldt.latency_p99_ms:.1f}", "latency_p99_ms")

    # Speedup
    local_sps = local.get("samples_per_sec", 0)
    if local_sps and local_sps > 0:
        speedup = veldt.samples_per_sec / local_sps
        print()
        print(f"  Throughput speedup vs Local:     {speedup:.1f}x")

    local_p99 = local.get("latency_p99_ms", 0)
    if local_p99 and local_p99 > 0:
        p99_improvement = local_p99 / veldt.latency_p99_ms if veldt.latency_p99_ms > 0 else float("inf")
        print(f"  P99 latency improvement vs Local: {p99_improvement:.1f}x")

    print()


def generate_comparison_plots(run_dir: Path, veldt_result: BenchmarkResult, baseline: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("  matplotlib not installed, skipping plot generation.")
        return

    local = baseline.get("results", {}).get("local", {})
    streaming = baseline.get("results", {}).get("streaming", {})

    labels = ["veldt", "Local (PyTorch)", "Streaming"]
    colors = ["#00C853", "#2F6BFF", "#FF7A18"]

    def save_bar(filename, title, ylabel, values):
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9,
            )
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=150)
        plt.close(fig)

    save_bar(
        "throughput_comparison.png",
        "Throughput Comparison",
        "Samples / sec",
        [
            veldt_result.samples_per_sec,
            local.get("samples_per_sec", 0),
            streaming.get("samples_per_sec", 0),
        ],
    )

    # Latency grouped bar
    latency_labels = ["mean", "p50", "p95", "p99"]
    veldt_lat = [
        veldt_result.latency_mean_ms,
        veldt_result.latency_p50_ms,
        veldt_result.latency_p95_ms,
        veldt_result.latency_p99_ms,
    ]
    local_lat = [
        local.get("latency_mean_ms", 0),
        local.get("latency_p50_ms", 0),
        local.get("latency_p95_ms", 0),
        local.get("latency_p99_ms", 0),
    ]
    streaming_lat = [
        streaming.get("latency_mean_ms", 0),
        streaming.get("latency_p50_ms", 0),
        streaming.get("latency_p95_ms", 0),
        streaming.get("latency_p99_ms", 0),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(latency_labels))
    width = 0.25
    ax.bar([i - width for i in x], veldt_lat, width=width, label="veldt", color=colors[0])
    ax.bar(list(x), local_lat, width=width, label="Local (PyTorch)", color=colors[1])
    ax.bar([i + width for i in x], streaming_lat, width=width, label="Streaming", color=colors[2])
    ax.set_title("Latency Comparison (ms)")
    ax.set_ylabel("Milliseconds")
    ax.set_xticks(list(x))
    ax.set_xticklabels(latency_labels)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "latency_comparison.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    run_name = args.run_name or f"veldt-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build veldt benchmark args
    veldt_args = argparse.Namespace(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        warmup_batches=args.warmup_batches,
        buffer_size=args.buffer_size,
        prefetch_depth=args.prefetch_depth,
        num_decode_threads=args.num_decode_threads,
        width=args.width,
        height=args.height,
        normalize=False,
        epochs=1,
    )

    print("Running veldt benchmark...")
    veldt_result = run_veldt_benchmark(veldt_args)

    # Load and compare with baseline
    baseline = load_baseline(args.baseline_results)
    if baseline:
        print_comparison(veldt_result, baseline)
        generate_comparison_plots(run_dir, veldt_result, baseline)

    # Save results
    payload = {
        "run_name": run_name,
        "veldt_args": {
            "dataset_path": args.dataset_path,
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "buffer_size": args.buffer_size,
            "prefetch_depth": args.prefetch_depth,
        },
        "results": {
            "veldt": veldt_result.to_dict(),
        },
    }
    if baseline:
        payload["baseline"] = baseline.get("results", {})

    write_results_json(run_dir / "results.json", payload)
    print(f"Results written to {run_dir}")


if __name__ == "__main__":
    main()
