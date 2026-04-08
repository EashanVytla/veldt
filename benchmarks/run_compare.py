"""Run local and streaming benchmarks, then generate comparison artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from bench_local import run_benchmark as run_local_benchmark
from bench_streaming import run_benchmark as run_streaming_benchmark
from utils import write_results_json


def parse_args():
    p = argparse.ArgumentParser(
        description="Run local and streaming LeRobot benchmarks and compare the results."
    )
    p.add_argument("--repo-id", default="lerobot/aloha_sim_insertion_human")
    p.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-batches", type=int, default=100)
    p.add_argument("--warmup-batches", type=int, default=10)
    p.add_argument("--local-num-workers", type=int, default=4)
    p.add_argument("--local-prefetch-factor", type=int, default=2)
    p.add_argument("--stream-buffer-size", type=int, default=1000)
    p.add_argument("--output-dir", default="benchmarks/results")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def resolve_run_dir(output_dir: str, run_name: str | None) -> Path:
    """Build the directory where comparison artifacts will be written."""

    base_dir = Path(output_dir)
    if run_name:
        return base_dir / run_name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return base_dir / timestamp


def build_local_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        repo_id=args.repo_id,
        episodes=args.episodes,
        batch_size=args.batch_size,
        num_workers=args.local_num_workers,
        num_batches=args.num_batches,
        warmup_batches=args.warmup_batches,
        prefetch_factor=args.local_prefetch_factor,
    )


def build_streaming_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        repo_id=args.repo_id,
        episodes=args.episodes,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        warmup_batches=args.warmup_batches,
        buffer_size=args.stream_buffer_size,
    )


def generate_plots(run_dir: Path, local_result, streaming_result):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to generate comparison plots. "
            "Install it in the benchmark environment and rerun the script."
        ) from exc

    labels = ["Local", "Streaming"]

    def save_bar_plot(filename: str, title: str, ylabel: str, values: list[float]):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(labels, values, color=["#2F6BFF", "#FF7A18"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=150)
        plt.close(fig)

    save_bar_plot(
        "throughput_samples_per_sec.png",
        "Throughput Comparison",
        "Samples / sec",
        [local_result.samples_per_sec, streaming_result.samples_per_sec],
    )
    save_bar_plot(
        "throughput_batches_per_sec.png",
        "Batch Rate Comparison",
        "Batches / sec",
        [local_result.batches_per_sec, streaming_result.batches_per_sec],
    )

    latency_labels = ["mean", "p50", "p95", "p99"]
    local_latency = [
        local_result.latency_mean_ms,
        local_result.latency_p50_ms,
        local_result.latency_p95_ms,
        local_result.latency_p99_ms,
    ]
    streaming_latency = [
        streaming_result.latency_mean_ms,
        streaming_result.latency_p50_ms,
        streaming_result.latency_p95_ms,
        streaming_result.latency_p99_ms,
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(latency_labels))
    width = 0.35
    ax.bar([i - width / 2 for i in x], local_latency, width=width, label="Local", color="#2F6BFF")
    ax.bar(
        [i + width / 2 for i in x],
        streaming_latency,
        width=width,
        label="Streaming",
        color="#FF7A18",
    )
    ax.set_title("Latency Comparison")
    ax.set_ylabel("Milliseconds")
    ax.set_xticks(list(x))
    ax.set_xticklabels(latency_labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "latency_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cpu_labels = ["mean", "max"]
    x = range(len(cpu_labels))
    ax.bar(
        [i - width / 2 for i in x],
        [local_result.cpu_mean_percent, local_result.cpu_max_percent],
        width=width,
        label="Local",
        color="#2F6BFF",
    )
    ax.bar(
        [i + width / 2 for i in x],
        [streaming_result.cpu_mean_percent, streaming_result.cpu_max_percent],
        width=width,
        label="Streaming",
        color="#FF7A18",
    )
    ax.set_title("CPU Utilization Comparison")
    ax.set_ylabel("CPU percent")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cpu_labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "cpu_comparison.png", dpi=150)
    plt.close(fig)

    save_bar_plot(
        "stall_ratio_comparison.png",
        "Pipeline Stall Comparison",
        "Stall ratio",
        [local_result.stall_ratio, streaming_result.stall_ratio],
    )


def main():
    args = parse_args()
    run_dir = resolve_run_dir(args.output_dir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Running local benchmark...")
    local_result = run_local_benchmark(build_local_args(args))

    print("Running streaming benchmark...")
    streaming_result = run_streaming_benchmark(build_streaming_args(args))

    print(f"Writing comparison artifacts to {run_dir}...")
    payload = {
        "run_name": run_dir.name,
        "shared_args": {
            "repo_id": args.repo_id,
            "episodes": args.episodes,
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "warmup_batches": args.warmup_batches,
        },
        "local_args": {
            "num_workers": args.local_num_workers,
            "prefetch_factor": args.local_prefetch_factor,
        },
        "streaming_args": {
            "buffer_size": args.stream_buffer_size,
        },
        "results": {
            "local": local_result.to_dict(),
            "streaming": streaming_result.to_dict(),
        },
        "plots": [
            "throughput_samples_per_sec.png",
            "throughput_batches_per_sec.png",
            "latency_comparison.png",
            "cpu_comparison.png",
            "stall_ratio_comparison.png",
        ],
    }
    write_results_json(run_dir / "results.json", payload)
    generate_plots(run_dir, local_result, streaming_result)
    print("Comparison run complete.")


if __name__ == "__main__":
    main()
