"""Shared utilities for LeRobot dataset benchmarks."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass
class BenchmarkResult:
    """Structured benchmark output for reporting and plotting."""

    mode: str
    repo_id: str
    episodes: list[int]
    batch_size: int
    num_batches: int
    total_samples: int
    total_time: float
    samples_per_sec: float
    batches_per_sec: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    cpu_mean_percent: float
    cpu_max_percent: float
    stall_ratio: float
    num_workers: int | None = None
    prefetch_factor: int | None = None
    buffer_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CPUMonitor:
    """Background thread that samples CPU utilization."""

    def __init__(self, interval: float = 0.1):
        self._interval = interval
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import psutil

        while not self._stop.is_set():
            self._samples.append(psutil.cpu_percent(interval=self._interval))

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    @property
    def mean(self) -> float:
        return float(mean(self._samples)) if self._samples else 0.0

    @property
    def max(self) -> float:
        return float(max(self._samples)) if self._samples else 0.0


def percentile(values: list[float], q: float) -> float:
    """Compute a percentile using linear interpolation."""

    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    rank = (len(sorted_values) - 1) * (q / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return float(
        sorted_values[lower]
        + (sorted_values[upper] - sorted_values[lower]) * fraction
    )


def build_result(
    mode: str,
    repo_id: str,
    episodes: list[int],
    batch_size: int,
    num_batches: int,
    total_samples: int,
    total_time: float,
    latencies_ms: list[float],
    cpu_mean: float,
    cpu_max: float,
    stall_ratio: float,
    num_workers: int | None = None,
    prefetch_factor: int | None = None,
    buffer_size: int | None = None,
) -> BenchmarkResult:
    """Convert raw benchmark measurements into a structured result."""

    return BenchmarkResult(
        mode=mode,
        repo_id=repo_id,
        episodes=episodes,
        batch_size=batch_size,
        num_batches=num_batches,
        total_samples=total_samples,
        total_time=total_time,
        samples_per_sec=float(total_samples / total_time),
        batches_per_sec=float(num_batches / total_time),
        latency_mean_ms=float(mean(latencies_ms)) if latencies_ms else 0.0,
        latency_p50_ms=percentile(latencies_ms, 50),
        latency_p95_ms=percentile(latencies_ms, 95),
        latency_p99_ms=percentile(latencies_ms, 99),
        cpu_mean_percent=cpu_mean,
        cpu_max_percent=cpu_max,
        stall_ratio=stall_ratio,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        buffer_size=buffer_size,
    )


def print_summary(result: BenchmarkResult, extra_info: dict[str, str] | None = None):
    print()
    print(f"=== LeRobot {result.mode} Benchmark ===")
    print(f"Dataset:          {result.repo_id} ({len(result.episodes)} episodes)")
    print(f"Batch size:       {result.batch_size}")
    if extra_info:
        for label, value in extra_info.items():
            print(f"{label + ':':<18}{value}")
    print(f"Batches:          {result.num_batches}")
    print()
    print("--- Results ---")
    print(
        "Throughput:       "
        f"{result.samples_per_sec:.1f} samples/sec | "
        f"{result.batches_per_sec:.1f} batches/sec"
    )
    print(
        "Latency (ms):     "
        f"mean={result.latency_mean_ms:.1f}  "
        f"p50={result.latency_p50_ms:.1f}  "
        f"p95={result.latency_p95_ms:.1f}  "
        f"p99={result.latency_p99_ms:.1f}"
    )
    print(
        "CPU utilization:  "
        f"mean={result.cpu_mean_percent:.1f}%  "
        f"max={result.cpu_max_percent:.1f}%"
    )
    print(
        "Pipeline stall:   "
        f"{result.stall_ratio * 100:.1f}% of iteration time spent waiting for data"
    )
    print()


def write_results_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Persist benchmark results to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
