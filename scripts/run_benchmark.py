"""Measure CPU and strict QNN in separate processes, without thermal guesses."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def measure(args):
    """Warm up then measure end-to-end inference and sampled host RSS."""
    import numpy as np
    import psutil
    from src.engine.qnn_detector import HexaGuardDetector
    from src.engine.decompiler import MAX_BYTES
    with args.input.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    process = psutil.Process()
    before = process.memory_info().rss
    detector = HexaGuardDetector(args.model, backend=args.worker, backend_path=args.qnn_backend)
    for _ in range(5):
        detector.predict(data)
    samples, peak = [], process.memory_info().rss
    start = perf_counter()
    for _ in range(args.runs):
        samples.append(detector.predict(data)["latency_ms"])
        peak = max(peak, process.memory_info().rss)
    elapsed = perf_counter() - start
    return {"backend": detector.backend, "mean_ms": float(np.mean(samples)),
            "p99_ms": float(np.percentile(samples, 99)), "throughput": args.runs / elapsed,
            "rss_delta_mib": (peak - before) / 2**20, "thermal": "Not measured"}


def main():
    """Never substitute CPU measurements for an unavailable NPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-model", type=Path)
    parser.add_argument("--qnn-model", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--qnn-backend", default="QnnHtp.dll")
    parser.add_argument("--worker", choices=["cpu", "qnn"], help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.worker:
        try:
            print(json.dumps(measure(args)))
        except Exception as exc:
            print(json.dumps({"error": str(exc)}))
            sys.exit(1)
        return
    if not args.cpu_model or not args.qnn_model:
        parser.error("Supply both --cpu-model and --qnn-model")
    from rich.console import Console
    from rich.table import Table
    table = Table(title=f"HexaGuard: {args.runs} runs after 5 warmups per backend")
    for title in ["Backend", "Mean ms", "P99 ms", "Files/s", "RSS delta MiB", "Thermal footprint"]:
        table.add_column(title)
    for backend, model in [("cpu", args.cpu_model), ("qnn", args.qnn_model)]:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", backend,
                   "--model", str(model), "--input", str(args.input), "--runs", str(args.runs),
                   "--qnn-backend", args.qnn_backend]
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=1800, check=False)
            result = json.loads(run.stdout.strip().splitlines()[-1])
            if run.returncode or "error" in result:
                raise RuntimeError(result.get("error", run.stderr[-1000:]))
            table.add_row(backend, f"{result['mean_ms']:.3f}", f"{result['p99_ms']:.3f}",
                          f"{result['throughput']:.1f}", f"{result['rss_delta_mib']:.2f}", result["thermal"])
        except (subprocess.TimeoutExpired, ValueError, IndexError, RuntimeError) as exc:
            table.add_row(backend, "Unavailable", "-", "-", "-", str(exc))
    Console().print(table)


if __name__ == "__main__":
    main()
