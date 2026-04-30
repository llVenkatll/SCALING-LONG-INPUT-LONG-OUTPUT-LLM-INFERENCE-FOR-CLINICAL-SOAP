from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clinical_speech.config import load_config


DEFAULT_POSTER_RUNS_DIR = PROJECT_ROOT / "poster_assets" / "runs"
DEFAULT_KERNEL_BENCH = Path("/data/project_runtime/benchmarks/kernels/paged_kv_benchmark.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_POSTER_RUNS_DIR)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--kernel-bench", type=Path, default=DEFAULT_KERNEL_BENCH)
    return parser.parse_args()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-").lower()
    return slug or "run"


def _run_command(command: list[str], *, env: dict[str, str]) -> None:
    subprocess.run(command, check=True, env=env, cwd=PROJECT_ROOT)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = _slugify(args.label or cfg.experiment.name)
    output_dir = args.output_root / f"{timestamp}_{label}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_ROOT)

    if not args.skip_benchmark:
        _run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_systems_benchmark.py"),
                "--config",
                str(args.config),
            ],
            env=env,
        )

    benchmark_dir = cfg.experiment.benchmark_dir / "systems"
    summary_csv = benchmark_dir / "systems_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Expected benchmark summary at {summary_csv}")

    _run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_poster_assets.py"),
            "--summary-csv",
            str(summary_csv),
            "--kernel-bench",
            str(args.kernel_bench),
            "--output-dir",
            str(output_dir),
        ],
        env=env,
    )

    manifest = {
        "config": str(args.config),
        "benchmark_dir": str(benchmark_dir),
        "summary_csv": str(summary_csv),
        "poster_assets_dir": str(output_dir),
        "skip_benchmark": args.skip_benchmark,
        "kernel_bench": str(args.kernel_bench),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    latest_path = args.output_root / "LATEST.txt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(output_dir) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
