from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clinical_speech.config import load_config


SWEEP_CONFIGS = [
    PROJECT_ROOT / "configs" / "systems_benchmark_decode_sweep_16_64.yaml",
    PROJECT_ROOT / "configs" / "systems_benchmark_decode_sweep_16_128.yaml",
    PROJECT_ROOT / "configs" / "systems_benchmark_decode_sweep_32_64.yaml",
    PROJECT_ROOT / "configs" / "systems_benchmark_decode_sweep_32_128.yaml",
]
SWEEP_OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
SWEEP_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def _run_command(command: list[str], *, env: dict[str, str]) -> None:
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=env)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(rows: list[dict], *, metric_key: str, ylabel: str, filename: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row["backend"] != "mistral_paged_static_batch_triton":
            continue
        grouped.setdefault(row["regime"], []).append(row)

    plt.figure(figsize=(9, 5.6))
    for regime, regime_rows in sorted(grouped.items()):
        ordered = sorted(regime_rows, key=lambda item: int(item["batch_size"]))
        plt.plot(
            [int(item["batch_size"]) for item in ordered],
            [float(item[metric_key]) for item in ordered],
            marker="o",
            linewidth=2.4,
            label=regime,
        )
    plt.xlabel("Batch size")
    plt.ylabel(ylabel)
    plt.title(f"Decode sweep: {ylabel}")
    plt.legend(frameon=False)
    plt.tight_layout()
    SWEEP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(SWEEP_FIGURES_DIR / filename, dpi=220)
    plt.close()


def main() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_ROOT)

    all_rows: list[dict] = []
    best_row: dict | None = None

    for config_path in SWEEP_CONFIGS:
        _run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_systems_benchmark.py"),
                "--config",
                str(config_path),
            ],
            env=env,
        )
        cfg = load_config(config_path)
        report_path = cfg.experiment.benchmark_dir / "systems" / "systems_benchmark_report.json"
        report = _load_json(report_path)
        regime = f"prompts={report['num_prompts']}, max_new_tokens={cfg.generation.max_new_tokens}"
        for row in report["summary_rows"]:
            enriched = {
                "config": str(config_path),
                "experiment_name": cfg.experiment.name,
                "regime": regime,
                **row,
            }
            if row["backend"] == "mistral_paged_static_batch_triton":
                hf_match = next(
                    candidate
                    for candidate in report["summary_rows"]
                    if candidate["backend"] == "hf_sequential" and candidate["batch_size"] == row["batch_size"]
                )
                enriched["tok_speedup_vs_hf_pct"] = 100.0 * (
                    float(row["mean_throughput_tok_per_sec"]) / float(hf_match["mean_throughput_tok_per_sec"]) - 1.0
                )
                enriched["req_speedup_vs_hf_pct"] = 100.0 * (
                    float(row["mean_requests_per_sec"]) / float(hf_match["mean_requests_per_sec"]) - 1.0
                )
                if best_row is None or enriched["tok_speedup_vs_hf_pct"] > best_row["tok_speedup_vs_hf_pct"]:
                    best_row = dict(enriched)
            all_rows.append(enriched)

    _write_csv(SWEEP_OUTPUT_DIR / "decode_sweep_summary.csv", all_rows)
    if best_row is not None:
        _write_csv(SWEEP_OUTPUT_DIR / "decode_sweep_best_regime.csv", [best_row])

    _plot_metric(all_rows, metric_key="mean_throughput_tok_per_sec", ylabel="Throughput (tok/s)", filename="decode_sweep_triton_toks.png")
    _plot_metric(all_rows, metric_key="mean_requests_per_sec", ylabel="Requests / sec", filename="decode_sweep_triton_reqs.png")
    _plot_metric(all_rows, metric_key="mean_ttft_sec", ylabel="TTFT (s)", filename="decode_sweep_triton_ttft.png")
    _plot_metric(all_rows, metric_key="mean_decode_latency_sec", ylabel="Decode latency (s)", filename="decode_sweep_triton_decode_latency.png")
    _plot_metric(all_rows, metric_key="mean_peak_gpu_mem_gb", ylabel="Peak GPU memory (GB)", filename="decode_sweep_triton_peak_mem.png")

    print(json.dumps({"summary_csv": str(SWEEP_OUTPUT_DIR / "decode_sweep_summary.csv"), "best_row": best_row}, indent=2))


if __name__ == "__main__":
    main()
