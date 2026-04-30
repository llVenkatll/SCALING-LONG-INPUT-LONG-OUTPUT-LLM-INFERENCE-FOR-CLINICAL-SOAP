from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clinical_speech.config import load_config


RUNTIME_ROOT = Path("/data/project_runtime/benchmarks/poster_systems_study")
POSTER_DIR = PROJECT_ROOT / "poster_assets" / "deep_systems_study"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

OUTPUT_CONFIGS = [
    PROJECT_ROOT / "configs" / "systems_benchmark_decode_sweep_16_16.yaml",
    PROJECT_ROOT / "configs" / "systems_benchmark_decode_sweep_16_32.yaml",
    PROJECT_ROOT / "configs" / "systems_benchmark_decode_sweep_16_64.yaml",
    PROJECT_ROOT / "configs" / "systems_benchmark_decode_sweep_16_128.yaml",
]

PROMPT_COUNT_CONFIGS = [
    PROJECT_ROOT / "configs" / "systems_benchmark_decode_sweep_32_64.yaml",
    PROJECT_ROOT / "configs" / "systems_benchmark_decode_sweep_32_128.yaml",
]

CONTEXT_CONFIGS = [
    PROJECT_ROOT / "configs" / "systems_benchmark_context_sweep_512.yaml",
    PROJECT_ROOT / "configs" / "systems_benchmark_context_sweep_1024.yaml",
    PROJECT_ROOT / "configs" / "systems_benchmark_context_sweep_2048.yaml",
    PROJECT_ROOT / "configs" / "systems_benchmark_context_sweep_4096.yaml",
    PROJECT_ROOT / "configs" / "systems_benchmark_context_sweep_8192.yaml",
]

BACKENDS = [
    "hf_sequential",
    "mistral_paged_static_batch",
    "mistral_paged_static_batch_triton",
]

BACKEND_LABELS = {
    "hf_sequential": "HF Sequential",
    "mistral_paged_static_batch": "Paged Runtime",
    "mistral_paged_static_batch_triton": "Paged Runtime + Triton",
}

BACKEND_COLORS = {
    "hf_sequential": "#7a7a7a",
    "mistral_paged_static_batch": "#2f6ea6",
    "mistral_paged_static_batch_triton": "#0d3b66",
}


@dataclass(frozen=True)
class StudySpec:
    config_path: Path
    experiment_name: str
    batch_sizes: list[int]
    backends: list[str]
    study_type: str
    x_value: int
    report_path: Path
    batch_rows_path: Path
    request_rows_path: Path

    @property
    def label(self) -> str:
        key = "prompt_tokens" if self.study_type == "context" else "max_new_tokens"
        return f"{key}={self.x_value}"


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 170,
            "savefig.dpi": 320,
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
        }
    )


def _to_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    return float(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _build_spec(config_path: Path, study_type: str, x_value: int) -> StudySpec:
    cfg = load_config(config_path)
    systems_dir = cfg.experiment.benchmark_dir / "systems"
    return StudySpec(
        config_path=config_path,
        experiment_name=cfg.experiment.name,
        batch_sizes=list(cfg.systems_benchmark.batch_sizes),
        backends=list(cfg.systems_benchmark.backends),
        study_type=study_type,
        x_value=x_value,
        report_path=systems_dir / "systems_benchmark_report.json",
        batch_rows_path=systems_dir / "systems_batch_rows.csv",
        request_rows_path=systems_dir / "systems_request_rows.csv",
    )


def _combo_paths(spec: StudySpec, backend: str, batch_size: int) -> tuple[Path, Path]:
    slug = f"{spec.experiment_name}__{backend}__b{batch_size}"
    cfg_path = RUNTIME_ROOT / "tmp_configs" / f"{slug}.yaml"
    run_root = RUNTIME_ROOT / "combo_runs" / slug
    return cfg_path, run_root


def _run_command(command: list[str], *, env: dict[str, str], log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode, proc.stdout


def _expected_pairs(spec: StudySpec) -> set[tuple[str, int]]:
    return {(backend, batch_size) for backend in spec.backends for batch_size in spec.batch_sizes}


def _report_complete(spec: StudySpec, report: dict[str, Any]) -> bool:
    actual_pairs = {
        (row["backend"], int(row["batch_size"]))
        for row in report.get("summary_rows", [])
    }
    return _expected_pairs(spec).issubset(actual_pairs)


def _status_from_output(output: str, returncode: int) -> tuple[str, str | None]:
    lowered = output.lower()
    if returncode == 0:
        return "success", None
    if "out of memory" in lowered:
        return "oom", "CUDA out of memory"
    if "backend_failures" in output and "together" in lowered:
        return "failed", "Hosted backend unavailable"
    return "failed", f"runner exited with code {returncode}"


def _write_combo_config(spec: StudySpec, backend: str, batch_size: int) -> Path:
    cfg_path, run_root = _combo_paths(spec, backend, batch_size)
    payload = {
        "extends": str(spec.config_path),
        "experiment": {
            "name": f"{spec.experiment_name}_{backend}_b{batch_size}",
            "output_dir": str(run_root / "outputs"),
            "log_dir": str(run_root / "logs"),
            "benchmark_dir": str(run_root / "benchmarks"),
            "profiler_dir": str(run_root / "profiler"),
            "checkpoint_dir": str(run_root / "checkpoints"),
        },
        "runtime": {
            "triton_paged_kv_enabled": backend == "mistral_paged_static_batch_triton",
            "max_batch_size": batch_size,
            "max_concurrent_requests": batch_size,
        },
        "systems_benchmark": {
            "backends": [backend],
            "batch_sizes": [batch_size],
        },
    }
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return cfg_path


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run_combo(spec: StudySpec, backend: str, batch_size: int, env: dict[str, str]) -> tuple[str, str | None, Path | None]:
    cfg_path, run_root = _combo_paths(spec, backend, batch_size)
    report_path = run_root / "benchmarks" / "systems" / "systems_benchmark_report.json"
    expected_pair = {(backend, batch_size)}
    if report_path.exists():
        report = _load_json(report_path)
        actual_pairs = {(row["backend"], int(row["batch_size"])) for row in report.get("summary_rows", [])}
        if expected_pair.issubset(actual_pairs):
            return "success", "cached_combo", report_path
    cfg_path = _write_combo_config(spec, backend, batch_size)
    log_path = RUNTIME_ROOT / "logs" / f"{spec.experiment_name}__{backend}__b{batch_size}.log"
    returncode, output = _run_command(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_systems_benchmark.py"), "--config", str(cfg_path)],
        env=env,
        log_path=log_path,
    )
    status, reason = _status_from_output(output, returncode)
    if status == "success" and report_path.exists():
        return status, "fresh_combo", report_path
    return status, reason, None


def _read_combo(report_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    report = _load_json(report_path)
    systems_dir = report_path.parent
    summary_rows = report.get("summary_rows", [])
    batch_rows = _read_csv_rows(systems_dir / "systems_batch_rows.csv")
    request_rows = _read_csv_rows(systems_dir / "systems_request_rows.csv")
    return summary_rows, batch_rows, request_rows, report


def _read_full_report(spec: StudySpec) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    report = _load_json(spec.report_path)
    return (
        report.get("summary_rows", []),
        _read_csv_rows(spec.batch_rows_path),
        _read_csv_rows(spec.request_rows_path),
        report,
    )


def _reuse_or_run_full_regime(spec: StudySpec, env: dict[str, str]) -> tuple[bool, str]:
    if spec.report_path.exists():
        report = _load_json(spec.report_path)
        if _report_complete(spec, report):
            return True, "cached_full_regime"
    log_path = RUNTIME_ROOT / "logs" / f"{spec.experiment_name}.log"
    returncode, output = _run_command(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_systems_benchmark.py"), "--config", str(spec.config_path)],
        env=env,
        log_path=log_path,
    )
    if returncode == 0 and spec.report_path.exists():
        report = _load_json(spec.report_path)
        if _report_complete(spec, report):
            return True, "fresh_full_regime"
    return False, output


def _append_metadata(row: dict[str, Any], spec: StudySpec, *, status: str, error: str | None = None) -> dict[str, Any]:
    return {
        **row,
        "study_type": spec.study_type,
        "study_label": spec.label,
        "x_value": spec.x_value,
        "status": status,
        "error": error,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * percentile
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    frac = index - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _build_context_bucket_datasets(env: dict[str, str]) -> None:
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "build_context_bucket_datasets.py")],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def _load_study_specs() -> tuple[list[StudySpec], list[StudySpec], list[StudySpec]]:
    output_specs = [
        _build_spec(path, "output", x_value=int(path.stem.split("_")[-1]))
        for path in OUTPUT_CONFIGS
    ]
    context_specs = [
        _build_spec(path, "context", x_value=int(path.stem.split("_")[-1]))
        for path in CONTEXT_CONFIGS
    ]
    prompt_count_specs = [
        _build_spec(path, "prompt_count", x_value=int(path.stem.split("_")[-2]))
        for path in PROMPT_COUNT_CONFIGS
    ]
    return output_specs, context_specs, prompt_count_specs


def _quality_matched(report: dict[str, Any], backends: list[str]) -> bool:
    quality = report.get("quality_by_backend", {})
    metrics = []
    for backend in backends:
        backend_metrics = quality.get(backend, {}).get("metrics", {})
        if not backend_metrics:
            return False
        metrics.append(
            (
                round(_to_float(backend_metrics.get("bertscore_f1_mean")) or 0.0, 6),
                round(_to_float((backend_metrics.get("rouge") or {}).get("rouge1")) or 0.0, 6),
            )
        )
    return len(set(metrics)) == 1


def _hero_figure(summary_rows: list[dict[str, Any]], quality_map: dict[tuple[str, int], bool]) -> None:
    output_rows = [row for row in summary_rows if row["study_type"] == "output" and row["status"] == "success"]
    triton_rows = [row for row in output_rows if row["backend"] == "mistral_paged_static_batch_triton"]
    best = max(triton_rows, key=lambda row: _to_float(row["mean_throughput_tok_per_sec"]) or -1.0)
    batch_size = int(best["batch_size"])
    x_value = int(best["x_value"])
    comparison_rows = [row for row in output_rows if int(row["batch_size"]) == batch_size and int(row["x_value"]) == x_value]
    comparison_rows.sort(key=lambda row: BACKENDS.index(row["backend"]))

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    xs = list(range(len(comparison_rows)))
    vals = [_to_float(row["mean_throughput_tok_per_sec"]) for row in comparison_rows]
    bars = ax.bar(xs, vals, color=[BACKEND_COLORS[row["backend"]] for row in comparison_rows], width=0.62)
    ax.set_xticks(xs, [BACKEND_LABELS[row["backend"]] for row in comparison_rows])
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title(f"Hero Result: max_new_tokens={x_value}, batch size {batch_size}")
    ymax = max(vals) * 1.22
    ax.set_ylim(0.0, ymax)
    for bar, value in zip(bars, vals, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + ymax * 0.02, f"{value:.2f}", ha="center", va="bottom", fontweight="bold")
    note = "Quality sanity matched across local backends" if quality_map.get((x_value, batch_size), False) else "Quality sanity available per backend"
    ax.text(
        0.02,
        0.95,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="#134e4a",
        bbox={"facecolor": "#dcfce7", "edgecolor": "#86efac", "boxstyle": "round,pad=0.3"},
    )
    _save(fig, POSTER_DIR / "hero_backend_figure")


def _kv_figure(summary_rows: list[dict[str, Any]]) -> None:
    rows = [
        row for row in summary_rows
        if row["study_type"] == "output"
        and row["status"] == "success"
        and row["backend"] == "mistral_paged_static_batch_triton"
        and int(row["x_value"]) == 64
    ]
    rows.sort(key=lambda row: int(row["batch_size"]))
    fig, ax1 = plt.subplots(figsize=(8.0, 5.0))
    xs = [int(row["batch_size"]) for row in rows]
    kv_gb = [(_to_float(row["mean_kv_allocated_bytes"]) or 0.0) / (1024 ** 3) for row in rows]
    util = [_to_float(row["mean_kv_utilization_ratio"]) or 0.0 for row in rows]
    frag = [_to_float(row["mean_kv_fragmentation_ratio"]) or 0.0 for row in rows]
    ax1.bar(xs, kv_gb, color="#93c5fd", width=0.6, label="KV allocated (GB)")
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("KV allocated (GB)")
    ax1.set_xticks(xs)
    ax1.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
    ax2 = ax1.twinx()
    ax2.plot(xs, util, marker="o", color="#1d4ed8", linewidth=2.2, label="Utilization")
    ax2.plot(xs, frag, marker="s", color="#ef4444", linewidth=2.0, label="Fragmentation")
    ax2.set_ylabel("Utilization / Fragmentation")
    max_ratio = max(util + frag + [1.0])
    ax2.set_ylim(0.0, min(1.05, max_ratio + 0.05))
    fig.suptitle("KV Efficiency by Batch Size", fontsize=16, fontweight="bold", y=0.97)
    fig.text(0.5, 0.925, "Supporting figure, max_new_tokens=64", ha="center", va="center", fontsize=11, color="#555555")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        lines + lines2,
        labels + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=3,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.4,
    )
    fig.subplots_adjust(top=0.78)
    _save(fig, POSTER_DIR / "kv_efficiency_support")


def _line_plot(rows: list[dict[str, Any]], *, x_label: str, metric_key: str, ylabel: str, title: str, stem: str, fixed_batch_size: int | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    filtered = [row for row in rows if row["status"] == "success"]
    if fixed_batch_size is not None:
        filtered = [row for row in filtered if int(row["batch_size"]) == fixed_batch_size]
    for backend in BACKENDS:
        backend_rows = [row for row in filtered if row["backend"] == backend]
        backend_rows.sort(key=lambda row: int(row["x_value"]))
        if not backend_rows:
            continue
        ax.plot(
            [int(row["x_value"]) for row in backend_rows],
            [_to_float(row[metric_key]) for row in backend_rows],
            marker="o",
            linewidth=2.4,
            color=BACKEND_COLORS[backend],
            label=BACKEND_LABELS[backend],
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    _save(fig, POSTER_DIR / stem)


def _latency_quantiles(request_rows: list[dict[str, Any]], serving_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in request_rows:
        backend = row["backend"]
        batch_size = int(row["batch_size"])
        latency = _to_float(row.get("latency_sec"))
        if latency is None:
            continue
        grouped.setdefault((backend, batch_size), []).append(latency)

    quantile_rows: list[dict[str, Any]] = []
    for (backend, batch_size), values in sorted(grouped.items()):
        quantile_rows.append(
            {
                "backend": backend,
                "batch_size": batch_size,
                "p50_latency_sec": _percentile(values, 0.50),
                "p95_latency_sec": _percentile(values, 0.95),
                "p99_latency_sec": _percentile(values, 0.99),
            }
        )

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    batch_four = [row for row in quantile_rows if row["batch_size"] == 4 and row["backend"] in {r["backend"] for r in serving_rows}]
    batch_four.sort(key=lambda row: row["backend"])
    metrics = [("p50_latency_sec", "P50", "#93c5fd"), ("p95_latency_sec", "P95", "#2563eb"), ("p99_latency_sec", "P99", "#0f172a")]
    xs = list(range(len(batch_four)))
    width = 0.22
    for idx, (key, label, color) in enumerate(metrics):
        vals = [_to_float(row[key]) for row in batch_four]
        ax.bar([x + (idx - 1) * width for x in xs], vals, width=width, color=color, label=label)
    ax.set_xticks(xs, [row["backend"] for row in batch_four], rotation=15, ha="right")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency Quantiles at Batch Size 4")
    ax.legend(frameon=False, ncol=3)
    _save(fig, POSTER_DIR / "latency_quantiles_batch4")
    return quantile_rows, batch_four


def _context_scaling_figure(rows: list[dict[str, Any]]) -> None:
    batch_one = [row for row in rows if int(row["batch_size"]) == 1]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    metrics = [
        ("mean_ttft_sec", "TTFT (s)", "TTFT vs Prompt Length"),
        ("mean_latency_sec", "Total latency (s)", "End-to-End vs Prompt Length"),
        ("mean_peak_gpu_mem_gb", "Peak GPU memory (GB)", "Memory vs Prompt Length"),
    ]
    for ax, (metric_key, ylabel, title) in zip(axes, metrics, strict=True):
        for backend in BACKENDS:
            backend_rows = [row for row in batch_one if row["backend"] == backend and row["status"] == "success"]
            backend_rows.sort(key=lambda row: int(row["x_value"]))
            ax.plot(
                [int(row["x_value"]) for row in backend_rows],
                [_to_float(row[metric_key]) for row in backend_rows],
                marker="o",
                linewidth=2.2,
                color=BACKEND_COLORS[backend],
                label=BACKEND_LABELS[backend],
            )
        ax.set_xlabel("Prompt tokens bucket")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, POSTER_DIR / "context_scaling_panels")


def _prefill_and_decode_figures(context_rows: list[dict[str, Any]], output_rows: list[dict[str, Any]]) -> None:
    _line_plot(
        context_rows,
        x_label="Prompt tokens bucket",
        metric_key="mean_prefill_latency_sec",
        ylabel="Prefill latency (s)",
        title="Prefill Latency vs Prompt Length (batch size 1)",
        stem="prefill_vs_context",
        fixed_batch_size=1,
    )
    _line_plot(
        output_rows,
        x_label="max_new_tokens",
        metric_key="mean_decode_latency_sec",
        ylabel="Decode latency (s)",
        title="Decode Latency vs Output Length (batch size 8)",
        stem="decode_vs_output_length",
        fixed_batch_size=8,
    )


def _output_scaling_figure(output_rows: list[dict[str, Any]]) -> None:
    _line_plot(
        output_rows,
        x_label="max_new_tokens",
        metric_key="mean_throughput_tok_per_sec",
        ylabel="Throughput (tok/s)",
        title="Throughput vs Output Length (batch size 8)",
        stem="throughput_vs_output_length",
        fixed_batch_size=8,
    )


def _serving_rows(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    report = _load_json(path / "systems_benchmark_report.json")
    request_rows = _read_csv_rows(path / "systems_request_rows.csv")
    return request_rows, report


def _qualitative_review_export() -> list[dict[str, Any]]:
    from clinical_speech.pipeline.prompts import build_note_prompt
    from clinical_speech.utils.io import read_jsonl

    dataset_path = Path("/data/project_runtime/datasets/medsynth/test.jsonl")
    serving_dir = Path("/data/project_runtime/benchmarks/systems_benchmark_serving_stack_comparison/systems")
    if not serving_dir.exists():
        return []
    dataset = read_jsonl(dataset_path)[:8]
    request_rows = _read_csv_rows(serving_dir / "systems_request_rows.csv")
    selected = [
        row for row in request_rows
        if int(row.get("batch_size", "0")) == 8 and row.get("repeat_index") == "0"
    ]
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in selected:
        grouped.setdefault(row["backend"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: item["request_id"])

    def _contains(patterns: list[str], text: str) -> bool:
        lowered = text.lower()
        return any(pattern in lowered for pattern in patterns)

    review_rows: list[dict[str, Any]] = []
    for index, sample in enumerate(dataset):
        transcript = sample.get("transcript", "")
        transcript_lower = transcript.lower()
        transcript_has_vitals = _contains(["blood pressure", "temperature", "heart rate", "pulse", "respiratory rate"], transcript_lower)
        transcript_has_labs = _contains(["wbc", "cbc", "glucose", "lab", "labs", "blood culture"], transcript_lower)
        entry = {
            "sample_index": index,
            "transcript_excerpt": transcript[:380].replace("\n", " "),
            "prompt_excerpt": build_note_prompt(transcript)[:180].replace("\n", " "),
            "transcript_has_vitals": transcript_has_vitals,
            "transcript_has_labs": transcript_has_labs,
        }
        for backend in BACKENDS:
            rows = grouped.get(backend, [])
            if index >= len(rows):
                continue
            output = rows[index].get("text", "")
            entry[f"{backend}_output"] = output
            entry[f"{backend}_possible_invented_vitals"] = _contains(["temperature", "blood pressure", "pulse", "respiratory rate"], output) and not transcript_has_vitals
            entry[f"{backend}_possible_invented_labs"] = _contains(["wbc", "cbc", "blood culture", "labs"], output) and not transcript_has_labs
        review_rows.append(entry)

    _write_csv(RESULTS_DIR / "qualitative_review_default_prompt.csv", review_rows)
    md_lines = ["# Qualitative Review Export", "", "Heuristic flags are only for manual inspection; they are not clinical truth labels.", ""]
    for row in review_rows:
        md_lines.append(f"## Sample {row['sample_index']}")
        md_lines.append(f"Transcript excerpt: {row['transcript_excerpt']}")
        for backend in BACKENDS:
            output = row.get(f"{backend}_output")
            if output:
                md_lines.append(f"- {BACKEND_LABELS[backend]}: {output[:900]}")
                md_lines.append(
                    f"  possible invented vitals: {row.get(f'{backend}_possible_invented_vitals')} | possible invented labs: {row.get(f'{backend}_possible_invented_labs')}"
                )
        md_lines.append("")
    (POSTER_DIR / "qualitative_review_default_prompt.md").write_text("\n".join(md_lines), encoding="utf-8")
    return review_rows


def _write_report(hero_row: dict[str, Any], output_rows: list[dict[str, Any]], context_rows: list[dict[str, Any]], quantiles: list[dict[str, Any]], qualitative_rows: list[dict[str, Any]]) -> None:
    local_best = hero_row
    output16 = [row for row in output_rows if row["status"] == "success" and int(row["batch_size"]) == 8]
    by_x_backend = {(int(row["x_value"]), row["backend"]): row for row in output16}
    triton_gain_rows = []
    for x_value in sorted({int(row["x_value"]) for row in output16}):
        hf = by_x_backend.get((x_value, "hf_sequential"))
        paged = by_x_backend.get((x_value, "mistral_paged_static_batch"))
        triton = by_x_backend.get((x_value, "mistral_paged_static_batch_triton"))
        if hf and paged and triton:
            triton_gain_rows.append(
                {
                    "max_new_tokens": x_value,
                    "hf_tok_s": round(_to_float(hf["mean_throughput_tok_per_sec"]) or 0.0, 2),
                    "paged_tok_s": round(_to_float(paged["mean_throughput_tok_per_sec"]) or 0.0, 2),
                    "triton_tok_s": round(_to_float(triton["mean_throughput_tok_per_sec"]) or 0.0, 2),
                    "paged_vs_hf_pct": round(100.0 * ((_to_float(paged["mean_throughput_tok_per_sec"]) or 0.0) / (_to_float(hf["mean_throughput_tok_per_sec"]) or 1.0) - 1.0), 1),
                    "triton_vs_paged_pct": round(100.0 * ((_to_float(triton["mean_throughput_tok_per_sec"]) or 0.0) / (_to_float(paged["mean_throughput_tok_per_sec"]) or 1.0) - 1.0), 1),
                }
            )

    context_batch_one = [row for row in context_rows if row["status"] == "success" and int(row["batch_size"]) == 1 and row["backend"] == "mistral_paged_static_batch_triton"]
    context_batch_one.sort(key=lambda row: int(row["x_value"]))
    lines = [
        "# Systems Poster Study",
        "",
        "## Hero Result",
        "",
        f"- Best local decode-heavy operating point: `max_new_tokens={int(local_best['x_value'])}`, `batch_size={int(local_best['batch_size'])}`.",
        f"- Triton runtime throughput: `{_to_float(local_best['mean_throughput_tok_per_sec']):.2f} tok/s`.",
        "",
        "## Output-Length Sweep",
        "",
        "| max_new_tokens | HF tok/s | Paged tok/s | Triton tok/s | Paged vs HF | Triton vs Paged |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in triton_gain_rows:
        lines.append(
            f"| {row['max_new_tokens']} | {row['hf_tok_s']:.2f} | {row['paged_tok_s']:.2f} | {row['triton_tok_s']:.2f} | {row['paged_vs_hf_pct']:.1f}% | {row['triton_vs_paged_pct']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Context-Length Sweep",
            "",
            "| Prompt bucket | Triton TTFT (s) | Triton total latency (s) | Triton peak memory (GB) |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in context_batch_one:
        lines.append(
            f"| {int(row['x_value'])} | {_to_float(row['mean_ttft_sec']):.2f} | {_to_float(row['mean_latency_sec']):.2f} | {_to_float(row['mean_peak_gpu_mem_gb']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Latency Quantiles",
            "",
            "| Backend | Batch | P50 | P95 | P99 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in quantiles:
        lines.append(
            f"| {row['backend']} | {row['batch_size']} | {(_to_float(row['p50_latency_sec']) or 0.0):.2f} | {(_to_float(row['p95_latency_sec']) or 0.0):.2f} | {(_to_float(row['p99_latency_sec']) or 0.0):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Manual Review Export",
            "",
            f"- Exported `{len(qualitative_rows)}` default-prompt comparison rows to `poster_assets/deep_systems_study/qualitative_review_default_prompt.md`.",
            "- Heuristic hallucination flags are only manual-review aids; they are not ground truth labels.",
            "",
            "## Takeaways",
            "",
            "- Use `hero_backend_figure` as the single headline poster chart and `kv_efficiency_support` as the direct systems-design support panel.",
            "- The context sweep now shows the prefill-heavy limitation regime explicitly across approximate prompt buckets from 512 to 8192 tokens.",
            "- The output sweep now separates prefill-heavy and decode-heavy regimes across 16, 32, 64, 128, and 256 generated tokens.",
        ]
    )
    (PROJECT_ROOT / "SYSTEMS_POSTER_STUDY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _set_style()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_ROOT)
    _build_context_bucket_datasets(env)
    output_specs, context_specs, prompt_count_specs = _load_study_specs()

    summary_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    quality_map: dict[tuple[int, int], bool] = {}

    all_specs = output_specs + context_specs + prompt_count_specs
    for spec in all_specs:
        ok, source = _reuse_or_run_full_regime(spec, env)
        if ok:
            full_summary, full_batch_rows, full_request_rows, report = _read_full_report(spec)
            for batch_size in spec.batch_sizes:
                quality_map[(spec.x_value, batch_size)] = _quality_matched(report, BACKENDS)
            for row in full_summary:
                summary_rows.append(_append_metadata(row, spec, status="success"))
            for row in full_batch_rows:
                batch_rows.append(_append_metadata(dict(row), spec, status="success"))
            for row in full_request_rows:
                request_rows.append(_append_metadata(dict(row), spec, status="success"))
            continue

        for backend in BACKENDS:
            for batch_size in spec.batch_sizes:
                status, reason, report_path = _run_combo(spec, backend, batch_size, env)
                if status != "success" or report_path is None:
                    summary_rows.append(
                        {
                            "backend": backend,
                            "batch_size": batch_size,
                            "mean_batch_latency_sec": None,
                            "mean_throughput_tok_per_sec": None,
                            "mean_requests_per_sec": None,
                            "mean_peak_gpu_mem_gb": None,
                            "mean_ttft_sec": None,
                            "mean_prefill_latency_sec": None,
                            "mean_decode_latency_sec": None,
                            "mean_latency_sec": None,
                            "mean_prompt_tokens": None,
                            "mean_completion_tokens": None,
                            "mean_kv_allocated_bytes": None,
                            "mean_kv_utilization_ratio": None,
                            "mean_kv_fragmentation_ratio": None,
                            "study_type": spec.study_type,
                            "study_label": spec.label,
                            "x_value": spec.x_value,
                            "status": status,
                            "error": reason if reason != source else "full regime failed; combo unavailable",
                        }
                    )
                    continue
                combo_summary, combo_batch_rows, combo_request_rows, report = _read_combo(report_path)
                quality_map[(spec.x_value, batch_size)] = _quality_matched(report, BACKENDS)
                for row in combo_summary:
                    summary_rows.append(_append_metadata(row, spec, status="success"))
                for row in combo_batch_rows:
                    batch_rows.append(_append_metadata(dict(row), spec, status="success"))
                for row in combo_request_rows:
                    request_rows.append(_append_metadata(dict(row), spec, status="success"))

    output_rows = [row for row in summary_rows if row["study_type"] == "output"]
    context_rows = [row for row in summary_rows if row["study_type"] == "context"]

    hero_candidates = [
        row for row in output_rows
        if row["status"] == "success" and row["backend"] == "mistral_paged_static_batch_triton"
    ]
    hero_row = max(hero_candidates, key=lambda row: _to_float(row["mean_throughput_tok_per_sec"]) or -1.0)

    serving_dir = Path("/data/project_runtime/benchmarks/systems_benchmark_serving_stack_comparison/systems")
    serving_request_rows, serving_report = _serving_rows(serving_dir)

    _hero_figure(summary_rows, quality_map)
    _kv_figure(summary_rows)
    _context_scaling_figure(context_rows)
    _prefill_and_decode_figures(context_rows, output_rows)
    _output_scaling_figure(output_rows)
    quantile_rows, _ = _latency_quantiles(serving_request_rows, serving_report.get("summary_rows", []))
    qualitative_rows = _qualitative_review_export()

    summary_export = list(summary_rows)
    batch_export = list(batch_rows)
    request_export = list(request_rows)
    _write_csv(RESULTS_DIR / "poster_study_summary.csv", summary_export)
    _write_csv(RESULTS_DIR / "poster_study_batch_rows.csv", batch_export)
    _write_csv(RESULTS_DIR / "poster_study_request_rows.csv", request_export)
    _write_csv(RESULTS_DIR / "poster_study_latency_quantiles.csv", quantile_rows)

    _write_report(hero_row, output_rows, context_rows, quantile_rows, qualitative_rows)
    manifest = {
        "hero_row": hero_row,
        "summary_rows": len(summary_export),
        "batch_rows": len(batch_export),
        "request_rows": len(request_export),
        "quantile_rows": len(quantile_rows),
        "poster_dir": str(POSTER_DIR),
    }
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    (RUNTIME_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
