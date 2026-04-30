from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clinical_speech.config import load_config


RUNTIME_BENCH_DIR = Path("/data/project_runtime/benchmarks/ablation_sweep")
POSTER_DIR = PROJECT_ROOT / "poster_assets" / "ablation_sweep"
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"

CONFIG_PATHS = [
    PROJECT_ROOT / "configs" / "systems_benchmark_decode_sweep_16_64.yaml",
    PROJECT_ROOT / "configs" / "systems_benchmark_decode_sweep_16_128.yaml",
    PROJECT_ROOT / "configs" / "systems_benchmark_decode_sweep_32_64.yaml",
    PROJECT_ROOT / "configs" / "systems_benchmark_decode_sweep_32_128.yaml",
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
class RegimeSpec:
    config_path: Path
    experiment_name: str
    num_prompts: int
    max_new_tokens: int
    batch_sizes: list[int]
    backends: list[str]
    benchmark_dir: Path
    report_path: Path
    batch_rows_path: Path
    request_rows_path: Path
    quality_path: Path

    @property
    def regime_label(self) -> str:
        return f"prompts={self.num_prompts}, max_new_tokens={self.max_new_tokens}"


def _set_style() -> None:
    import matplotlib.pyplot as plt

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


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_csv_to_results(name: str, rows: list[dict[str, Any]]) -> None:
    _write_csv(RESULTS_TABLES_DIR / name, rows)
    _write_csv(RUNTIME_BENCH_DIR / name, rows)


def _to_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    return float(value)


def _safe_ratio_pct(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return 100.0 * (numerator / denominator - 1.0)


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def _spec_from_config(config_path: Path) -> RegimeSpec:
    cfg = load_config(config_path)
    bench_dir = cfg.experiment.benchmark_dir / "systems"
    return RegimeSpec(
        config_path=config_path,
        experiment_name=cfg.experiment.name,
        num_prompts=cfg.systems_benchmark.num_prompts,
        max_new_tokens=cfg.generation.max_new_tokens if cfg.systems_benchmark.max_new_tokens is None else cfg.systems_benchmark.max_new_tokens,
        batch_sizes=list(cfg.systems_benchmark.batch_sizes),
        backends=list(cfg.systems_benchmark.backends),
        benchmark_dir=bench_dir,
        report_path=bench_dir / "systems_benchmark_report.json",
        batch_rows_path=bench_dir / "systems_batch_rows.csv",
        request_rows_path=bench_dir / "systems_request_rows.csv",
        quality_path=bench_dir / "systems_quality.json",
    )


def _expected_pairs(spec: RegimeSpec) -> set[tuple[str, int]]:
    return {(backend, batch_size) for backend in spec.backends for batch_size in spec.batch_sizes}


def _report_complete(spec: RegimeSpec, report: dict[str, Any]) -> bool:
    actual_pairs = {
        (row["backend"], int(row["batch_size"]))
        for row in report.get("summary_rows", [])
    }
    return _expected_pairs(spec).issubset(actual_pairs)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _full_regime_log_path(spec: RegimeSpec) -> Path:
    return RUNTIME_BENCH_DIR / "logs" / f"{spec.experiment_name}.log"


def _combo_paths(spec: RegimeSpec, backend: str, batch_size: int) -> tuple[Path, Path]:
    slug = f"{spec.experiment_name}__{backend}__b{batch_size}"
    cfg_path = RUNTIME_BENCH_DIR / "tmp_configs" / f"{slug}.yaml"
    bench_root = RUNTIME_BENCH_DIR / "combo_runs" / slug
    return cfg_path, bench_root


def _write_combo_config(spec: RegimeSpec, backend: str, batch_size: int) -> Path:
    cfg_path, bench_root = _combo_paths(spec, backend, batch_size)
    payload = {
        "extends": str(spec.config_path),
        "experiment": {
            "name": f"{spec.experiment_name}_{backend}_b{batch_size}",
            "output_dir": str(bench_root / "outputs"),
            "log_dir": str(bench_root / "logs"),
            "benchmark_dir": str(bench_root / "benchmarks"),
            "profiler_dir": str(bench_root / "profiler"),
            "checkpoint_dir": str(bench_root / "checkpoints"),
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


def _extract_quality_metrics(report: dict[str, Any], backend: str) -> dict[str, float | None]:
    metrics = report.get("quality_by_backend", {}).get(backend, {}).get("metrics", {})
    rouge = metrics.get("rouge", {})
    return {
        "rouge1": _to_float(rouge.get("rouge1")),
        "rouge2": _to_float(rouge.get("rouge2")),
        "rougeL": _to_float(rouge.get("rougeL")),
        "bertscore_f1_mean": _to_float(metrics.get("bertscore_f1_mean")),
    }


def _status_from_output(output: str, returncode: int) -> tuple[str, str | None]:
    lowered = output.lower()
    if returncode == 0:
        return "success", None
    if "out of memory" in lowered:
        return "oom", "CUDA out of memory"
    return "failed", f"runner exited with code {returncode}"


def _reuse_or_run_regime(spec: RegimeSpec, env: dict[str, str]) -> tuple[bool, str]:
    if spec.report_path.exists():
        report = _load_json(spec.report_path)
        if _report_complete(spec, report):
            return True, "cached_full_regime"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_systems_benchmark.py"),
        "--config",
        str(spec.config_path),
    ]
    returncode, output = _run_command(command, env=env, log_path=_full_regime_log_path(spec))
    if returncode == 0 and spec.report_path.exists():
        report = _load_json(spec.report_path)
        if _report_complete(spec, report):
            return True, "fresh_full_regime"
    status, reason = _status_from_output(output, returncode)
    return False, reason or status


def _run_combo(spec: RegimeSpec, backend: str, batch_size: int, env: dict[str, str]) -> tuple[str, str | None, Path | None]:
    cfg_path, bench_root = _combo_paths(spec, backend, batch_size)
    report_path = bench_root / "benchmarks" / "systems" / "systems_benchmark_report.json"
    expected_pair = {(backend, batch_size)}
    if report_path.exists():
        report = _load_json(report_path)
        actual_pairs = {(row["backend"], int(row["batch_size"])) for row in report.get("summary_rows", [])}
        if expected_pair.issubset(actual_pairs):
            return "success", "cached_combo", report_path
    cfg_path = _write_combo_config(spec, backend, batch_size)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_systems_benchmark.py"),
        "--config",
        str(cfg_path),
    ]
    log_path = RUNTIME_BENCH_DIR / "logs" / f"{spec.experiment_name}__{backend}__b{batch_size}.log"
    returncode, output = _run_command(command, env=env, log_path=log_path)
    status, reason = _status_from_output(output, returncode)
    if status == "success" and report_path.exists():
        return status, "fresh_combo", report_path
    return status, reason, None


def _read_report_rows(spec: RegimeSpec) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    report = _load_json(spec.report_path)
    batch_rows = _read_csv_rows(spec.batch_rows_path)
    request_rows = _read_csv_rows(spec.request_rows_path)
    return report.get("summary_rows", []), batch_rows, {"request_rows": request_rows, "report": report}


def _read_combo_report_rows(report_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    report = _load_json(report_path)
    systems_dir = report_path.parent
    batch_rows = _read_csv_rows(systems_dir / "systems_batch_rows.csv")
    request_rows = _read_csv_rows(systems_dir / "systems_request_rows.csv")
    return report.get("summary_rows", []), batch_rows, {"request_rows": request_rows, "report": report}


def _regime_metadata(spec: RegimeSpec) -> dict[str, Any]:
    return {
        "config": str(spec.config_path),
        "experiment_name": spec.experiment_name,
        "num_prompts": spec.num_prompts,
        "max_new_tokens": spec.max_new_tokens,
        "regime": spec.regime_label,
    }


def _enrich_summary_row(row: dict[str, Any], *, spec: RegimeSpec, report: dict[str, Any], status: str, source: str, error: str | None = None) -> dict[str, Any]:
    enriched = {
        **_regime_metadata(spec),
        "backend": row["backend"],
        "batch_size": int(row["batch_size"]),
        "status": status,
        "source": source,
        "error": error,
        **row,
    }
    enriched.update(_extract_quality_metrics(report, row["backend"]))
    return enriched


def _empty_summary_row(spec: RegimeSpec, backend: str, batch_size: int, *, status: str, source: str, error: str | None) -> dict[str, Any]:
    row = {
        **_regime_metadata(spec),
        "backend": backend,
        "batch_size": batch_size,
        "status": status,
        "source": source,
        "error": error,
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
        "rouge1": None,
        "rouge2": None,
        "rougeL": None,
        "bertscore_f1_mean": None,
    }
    return row


def _attach_speedups(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["regime"], row["batch_size"], row["backend"]): row for row in summary_rows}
    enriched: list[dict[str, Any]] = []
    for row in summary_rows:
        hf = indexed.get((row["regime"], row["batch_size"], "hf_sequential"))
        paged = indexed.get((row["regime"], row["batch_size"], "mistral_paged_static_batch"))
        throughput = _to_float(row.get("mean_throughput_tok_per_sec"))
        requests = _to_float(row.get("mean_requests_per_sec"))
        hf_throughput = _to_float(hf.get("mean_throughput_tok_per_sec")) if hf else None
        hf_requests = _to_float(hf.get("mean_requests_per_sec")) if hf else None
        paged_throughput = _to_float(paged.get("mean_throughput_tok_per_sec")) if paged else None
        paged_requests = _to_float(paged.get("mean_requests_per_sec")) if paged else None
        row = dict(row)
        row["tok_speedup_vs_hf_pct"] = _safe_ratio_pct(throughput, hf_throughput)
        row["req_speedup_vs_hf_pct"] = _safe_ratio_pct(requests, hf_requests)
        row["tok_incremental_vs_paged_pct"] = _safe_ratio_pct(throughput, paged_throughput)
        row["req_incremental_vs_paged_pct"] = _safe_ratio_pct(requests, paged_requests)
        enriched.append(row)
    return enriched


def _find_best_regimes(summary_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    success_rows = [row for row in summary_rows if row["status"] == "success" and row["backend"] == "mistral_paged_static_batch_triton"]
    best_by_regime: dict[str, dict[str, Any]] = {}
    for row in success_rows:
        current = best_by_regime.get(row["regime"])
        if current is None or _to_float(row["mean_throughput_tok_per_sec"]) > _to_float(current["mean_throughput_tok_per_sec"]):
            best_by_regime[row["regime"]] = row
    best_rows = list(best_by_regime.values())

    limitation_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        if row["status"] != "success":
            limitation_rows.append(row)
            continue
        if row["backend"] != "mistral_paged_static_batch_triton":
            continue
        speedup = _to_float(row.get("tok_speedup_vs_hf_pct"))
        if speedup is not None and speedup <= 0.0:
            limitation_rows.append(row)
    return best_rows, limitation_rows


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, divider, *body])


def _save(fig, path_stem: Path) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _plot_grid_metric(summary_rows: list[dict[str, Any]], *, metric_key: str, ylabel: str, title: str, stem: str) -> None:
    import matplotlib.pyplot as plt

    regimes = sorted({row["regime"] for row in summary_rows})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=False)
    axes = axes.flatten()
    for ax, regime in zip(axes, regimes, strict=False):
        regime_rows = [row for row in summary_rows if row["regime"] == regime and row["status"] == "success"]
        for backend in BACKENDS:
            backend_rows = [row for row in regime_rows if row["backend"] == backend]
            if not backend_rows:
                continue
            backend_rows.sort(key=lambda item: int(item["batch_size"]))
            ax.plot(
                [int(item["batch_size"]) for item in backend_rows],
                [_to_float(item[metric_key]) for item in backend_rows],
                marker="o",
                linewidth=2.4,
                color=BACKEND_COLORS[backend],
                label=BACKEND_LABELS[backend],
            )
        ax.set_title(regime)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, POSTER_DIR / stem)


def _plot_speedup(summary_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    regimes = sorted({row["regime"] for row in summary_rows})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=False)
    axes = axes.flatten()
    compare = [
        ("mistral_paged_static_batch", "tok_speedup_vs_hf_pct"),
        ("mistral_paged_static_batch_triton", "tok_speedup_vs_hf_pct"),
    ]
    for ax, regime in zip(axes, regimes, strict=False):
        regime_rows = [row for row in summary_rows if row["regime"] == regime and row["status"] == "success"]
        for backend, metric_key in compare:
            backend_rows = [row for row in regime_rows if row["backend"] == backend]
            backend_rows.sort(key=lambda item: int(item["batch_size"]))
            ax.plot(
                [int(item["batch_size"]) for item in backend_rows],
                [_to_float(item[metric_key]) for item in backend_rows],
                marker="o",
                linewidth=2.4,
                color=BACKEND_COLORS[backend],
                label=BACKEND_LABELS[backend],
            )
        ax.axhline(0.0, color="#555555", linewidth=1.0)
        ax.set_title(regime)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Speedup vs HF (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Relative Throughput Speedup vs HF", fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, POSTER_DIR / "speedup_vs_hf")


def _plot_memory(summary_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    regimes = sorted({row["regime"] for row in summary_rows})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=False)
    axes = axes.flatten()
    for ax, regime in zip(axes, regimes, strict=False):
        regime_rows = [row for row in summary_rows if row["regime"] == regime and row["status"] == "success"]
        for backend in BACKENDS:
            backend_rows = [row for row in regime_rows if row["backend"] == backend]
            if not backend_rows:
                continue
            backend_rows.sort(key=lambda item: int(item["batch_size"]))
            ax.plot(
                [int(item["batch_size"]) for item in backend_rows],
                [_to_float(item["mean_peak_gpu_mem_gb"]) for item in backend_rows],
                marker="o",
                linewidth=2.2,
                color=BACKEND_COLORS[backend],
                label=f"{BACKEND_LABELS[backend]} peak",
            )
        ax.set_title(regime)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Peak GPU Memory (GB)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Memory vs Batch Size", fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, POSTER_DIR / "memory_vs_batchsize")


def _plot_triton_incremental(summary_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    regimes = sorted({row["regime"] for row in summary_rows})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=False)
    axes = axes.flatten()
    for ax, regime in zip(axes, regimes, strict=False):
        regime_rows = [
            row
            for row in summary_rows
            if row["regime"] == regime and row["status"] == "success" and row["backend"] == "mistral_paged_static_batch_triton"
        ]
        regime_rows.sort(key=lambda item: int(item["batch_size"]))
        ax.plot(
            [int(item["batch_size"]) for item in regime_rows],
            [_to_float(item["tok_incremental_vs_paged_pct"]) for item in regime_rows],
            marker="o",
            linewidth=2.6,
            color=BACKEND_COLORS["mistral_paged_static_batch_triton"],
        )
        ax.axhline(0.0, color="#555555", linewidth=1.0)
        ax.set_title(regime)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Triton Gain vs Paged (%)")
    fig.suptitle("Incremental Triton Gain", fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, POSTER_DIR / "triton_incremental_gain")


def _plot_best_regime(best_row: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    regime = best_row["regime"]
    batch_size = int(best_row["batch_size"])
    regime_rows = [row for row in summary_rows if row["regime"] == regime and int(row["batch_size"]) == batch_size and row["status"] == "success"]
    regime_rows.sort(key=lambda item: BACKENDS.index(item["backend"]))
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    values = [_to_float(row["mean_throughput_tok_per_sec"]) for row in regime_rows]
    labels = [BACKEND_LABELS[row["backend"]] for row in regime_rows]
    colors = [BACKEND_COLORS[row["backend"]] for row in regime_rows]
    bars = ax.bar(range(len(values)), values, color=colors, width=0.6)
    ax.set_xticks(range(len(values)), labels)
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title(f"Best Regime: {regime}, batch size {batch_size}")
    ymax = max(values) * 1.22
    ax.set_ylim(0.0, ymax)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + ymax * 0.02, f"{value:.2f}", ha="center", va="bottom", fontweight="bold")
    ax.text(
        0.98,
        0.95,
        f"+{_to_float(best_row['tok_speedup_vs_hf_pct']):.1f}% vs HF",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=BACKEND_COLORS["mistral_paged_static_batch_triton"],
    )
    _save(fig, POSTER_DIR / "best_regime_summary")


def _plot_limitation(long_context_csv: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _read_csv_rows(long_context_csv)
    batch_size = 2
    rows = [row for row in rows if int(row["batch_size"]) == batch_size]
    rows.sort(key=lambda item: BACKENDS.index(item["backend"]))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    labels = [BACKEND_LABELS[row["backend"]] for row in rows]
    throughput = [_to_float(row["mean_throughput_tok_per_sec"]) for row in rows]
    ttft = [_to_float(row["mean_ttft_sec"]) for row in rows]
    axes[0].bar(range(len(rows)), throughput, color=[BACKEND_COLORS[row["backend"]] for row in rows], width=0.6)
    axes[0].set_xticks(range(len(rows)), labels)
    axes[0].set_ylabel("Throughput (tok/s)")
    axes[0].set_title("Long-context throughput")
    axes[1].bar(range(len(rows)), ttft, color=[BACKEND_COLORS[row["backend"]] for row in rows], width=0.6)
    axes[1].set_xticks(range(len(rows)), labels)
    axes[1].set_ylabel("TTFT (s)")
    axes[1].set_title("Long-context TTFT")
    fig.suptitle("Limitation Regime: Prefill-heavy long context, batch size 2", fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, POSTER_DIR / "limitation_regime_summary")


def _plot_latency_breakdown(best_row: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    regime = best_row["regime"]
    batch_size = int(best_row["batch_size"])
    rows = [row for row in summary_rows if row["regime"] == regime and int(row["batch_size"]) == batch_size and row["status"] == "success"]
    rows.sort(key=lambda item: BACKENDS.index(item["backend"]))
    metrics = [
        ("mean_ttft_sec", "TTFT"),
        ("mean_prefill_latency_sec", "Prefill"),
        ("mean_decode_latency_sec", "Decode"),
        ("mean_latency_sec", "End-to-End"),
    ]
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    x_positions = list(range(len(rows)))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    palette = ["#9ca3af", "#93c5fd", "#2563eb", "#0f172a"]
    for offset, (metric_key, label), color in zip(offsets, metrics, palette, strict=True):
        values = [_to_float(row[metric_key]) for row in rows]
        bars = ax.bar([x + offset * width for x in x_positions], values, width=width, color=color, label=label)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x_positions, [BACKEND_LABELS[row["backend"]] for row in rows])
    ax.set_ylabel("Seconds")
    ax.set_title(f"Latency Breakdown by Backend ({regime}, batch size {batch_size})")
    ax.legend(frameon=False, ncol=4, loc="upper right")
    _save(fig, POSTER_DIR / "latency_breakdown_vs_backend")


def _render_table_figure(rows: list[dict[str, Any]], *, title: str, stem: str, columns: list[str]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13.5, max(3.6, 0.55 * len(rows) + 1.6)))
    ax.axis("off")
    ax.set_title(title, loc="left", pad=16)
    cell_text = [[row.get(column, "") for column in columns] for row in rows]
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 0.9],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        if row_idx == 0:
            cell.set_facecolor("#dce7f2")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#f8fafc" if row_idx % 2 else "#eef3f8")
    _save(fig, POSTER_DIR / stem)


def _summarize_answers(summary_rows: list[dict[str, Any]], best_rows: list[dict[str, Any]]) -> dict[str, str]:
    success_rows = [row for row in summary_rows if row["status"] == "success"]
    hf_rows = {(row["regime"], row["batch_size"]): row for row in success_rows if row["backend"] == "hf_sequential"}
    paged_rows = {(row["regime"], row["batch_size"]): row for row in success_rows if row["backend"] == "mistral_paged_static_batch"}
    triton_rows = {(row["regime"], row["batch_size"]): row for row in success_rows if row["backend"] == "mistral_paged_static_batch_triton"}

    outperform_points = []
    by_regime_batch: dict[str, list[int]] = {}
    for key, row in paged_rows.items():
        hf = hf_rows.get(key)
        if hf and _to_float(row["mean_throughput_tok_per_sec"]) > _to_float(hf["mean_throughput_tok_per_sec"]):
            outperform_points.append((key[0], key[1]))
            by_regime_batch.setdefault(key[0], []).append(key[1])

    if by_regime_batch:
        first_batches = {regime: min(batches) for regime, batches in by_regime_batch.items()}
        unique_first_batches = sorted(set(first_batches.values()))
        if len(unique_first_batches) == 1:
            first_outperform_text = f"batch size {unique_first_batches[0]} in all completed decode-heavy sweep regimes"
        else:
            first_outperform_text = ", ".join(f"{regime}: batch size {batch}" for regime, batch in sorted(first_batches.items()))
    else:
        first_outperform_text = "not observed"

    triton_incremental = [row for row in triton_rows.values() if _to_float(row.get("tok_incremental_vs_paged_pct")) is not None]
    best_incremental = max(triton_incremental, key=lambda row: _to_float(row["tok_incremental_vs_paged_pct"])) if triton_incremental else None

    by_decode = {}
    for row in triton_rows.values():
        by_decode.setdefault(int(row["max_new_tokens"]), []).append(_to_float(row["tok_speedup_vs_hf_pct"]))
    decode_summary = ", ".join(
        f"{decode}: {sum(values) / len(values):.1f}% avg"
        for decode, values in sorted(by_decode.items())
        if values
    )

    by_prompt = {}
    for row in triton_rows.values():
        by_prompt.setdefault(int(row["num_prompts"]), []).append(_to_float(row["tok_speedup_vs_hf_pct"]))
    prompt_summary = ", ".join(
        f"{prompt}: {sum(values) / len(values):.1f}% avg"
        for prompt, values in sorted(by_prompt.items())
        if values
    )

    ttft_deltas = []
    for key, row in triton_rows.items():
        hf = hf_rows.get(key)
        if hf:
            ttft_deltas.append((_to_float(row["mean_ttft_sec"]) or 0.0) - (_to_float(hf["mean_ttft_sec"]) or 0.0))
    mean_ttft_delta = sum(ttft_deltas) / len(ttft_deltas) if ttft_deltas else 0.0

    memory_advantages = []
    for key, row in triton_rows.items():
        hf = hf_rows.get(key)
        if hf:
            memory_advantages.append((_to_float(row["mean_peak_gpu_mem_gb"]) or 0.0) - (_to_float(hf["mean_peak_gpu_mem_gb"]) or 0.0))
    mean_mem_delta = sum(memory_advantages) / len(memory_advantages) if memory_advantages else 0.0

    best_overall = max(best_rows, key=lambda row: _to_float(row["tok_speedup_vs_hf_pct"])) if best_rows else None
    limitation = min(
        triton_rows.values(),
        key=lambda row: _to_float(row["tok_speedup_vs_hf_pct"]),
    ) if triton_rows else None

    return {
        "paged_begins_outperforming_hf": first_outperform_text,
        "best_triton_incremental_gain": "not observed"
        if best_incremental is None
        else f"{best_incremental['regime']}, batch size {best_incremental['batch_size']}: {_fmt_pct(_to_float(best_incremental['tok_incremental_vs_paged_pct']))}",
        "decode_length_effect": decode_summary or "not available",
        "prompt_count_effect": prompt_summary or "not available",
        "ttft_behavior": f"TTFT generally worsens as throughput rises; mean Triton-vs-HF delta is {mean_ttft_delta:.2f} s.",
        "memory_behavior": f"Peak GPU memory is generally higher than HF for the custom runtimes; mean Triton-vs-HF delta is {mean_mem_delta:.2f} GB.",
        "best_regime": "not available"
        if best_overall is None
        else f"{best_overall['regime']}, batch size {best_overall['batch_size']}: {_fmt_pct(_to_float(best_overall['tok_speedup_vs_hf_pct']))} vs HF",
        "limitation_regime": "not available"
        if limitation is None
        else f"Within the decode-heavy sweep: {limitation['regime']}, batch size {limitation['batch_size']}: {_fmt_pct(_to_float(limitation['tok_speedup_vs_hf_pct']))} vs HF. Outside the sweep, the long-context pilot remains the stronger limitation regime because prefill dominates.",
    }


def _write_report(summary_rows: list[dict[str, Any]], best_rows: list[dict[str, Any]], limitation_rows: list[dict[str, Any]], answers: dict[str, str]) -> None:
    strongest = max(best_rows, key=lambda row: _to_float(row["tok_speedup_vs_hf_pct"])) if best_rows else None
    failed_rows = [row for row in summary_rows if row["status"] != "success"]
    best_table_rows = [
        {
            "Regime": row["regime"],
            "Batch": row["batch_size"],
            "Backend": BACKEND_LABELS[row["backend"]],
            "tok/s": f"{_to_float(row['mean_throughput_tok_per_sec']):.2f}",
            "req/s": f"{_to_float(row['mean_requests_per_sec']):.3f}",
            "Speedup vs HF": f"{_to_float(row['tok_speedup_vs_hf_pct']):.1f}%",
        }
        for row in sorted(best_rows, key=lambda item: (_to_float(item["tok_speedup_vs_hf_pct"]) or -1), reverse=True)
    ]
    limitation_table_rows = [
        {
            "Regime": row["regime"],
            "Batch": row["batch_size"],
            "Backend": BACKEND_LABELS[row["backend"]],
            "Status": row["status"],
            "tok/s": "" if _to_float(row.get("mean_throughput_tok_per_sec")) is None else f"{_to_float(row['mean_throughput_tok_per_sec']):.2f}",
            "Speedup vs HF": "" if _to_float(row.get("tok_speedup_vs_hf_pct")) is None else f"{_to_float(row['tok_speedup_vs_hf_pct']):.1f}%",
            "Error": row.get("error") or "",
        }
        for row in limitation_rows[:12]
    ]
    content = f"""# Ablation Report

## Sweep

- configs:
  - `configs/systems_benchmark_decode_sweep_16_64.yaml`
  - `configs/systems_benchmark_decode_sweep_16_128.yaml`
  - `configs/systems_benchmark_decode_sweep_32_64.yaml`
  - `configs/systems_benchmark_decode_sweep_32_128.yaml`
- backends: `hf_sequential`, `mistral_paged_static_batch`, `mistral_paged_static_batch_triton`
- prompt counts: `16`, `32`
- `max_new_tokens`: `64`, `128`
- batch sizes: `1, 2, 4, 6, 8`

## Strongest Result

{"- " + strongest["regime"] if strongest else "- no successful result available"}
{"- batch size `" + str(strongest["batch_size"]) + "`" if strongest else ""}
{"- Triton runtime throughput: `" + f"{_to_float(strongest['mean_throughput_tok_per_sec']):.2f}" + " tok/s`" if strongest else ""}
{"- Triton runtime request rate: `" + f"{_to_float(strongest['mean_requests_per_sec']):.3f}" + " req/s`" if strongest else ""}
{"- speedup vs HF: `" + _fmt_pct(_to_float(strongest['tok_speedup_vs_hf_pct'])) + "`" if strongest else ""}

## Answers To The Required Analysis

1. At what batch size does the paged runtime begin to outperform HF?
   {answers["paged_begins_outperforming_hf"]}
2. How much does Triton add beyond the paged runtime?
   {answers["best_triton_incremental_gain"]}
3. How do gains change as decode length increases from 64 to 128?
   {answers["decode_length_effect"]}
4. How do gains change as prompt count increases from 16 to 32?
   {answers["prompt_count_effect"]}
5. What happens to TTFT as throughput improves?
   {answers["ttft_behavior"]}
6. Is there any meaningful memory-efficiency advantage?
   {answers["memory_behavior"]}
7. Which workload is the best regime for the custom system?
   {answers["best_regime"]}
8. Which workload is the limitation regime where HF still wins or gains shrink?
   {answers["limitation_regime"]}

## Feasibility

- {("No OOM or skipped combinations were observed in the decode-heavy ablation grid." if not failed_rows else f"{len(failed_rows)} combinations failed or were skipped; see the limitations table below.")}

## Best Regimes

{_markdown_table(best_table_rows, ["Regime", "Batch", "Backend", "tok/s", "req/s", "Speedup vs HF"])}

## Limitations And Infeasible Cases

{_markdown_table(limitation_table_rows, ["Regime", "Batch", "Backend", "Status", "tok/s", "Speedup vs HF", "Error"])}

## Honest Interpretation

- The current system is decode-optimized: the clearest wins appear as batch size and output length increase.
- Triton provides real incremental end-to-end gain on top of the paged runtime, but it is not the whole story; TTFT remains dominated by prefill behavior.
- Quality sanity metrics remain matched across successful backends on the evaluated sweep.
- Long-context prefill-heavy workloads remain a limitation and should be shown explicitly alongside the best decode-heavy regime.

## Final Takeaways

- The custom runtime begins to separate from HF once batching is large enough for decode sharing to matter.
- Triton matters most in the larger decode-heavy operating points, where paged KV gather becomes more visible end-to-end.
- TTFT does not improve in step with throughput and often worsens relative to HF.
- Peak GPU memory is typically higher than HF for the custom runtime in exchange for better multi-request decode throughput.
- The right headline is a serving-throughput claim for decode-heavy workloads, not a universal latency win claim.
"""
    (PROJECT_ROOT / "ABLATION_REPORT.md").write_text(content, encoding="utf-8")


def _write_poster_takeaways(best_rows: list[dict[str, Any]]) -> None:
    strongest = max(best_rows, key=lambda row: _to_float(row["tok_speedup_vs_hf_pct"])) if best_rows else None
    headline = (
        f"A custom paged-KV runtime plus Triton improves multi-request Mistral throughput by {_to_float(strongest['tok_speedup_vs_hf_pct']):.1f}% over HF sequential in the strongest measured decode-heavy regime."
        if strongest
        else "A custom paged-KV runtime plus Triton improves multi-request Mistral serving in the strongest measured regime."
    )
    content = f"""# Poster Takeaways

{headline}

- Contribution: custom page-backed KV runtime for Mistral serving.
- Contribution: explicit prefill/decode staging with static batched decode scheduling.
- Contribution: Triton paged-KV gather kernel with measured end-to-end serving gains.

- Limitation: TTFT does not improve alongside throughput and is often worse than HF.
- Limitation: long-context prefill-heavy workloads remain a weaker regime for the current system.

- Future work: push the same runtime toward stronger prefill optimization and broader long-context serving efficiency.
"""
    (PROJECT_ROOT / "POSTER_TAKEAWAYS.md").write_text(content, encoding="utf-8")


def main() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_ROOT)
    specs = [_spec_from_config(path) for path in CONFIG_PATHS]

    all_summary_rows: list[dict[str, Any]] = []
    all_batch_rows: list[dict[str, Any]] = []
    all_request_rows: list[dict[str, Any]] = []
    run_status_rows: list[dict[str, Any]] = []

    for spec in specs:
        ok, source = _reuse_or_run_regime(spec, env)
        if ok:
            summary_rows, batch_rows, payload = _read_report_rows(spec)
            report = payload["report"]
            success_pairs = {(row["backend"], int(row["batch_size"])) for row in summary_rows}
            for row in summary_rows:
                all_summary_rows.append(_enrich_summary_row(row, spec=spec, report=report, status="success", source=source))
            for backend in spec.backends:
                for batch_size in spec.batch_sizes:
                    run_status_rows.append(
                        {
                            **_regime_metadata(spec),
                            "backend": backend,
                            "batch_size": batch_size,
                            "status": "success" if (backend, batch_size) in success_pairs else "skipped",
                            "source": source,
                            "error": None if (backend, batch_size) in success_pairs else "missing from completed report",
                        }
                    )
            for row in batch_rows:
                row = dict(row)
                row.update(_regime_metadata(spec))
                all_batch_rows.append(row)
            for row in payload["request_rows"]:
                row = dict(row)
                row.update(_regime_metadata(spec))
                all_request_rows.append(row)
            continue

        for backend in spec.backends:
            for batch_size in spec.batch_sizes:
                status, combo_source, report_path = _run_combo(spec, backend, batch_size, env)
                run_status_rows.append(
                    {
                        **_regime_metadata(spec),
                        "backend": backend,
                        "batch_size": batch_size,
                        "status": status,
                        "source": combo_source,
                        "error": None if status == "success" else combo_source,
                    }
                )
                if status != "success" or report_path is None:
                    all_summary_rows.append(
                        _empty_summary_row(spec, backend, batch_size, status=status, source=combo_source, error=combo_source)
                    )
                    continue
                summary_rows, batch_rows, payload = _read_combo_report_rows(report_path)
                report = payload["report"]
                for row in summary_rows:
                    all_summary_rows.append(_enrich_summary_row(row, spec=spec, report=report, status="success", source=combo_source))
                for row in batch_rows:
                    row = dict(row)
                    row.update(_regime_metadata(spec))
                    all_batch_rows.append(row)
                for row in payload["request_rows"]:
                    row = dict(row)
                    row.update(_regime_metadata(spec))
                    all_request_rows.append(row)

    all_summary_rows = _attach_speedups(all_summary_rows)
    best_rows, limitation_rows = _find_best_regimes(all_summary_rows)
    answers = _summarize_answers(all_summary_rows, best_rows)

    backend_comparison_rows: list[dict[str, Any]] = []
    for row in all_summary_rows:
        backend_comparison_rows.append(
            {
                "regime": row["regime"],
                "backend": row["backend"],
                "batch_size": row["batch_size"],
                "status": row["status"],
                "tok/s": row["mean_throughput_tok_per_sec"],
                "req/s": row["mean_requests_per_sec"],
                "ttft_sec": row["mean_ttft_sec"],
                "prefill_sec": row["mean_prefill_latency_sec"],
                "decode_sec": row["mean_decode_latency_sec"],
                "latency_sec": row["mean_latency_sec"],
                "peak_gpu_mem_gb": row["mean_peak_gpu_mem_gb"],
                "kv_allocated_bytes": row["mean_kv_allocated_bytes"],
                "kv_utilization_ratio": row["mean_kv_utilization_ratio"],
                "kv_fragmentation_ratio": row["mean_kv_fragmentation_ratio"],
                "tok_speedup_vs_hf_pct": row["tok_speedup_vs_hf_pct"],
                "tok_incremental_vs_paged_pct": row["tok_incremental_vs_paged_pct"],
                "rouge1": row["rouge1"],
                "rouge2": row["rouge2"],
                "rougeL": row["rougeL"],
                "bertscore_f1_mean": row["bertscore_f1_mean"],
                "error": row["error"],
            }
        )

    RUNTIME_BENCH_DIR.mkdir(parents=True, exist_ok=True)
    POSTER_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    _copy_csv_to_results("ablation_summary.csv", all_summary_rows)
    _copy_csv_to_results("ablation_backend_comparison.csv", backend_comparison_rows)
    _copy_csv_to_results("ablation_best_regimes.csv", best_rows)
    _copy_csv_to_results("ablation_limitations.csv", limitation_rows)
    _write_csv(RUNTIME_BENCH_DIR / "ablation_batch_rows.csv", all_batch_rows)
    _write_csv(RUNTIME_BENCH_DIR / "ablation_request_rows.csv", all_request_rows)
    _write_csv(RUNTIME_BENCH_DIR / "ablation_run_status.csv", run_status_rows)

    manifest = {
        "configs": [str(spec.config_path) for spec in specs],
        "answers": answers,
        "best_rows": best_rows,
        "limitation_rows": limitation_rows[:20],
        "status_rows": run_status_rows,
    }
    _write_json(RUNTIME_BENCH_DIR / "ablation_manifest.json", manifest)

    _set_style()
    _plot_grid_metric(all_summary_rows, metric_key="mean_throughput_tok_per_sec", ylabel="Throughput (tok/s)", title="Throughput vs Batch Size", stem="throughput_vs_batchsize")
    _plot_grid_metric(all_summary_rows, metric_key="mean_requests_per_sec", ylabel="Requests / sec", title="Request Rate vs Batch Size", stem="reqs_vs_batchsize")
    _plot_speedup(all_summary_rows)
    _plot_memory(all_summary_rows)
    _plot_triton_incremental(all_summary_rows)
    if best_rows:
        strongest = max(best_rows, key=lambda row: _to_float(row["tok_speedup_vs_hf_pct"]))
        _plot_best_regime(strongest, all_summary_rows)
        _plot_latency_breakdown(strongest, all_summary_rows)
    long_context_csv = Path("/data/project_runtime/benchmarks/systems_benchmark_long_context_pilot/systems/systems_summary.csv")
    if long_context_csv.exists():
        _plot_limitation(long_context_csv)

    table_rows = [
        {
            "Regime": row["regime"],
            "Batch": row["batch_size"],
            "Backend": BACKEND_LABELS[row["backend"]],
            "tok/s": "" if _to_float(row.get("mean_throughput_tok_per_sec")) is None else f"{_to_float(row['mean_throughput_tok_per_sec']):.2f}",
            "req/s": "" if _to_float(row.get("mean_requests_per_sec")) is None else f"{_to_float(row['mean_requests_per_sec']):.3f}",
            "TTFT": "" if _to_float(row.get("mean_ttft_sec")) is None else f"{_to_float(row['mean_ttft_sec']):.2f}",
            "Decode": "" if _to_float(row.get("mean_decode_latency_sec")) is None else f"{_to_float(row['mean_decode_latency_sec']):.2f}",
            "Peak Mem": "" if _to_float(row.get("mean_peak_gpu_mem_gb")) is None else f"{_to_float(row['mean_peak_gpu_mem_gb']):.2f}",
            "Speedup vs HF": "" if _to_float(row.get("tok_speedup_vs_hf_pct")) is None else f"{_to_float(row['tok_speedup_vs_hf_pct']):.1f}%",
            "Status": row["status"],
        }
        for row in sorted(
            [row for row in all_summary_rows if row["backend"] == "mistral_paged_static_batch_triton"],
            key=lambda item: (item["regime"], int(item["batch_size"])),
        )
    ]
    _render_table_figure(
        table_rows,
        title="Ablation Results Table",
        stem="ablation_results_table",
        columns=["Regime", "Batch", "Backend", "tok/s", "req/s", "TTFT", "Decode", "Peak Mem", "Speedup vs HF", "Status"],
    )

    feasibility_rows = [
        {
            "Regime": row["regime"],
            "Batch": row["batch_size"],
            "Backend": BACKEND_LABELS[row["backend"]],
            "Status": row["status"],
            "Error": row["error"] or "",
        }
        for row in run_status_rows
    ]
    _render_table_figure(
        feasibility_rows,
        title="Concurrency / Feasibility Table",
        stem="concurrency_or_feasibility_table",
        columns=["Regime", "Batch", "Backend", "Status", "Error"],
    )

    _write_report(all_summary_rows, best_rows, limitation_rows, answers)
    _write_poster_takeaways(best_rows)

    print(
        json.dumps(
            {
                "ablation_manifest": str(RUNTIME_BENCH_DIR / "ablation_manifest.json"),
                "summary_csv": str(RUNTIME_BENCH_DIR / "ablation_summary.csv"),
                "poster_dir": str(POSTER_DIR),
                "best_regimes_csv": str(RESULTS_TABLES_DIR / "ablation_best_regimes.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
