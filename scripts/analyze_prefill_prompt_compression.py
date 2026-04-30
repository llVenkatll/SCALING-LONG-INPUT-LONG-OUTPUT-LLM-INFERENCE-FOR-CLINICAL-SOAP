from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/data/project")
RUNTIME_ROOT = Path("/data/project_runtime/benchmarks")
BASELINE_DIR = RUNTIME_ROOT / "systems_benchmark_long_context_pilot" / "systems"
COMPACT_DIR = RUNTIME_ROOT / "systems_benchmark_long_context_compact_prompt" / "systems"
ASSET_DIR = PROJECT_ROOT / "poster_assets" / "prefill_optimization"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _float(value: Any) -> float | None:
    if value in (None, "", "null", "None", "N/A"):
        return None
    return float(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _quality(report: dict[str, Any], backend: str) -> dict[str, Any]:
    metrics = report.get("quality_by_backend", {}).get(backend, {}).get("metrics", {})
    rouge = metrics.get("rouge") or {}
    return {
        "rouge1": rouge.get("rouge1"),
        "rouge2": rouge.get("rouge2"),
        "rougeL": rouge.get("rougeL"),
        "bertscore_f1_mean": metrics.get("bertscore_f1_mean"),
    }


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_comparison() -> list[dict[str, Any]]:
    baseline_rows = _read_csv(BASELINE_DIR / "systems_summary.csv")
    compact_rows = _read_csv(COMPACT_DIR / "systems_summary.csv")
    baseline_report = _read_json(BASELINE_DIR / "systems_benchmark_report.json")
    compact_report = _read_json(COMPACT_DIR / "systems_benchmark_report.json")
    baseline_by_key = {(row["backend"], row["batch_size"]): row for row in baseline_rows}
    compact_by_key = {(row["backend"], row["batch_size"]): row for row in compact_rows}

    rows: list[dict[str, Any]] = []
    for key, base in sorted(baseline_by_key.items()):
        compact = compact_by_key.get(key)
        if compact is None:
            continue
        backend, batch_size = key
        base_prompt = _float(base["mean_prompt_tokens"])
        compact_prompt = _float(compact["mean_prompt_tokens"])
        base_ttft = _float(base["mean_ttft_sec"])
        compact_ttft = _float(compact["mean_ttft_sec"])
        base_prefill = _float(base["mean_prefill_latency_sec"])
        compact_prefill = _float(compact["mean_prefill_latency_sec"])
        base_latency = _float(base["mean_latency_sec"])
        compact_latency = _float(compact["mean_latency_sec"])
        rows.append(
            {
                "backend": backend,
                "batch_size": int(float(batch_size)),
                "baseline_prompt_tokens": base_prompt,
                "compact_prompt_tokens": compact_prompt,
                "prompt_token_delta": None if base_prompt is None or compact_prompt is None else compact_prompt - base_prompt,
                "prompt_token_reduction_pct": None if base_prompt is None or compact_prompt is None else 100.0 * (1.0 - compact_prompt / base_prompt),
                "baseline_ttft_sec": base_ttft,
                "compact_ttft_sec": compact_ttft,
                "ttft_delta_sec": None if base_ttft is None or compact_ttft is None else compact_ttft - base_ttft,
                "ttft_improvement_pct": None if base_ttft is None or compact_ttft is None else 100.0 * (1.0 - compact_ttft / base_ttft),
                "baseline_prefill_sec": base_prefill,
                "compact_prefill_sec": compact_prefill,
                "prefill_delta_sec": None if base_prefill is None or compact_prefill is None else compact_prefill - base_prefill,
                "prefill_improvement_pct": None if base_prefill is None or compact_prefill is None else 100.0 * (1.0 - compact_prefill / base_prefill),
                "baseline_total_latency_sec": base_latency,
                "compact_total_latency_sec": compact_latency,
                "latency_delta_sec": None if base_latency is None or compact_latency is None else compact_latency - base_latency,
                "latency_improvement_pct": None if base_latency is None or compact_latency is None else 100.0 * (1.0 - compact_latency / base_latency),
                **{f"baseline_{k}": v for k, v in _quality(baseline_report, backend).items()},
                **{f"compact_{k}": v for k, v in _quality(compact_report, backend).items()},
            }
        )
    return rows


def make_figures(rows: list[dict[str, Any]]) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 170,
            "savefig.dpi": 320,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linestyle": "--",
            "legend.frameon": False,
        }
    )
    batch_one = [row for row in rows if row["batch_size"] == 1]
    labels = [row["backend"].replace("mistral_paged_static_batch_triton", "paged+triton").replace("mistral_paged_static_batch", "paged") for row in batch_one]
    x = list(range(len(batch_one)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.bar([i - width / 2 for i in x], [row["baseline_prompt_tokens"] for row in batch_one], width=width, color="#94a3b8", label="standard prompt")
    ax.bar([i + width / 2 for i in x], [row["compact_prompt_tokens"] for row in batch_one], width=width, color="#2563eb", label="compact prompt")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylabel("Prompt tokens")
    ax.set_title("Compact Prompt Barely Changes Long-Context Token Count")
    ax.legend()
    _save(fig, ASSET_DIR / "prompt_tokens_standard_vs_compact")

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.bar([i - width / 2 for i in x], [row["baseline_prefill_sec"] for row in batch_one], width=width, color="#f59e0b", label="standard prompt")
    ax.bar([i + width / 2 for i in x], [row["compact_prefill_sec"] for row in batch_one], width=width, color="#0d3b66", label="compact prompt")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylabel("Prefill latency (s)")
    ax.set_title("Prefill Does Not Improve Materially With Wrapper-Only Compression")
    ax.legend()
    _save(fig, ASSET_DIR / "prefill_standard_vs_compact")


def write_report(rows: list[dict[str, Any]]) -> None:
    best_ttft = max(rows, key=lambda row: row["ttft_improvement_pct"] if row["ttft_improvement_pct"] is not None else -999.0)
    worst_ttft = min(rows, key=lambda row: row["ttft_improvement_pct"] if row["ttft_improvement_pct"] is not None else 999.0)
    report = [
        "# Prefill Study Report",
        "",
        "## Goal",
        "Evaluate the simplest high-impact prefill optimization: a config-driven compact SOAP prompt that reduces instruction-wrapper tokens while keeping the transcript and task unchanged.",
        "",
        "## Implementation",
        "- Added `systems_benchmark.prompt_mode` with default `standard`.",
        "- Added `compact_soap` prompt mode using a shorter instruction wrapper.",
        "- Kept all baseline/runtime code paths intact.",
        "- Ran `configs/systems_benchmark_long_context_compact_prompt.yaml` against the existing long-context pilot workload.",
        "",
        "## Result",
        f"- Best TTFT change: `{best_ttft['backend']}` batch `{best_ttft['batch_size']}` changed by `{best_ttft['ttft_improvement_pct']:.2f}%`.",
        f"- Worst TTFT change: `{worst_ttft['backend']}` batch `{worst_ttft['batch_size']}` changed by `{worst_ttft['ttft_improvement_pct']:.2f}%`.",
        "- Prompt-wrapper compression reduced only a tiny fraction of total prompt tokens because transcript tokens dominate the long-context workload.",
        "",
        "## Conclusion",
        "This tested method does not help enough. The long-context bottleneck is the transcript prefill itself, not the SOAP instruction wrapper. A meaningful prefill improvement likely requires transcript compression/fact extraction before final note generation, prefix/template KV reuse, or chunked prefill scheduling.",
        "",
        "## Quality",
        "ROUGE/BERTScore are reported in the CSV table. Treat them as pilot sanity metrics because the long-context pilot has only two samples.",
        "",
        "## Artifacts",
        "- `/data/project/results/tables/prefill_prompt_compression_comparison.csv`",
        "- `/data/project/poster_assets/prefill_optimization/prompt_tokens_standard_vs_compact.png`",
        "- `/data/project/poster_assets/prefill_optimization/prefill_standard_vs_compact.png`",
    ]
    (PROJECT_ROOT / "PREFILL_STUDY_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_comparison()
    _write_csv(RESULTS_DIR / "prefill_prompt_compression_comparison.csv", rows)
    _write_csv(ASSET_DIR / "prefill_prompt_compression_comparison.csv", rows)
    make_figures(rows)
    write_report(rows)
    print(json.dumps({"rows": rows, "asset_dir": str(ASSET_DIR)}, indent=2))


if __name__ == "__main__":
    main()
