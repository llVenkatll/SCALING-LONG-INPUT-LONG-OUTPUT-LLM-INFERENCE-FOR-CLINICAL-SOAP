from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clinical_speech.config import BenchmarkConfig, GenerationConfig, load_config
from clinical_speech.data.dataset import load_dataset
from clinical_speech.evaluation.metrics import maybe_score_references
from clinical_speech.models.note_generator import NoteGenerator
from clinical_speech.pipeline.prompts import (
    build_clinical_facts_prompt,
    build_note_from_facts_prompt,
)
from clinical_speech.utils.io import write_json


BASELINE_DIR = Path("/data/project_runtime/benchmarks/systems_benchmark_long_context_pilot/systems")
ASSET_DIR = PROJECT_ROOT / "poster_assets" / "prefill_optimization"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "prefill_two_stage_long_context.yaml")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def _float(value: Any) -> float | None:
    if value in (None, "", "null", "None", "N/A"):
        return None
    return float(value)


def _quality_fields(report: dict[str, Any], backend: str = "hf_sequential") -> dict[str, Any]:
    metrics = report.get("quality_by_backend", {}).get(backend, {}).get("metrics", {})
    rouge = metrics.get("rouge") or {}
    return {
        "rouge1": rouge.get("rouge1"),
        "rouge2": rouge.get("rouge2"),
        "rougeL": rouge.get("rougeL"),
        "bertscore_f1_mean": metrics.get("bertscore_f1_mean"),
    }


def _baseline_row() -> tuple[dict[str, str], dict[str, Any]]:
    rows = _read_csv(BASELINE_DIR / "systems_summary.csv")
    report = _read_json(BASELINE_DIR / "systems_benchmark_report.json")
    for row in rows:
        if row["backend"] == "hf_sequential" and str(row["batch_size"]) == "1":
            return row, report
    raise RuntimeError("Could not find hf_sequential batch=1 baseline row")


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _make_figures(summary: dict[str, Any]) -> None:
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

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    labels = ["Baseline SOAP prompt", "Stage 1 facts prompt", "Stage 2 facts->SOAP prompt"]
    values = [
        summary["baseline_prompt_tokens"],
        summary["stage1_prompt_tokens"],
        summary["stage2_prompt_tokens"],
    ]
    colors = ["#94a3b8", "#f59e0b", "#2563eb"]
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_ylabel("Mean prompt tokens")
    ax.set_title("Two-Stage Compression Shrinks Final SOAP Prefill Prompt")
    ax.tick_params(axis="x", rotation=12)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0f}", ha="center", va="bottom", fontweight="bold")
    _save(fig, ASSET_DIR / "two_stage_prompt_token_reduction")

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    labels = ["Baseline", "Stage 2 only", "End-to-end two-stage"]
    ttft_values = [
        summary["baseline_ttft_sec"],
        summary["stage2_ttft_sec"],
        summary["two_stage_end_to_end_ttft_sec"],
    ]
    prefill_values = [
        summary["baseline_prefill_sec"],
        summary["stage2_prefill_sec"],
        summary["two_stage_end_to_end_prefill_path_sec"],
    ]
    x = list(range(len(labels)))
    width = 0.36
    ax.bar([i - width / 2 for i in x], ttft_values, width=width, color="#0d3b66", label="TTFT")
    ax.bar([i + width / 2 for i in x], prefill_values, width=width, color="#f59e0b", label="Prefill path")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Seconds")
    ax.set_title("Final-Stage TTFT Improves, But End-to-End Pipeline Adds Stage-1 Cost")
    ax.legend()
    _save(fig, ASSET_DIR / "two_stage_ttft_prefill_comparison")


def _write_report(summary: dict[str, Any], quality: dict[str, Any]) -> None:
    stage2_reduction = summary["stage2_prompt_token_reduction_pct"]
    stage2_ttft_gain = summary["stage2_ttft_improvement_pct"]
    e2e_ttft_gain = summary["two_stage_end_to_end_ttft_improvement_pct"]
    lines = [
        "# Prefill Study Report",
        "",
        "## Goal",
        "Evaluate transcript-to-facts compression as a higher-impact prefill optimization than wrapper-only prompt compression.",
        "",
        "## Method",
        "- Stage 1: long transcript -> compact structured clinical facts.",
        "- Stage 2: compact clinical facts -> SOAP note.",
        "- Existing long-context HF sequential baseline is left intact and used as the comparison point.",
        "- Stage-1 facts and Stage-2 SOAP notes are saved separately under `/data/project_runtime`.",
        "",
        "## Result",
        f"- Final SOAP-stage prompt tokens changed from `{summary['baseline_prompt_tokens']:.1f}` to `{summary['stage2_prompt_tokens']:.1f}` (`{stage2_reduction:.2f}%` reduction).",
        f"- Final SOAP-stage TTFT changed from `{summary['baseline_ttft_sec']:.4f}s` to `{summary['stage2_ttft_sec']:.4f}s` (`{stage2_ttft_gain:.2f}%` improvement).",
        f"- End-to-end two-stage TTFT-equivalent changed to `{summary['two_stage_end_to_end_ttft_sec']:.4f}s` (`{e2e_ttft_gain:.2f}%` vs baseline) after including Stage-1 extraction.",
        f"- End-to-end two-stage latency is `{summary['two_stage_total_latency_sec']:.4f}s` vs baseline `{summary['baseline_total_latency_sec']:.4f}s`.",
        "",
        "## Quality",
        f"- Two-stage ROUGE-1: `{quality.get('rouge1')}`",
        f"- Two-stage BERTScore F1: `{quality.get('bertscore_f1_mean')}`",
        "- Treat these as pilot sanity metrics because the long-context pilot has only two samples.",
        "",
        "## Conclusion",
    ]
    if e2e_ttft_gain and e2e_ttft_gain > 0:
        lines.append("Two-stage transcript compression improves the final SOAP prefill and also improves the end-to-end TTFT-equivalent on this pilot.")
    else:
        lines.append(
            "Two-stage transcript compression materially reduces the final SOAP prompt, but the extra Stage-1 generation cost outweighs the final-stage prefill savings in this pilot. "
            "This suggests the next useful design would need cheap/non-generative fact extraction, reusable facts, or amortized extraction across multiple downstream notes."
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "- `/data/project_runtime/benchmarks/prefill_two_stage_long_context/two_stage/two_stage_summary.csv`",
            "- `/data/project_runtime/benchmarks/prefill_two_stage_long_context/two_stage/stage1_facts.jsonl`",
            "- `/data/project_runtime/benchmarks/prefill_two_stage_long_context/two_stage/final_notes.jsonl`",
            "- `/data/project/poster_assets/prefill_optimization/two_stage_prompt_token_reduction.png`",
            "- `/data/project/poster_assets/prefill_optimization/two_stage_ttft_prefill_comparison.png`",
        ]
    )
    (PROJECT_ROOT / "PREFILL_STUDY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if cfg.systems_benchmark.prompt_mode != "two_stage_facts":
        raise SystemExit("This study expects systems_benchmark.prompt_mode: two_stage_facts")

    dataset = load_dataset(cfg.dataset.path, cfg.systems_benchmark.num_prompts)
    benchmark_cfg = BenchmarkConfig(enabled=False, synchronize_cuda=True)
    generator = NoteGenerator(
        cfg.model,
        GenerationConfig(max_new_tokens=cfg.systems_benchmark.stage1_max_new_tokens, temperature=0.0, top_p=1.0, do_sample=False),
    )

    output_dir = cfg.experiment.benchmark_dir / "two_stage"
    output_dir.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stage1_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    quality_predictions: list[dict[str, Any]] = []

    for index, sample in enumerate(dataset):
        transcript = sample.transcript or ""
        stage1_prompt = build_clinical_facts_prompt(transcript)
        generator.generation_config.max_new_tokens = cfg.systems_benchmark.stage1_max_new_tokens
        facts_result = generator.generate(stage1_prompt, benchmark_cfg)
        facts = facts_result.text

        stage2_prompt = build_note_from_facts_prompt(facts)
        generator.generation_config.max_new_tokens = cfg.systems_benchmark.stage2_max_new_tokens or cfg.generation.max_new_tokens
        note_result = generator.generate(stage2_prompt, benchmark_cfg)

        sample_id = sample.id or str(index)
        stage1_rows.append(
            {
                "id": sample_id,
                "sample_index": index,
                "facts": facts,
                "prompt_tokens": facts_result.prompt_tokens,
                "completion_tokens": facts_result.completion_tokens,
                "ttft_sec": facts_result.ttft_sec,
                "prefill_latency_sec": facts_result.prefill_latency_sec,
                "decode_latency_sec": facts_result.decode_latency_sec,
                "latency_sec": facts_result.latency_sec,
            }
        )
        final_rows.append(
            {
                "id": sample_id,
                "sample_index": index,
                "facts": facts,
                "soap_note": note_result.text,
                "stage2_prompt_tokens": note_result.prompt_tokens,
                "stage2_completion_tokens": note_result.completion_tokens,
                "stage2_ttft_sec": note_result.ttft_sec,
                "stage2_prefill_latency_sec": note_result.prefill_latency_sec,
                "stage2_decode_latency_sec": note_result.decode_latency_sec,
                "stage2_latency_sec": note_result.latency_sec,
                "end_to_end_ttft_sec": (facts_result.latency_sec + (note_result.ttft_sec or 0.0)),
                "end_to_end_prefill_path_sec": (facts_result.latency_sec + (note_result.prefill_latency_sec or 0.0)),
                "end_to_end_latency_sec": facts_result.latency_sec + note_result.latency_sec,
            }
        )
        request_rows.append(
            {
                "id": sample_id,
                "stage1_prompt_tokens": facts_result.prompt_tokens,
                "stage1_completion_tokens": facts_result.completion_tokens,
                "stage1_ttft_sec": facts_result.ttft_sec,
                "stage1_prefill_latency_sec": facts_result.prefill_latency_sec,
                "stage1_decode_latency_sec": facts_result.decode_latency_sec,
                "stage1_latency_sec": facts_result.latency_sec,
                "stage2_prompt_tokens": note_result.prompt_tokens,
                "stage2_completion_tokens": note_result.completion_tokens,
                "stage2_ttft_sec": note_result.ttft_sec,
                "stage2_prefill_latency_sec": note_result.prefill_latency_sec,
                "stage2_decode_latency_sec": note_result.decode_latency_sec,
                "stage2_latency_sec": note_result.latency_sec,
                "end_to_end_ttft_sec": facts_result.latency_sec + (note_result.ttft_sec or 0.0),
                "end_to_end_prefill_path_sec": facts_result.latency_sec + (note_result.prefill_latency_sec or 0.0),
                "end_to_end_latency_sec": facts_result.latency_sec + note_result.latency_sec,
            }
        )
        quality_predictions.append(
            {
                "id": sample_id,
                "reference_note": sample.reference_note,
                "predicted_note": note_result.text,
                "metadata": sample.metadata,
            }
        )

    baseline, baseline_report = _baseline_row()
    quality = maybe_score_references(quality_predictions)
    quality_fields = {
        "rouge1": (quality.get("metrics", {}).get("rouge") or {}).get("rouge1"),
        "rouge2": (quality.get("metrics", {}).get("rouge") or {}).get("rouge2"),
        "rougeL": (quality.get("metrics", {}).get("rouge") or {}).get("rougeL"),
        "bertscore_f1_mean": quality.get("metrics", {}).get("bertscore_f1_mean"),
    }

    def mean(key: str) -> float:
        values = [float(row[key]) for row in request_rows if row.get(key) is not None]
        return sum(values) / len(values)

    baseline_prompt = _float(baseline["mean_prompt_tokens"])
    baseline_ttft = _float(baseline["mean_ttft_sec"])
    baseline_prefill = _float(baseline["mean_prefill_latency_sec"])
    baseline_decode = _float(baseline["mean_decode_latency_sec"])
    baseline_latency = _float(baseline["mean_latency_sec"])

    summary = {
        "baseline_backend": "hf_sequential",
        "baseline_prompt_tokens": baseline_prompt,
        "baseline_ttft_sec": baseline_ttft,
        "baseline_prefill_sec": baseline_prefill,
        "baseline_decode_sec": baseline_decode,
        "baseline_total_latency_sec": baseline_latency,
        **{f"baseline_{key}": value for key, value in _quality_fields(baseline_report).items()},
        "stage1_prompt_tokens": mean("stage1_prompt_tokens"),
        "stage1_completion_tokens": mean("stage1_completion_tokens"),
        "stage1_ttft_sec": mean("stage1_ttft_sec"),
        "stage1_prefill_sec": mean("stage1_prefill_latency_sec"),
        "stage1_decode_sec": mean("stage1_decode_latency_sec"),
        "stage1_latency_sec": mean("stage1_latency_sec"),
        "stage2_prompt_tokens": mean("stage2_prompt_tokens"),
        "stage2_completion_tokens": mean("stage2_completion_tokens"),
        "stage2_ttft_sec": mean("stage2_ttft_sec"),
        "stage2_prefill_sec": mean("stage2_prefill_latency_sec"),
        "stage2_decode_sec": mean("stage2_decode_latency_sec"),
        "stage2_latency_sec": mean("stage2_latency_sec"),
        "two_stage_end_to_end_ttft_sec": mean("end_to_end_ttft_sec"),
        "two_stage_end_to_end_prefill_path_sec": mean("end_to_end_prefill_path_sec"),
        "two_stage_total_latency_sec": mean("end_to_end_latency_sec"),
        **{f"two_stage_{key}": value for key, value in quality_fields.items()},
    }
    summary["stage2_prompt_token_reduction_pct"] = 100.0 * (1.0 - summary["stage2_prompt_tokens"] / summary["baseline_prompt_tokens"])
    summary["stage2_ttft_improvement_pct"] = 100.0 * (1.0 - summary["stage2_ttft_sec"] / summary["baseline_ttft_sec"])
    summary["stage2_prefill_improvement_pct"] = 100.0 * (1.0 - summary["stage2_prefill_sec"] / summary["baseline_prefill_sec"])
    summary["two_stage_end_to_end_ttft_improvement_pct"] = 100.0 * (1.0 - summary["two_stage_end_to_end_ttft_sec"] / summary["baseline_ttft_sec"])
    summary["two_stage_total_latency_improvement_pct"] = 100.0 * (1.0 - summary["two_stage_total_latency_sec"] / summary["baseline_total_latency_sec"])

    _write_csv(output_dir / "two_stage_request_rows.csv", request_rows)
    _write_csv(output_dir / "two_stage_summary.csv", [summary])
    _write_csv(RESULTS_DIR / "prefill_two_stage_summary.csv", [summary])
    _write_csv(ASSET_DIR / "prefill_two_stage_summary.csv", [summary])
    for path, rows in [(output_dir / "stage1_facts.jsonl", stage1_rows), (output_dir / "final_notes.jsonl", final_rows)]:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    write_json(output_dir / "two_stage_quality.json", quality)
    write_json(output_dir / "two_stage_summary.json", summary)
    _make_figures(summary)
    _write_report(summary, quality_fields)
    print(json.dumps({"summary": summary, "quality": quality, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
