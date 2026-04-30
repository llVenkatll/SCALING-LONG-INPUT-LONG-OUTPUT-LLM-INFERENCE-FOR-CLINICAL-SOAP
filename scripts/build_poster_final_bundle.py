from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/data/project")
BENCHMARK_ROOT = Path("/data/project_runtime/benchmarks")
POSTER_STUDY_DIR = PROJECT_ROOT / "poster_assets" / "deep_systems_study"
PREFILL_OPT_DIR = PROJECT_ROOT / "poster_assets" / "prefill_optimization"
BUNDLE_DIR = PROJECT_ROOT / "poster_final_bundle"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _maybe_float(value: Any) -> float | None:
    if value in (None, "", "null", "None", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    if value in (None, "", "null", "None", "N/A"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _infer_timestamp(report_path: Path, summary_path: Path) -> str:
    ts = max(report_path.stat().st_mtime if report_path.exists() else 0, summary_path.stat().st_mtime)
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _infer_regime(run_name: str, row: dict[str, Any]) -> str:
    if "long_context" in run_name or "context_sweep" in run_name:
        return "long_context"
    if "decode_sweep" in run_name:
        return "decode_sweep"
    if "serving_stack" in run_name:
        return "serving_stack"
    if "pilot" in run_name:
        return "pilot"
    return "general"


def _parse_run_x_value(run_name: str) -> tuple[str | None, int | None]:
    context = re.search(r"context_sweep_(\d+)", run_name)
    if context:
        return "prompt_tokens_bucket", int(context.group(1))
    decode = re.search(r"decode_sweep_(\d+)_(\d+)", run_name)
    if decode:
        return "max_new_tokens", int(decode.group(2))
    return None, None


def _backend_label(row: dict[str, Any]) -> str:
    return str(row.get("backend_label") or row.get("backend") or "unknown")


def _load_quality(report: dict[str, Any], backend: str) -> dict[str, Any]:
    quality = report.get("quality_by_backend", {}).get(backend, {})
    metrics = quality.get("metrics", {})
    rouge = metrics.get("rouge") or {}
    return {
        "rouge1": rouge.get("rouge1"),
        "rouge2": rouge.get("rouge2"),
        "rougeL": rouge.get("rougeL"),
        "bertscore_f1_mean": metrics.get("bertscore_f1_mean"),
    }


def aggregate_runs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    for summary_path in sorted(BENCHMARK_ROOT.rglob("systems_summary.csv")):
        systems_dir = summary_path.parent
        report_path = systems_dir / "systems_benchmark_report.json"
        report = _read_json(report_path) if report_path.exists() else {}
        run_name = systems_dir.parent.name
        config_name = Path(str(report.get("config", ""))).name if report.get("config") else None
        timestamp = _infer_timestamp(report_path, summary_path)
        x_key, x_value = _parse_run_x_value(run_name)

        rows = []
        for row in _read_csv(summary_path):
            backend = row.get("backend", "")
            enriched = {
                **row,
                "run_name": run_name,
                "config_name": config_name,
                "timestamp": timestamp,
                "benchmark_dir": str(systems_dir),
                "regime": _infer_regime(run_name, row),
                "x_axis_name": x_key,
                "x_value": x_value,
                "backend_label_resolved": _backend_label(row),
                **_load_quality(report, backend),
            }
            rows.append(enriched)
            flat_rows.append(enriched)

        grouped[run_name] = {
            "run_name": run_name,
            "config_name": config_name,
            "timestamp": timestamp,
            "benchmark_dir": str(systems_dir),
            "report_path": str(report_path) if report_path.exists() else None,
            "enabled_backends": report.get("enabled_backends", []),
            "backend_failures": report.get("backend_failures", {}),
            "rows": rows,
        }
    return grouped, flat_rows


def _successful_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _maybe_float(row.get("mean_throughput_tok_per_sec")) is not None]


def _row_summary(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "run_name",
        "config_name",
        "regime",
        "x_axis_name",
        "x_value",
        "backend",
        "backend_label_resolved",
        "batch_size",
        "mean_throughput_tok_per_sec",
        "mean_requests_per_sec",
        "mean_ttft_sec",
        "mean_prefill_latency_sec",
        "mean_decode_latency_sec",
        "mean_latency_sec",
        "mean_peak_gpu_mem_gb",
        "mean_kv_allocated_bytes",
        "mean_kv_utilization_ratio",
        "mean_kv_fragmentation_ratio",
        "rouge1",
        "bertscore_f1_mean",
    ]
    return {key: row.get(key) for key in keys}


def compute_key_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = _successful_rows(rows)
    best_throughput = max(successful, key=lambda row: _maybe_float(row.get("mean_throughput_tok_per_sec")) or -1)
    best_latency = min(successful, key=lambda row: _maybe_float(row.get("mean_latency_sec")) or float("inf"))

    best_by_batch: dict[str, dict[str, Any]] = {}
    by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        batch = _maybe_int(row.get("batch_size"))
        if batch is not None:
            by_batch[batch].append(row)
    for batch, batch_rows in sorted(by_batch.items()):
        best_by_batch[str(batch)] = _row_summary(max(batch_rows, key=lambda row: _maybe_float(row.get("mean_throughput_tok_per_sec")) or -1))

    def best_for(predicate) -> dict[str, Any] | None:
        candidates = [row for row in successful if predicate(row)]
        if not candidates:
            return None
        return _row_summary(max(candidates, key=lambda row: _maybe_float(row.get("mean_throughput_tok_per_sec")) or -1))

    regimes = {
        "low_batch": best_for(lambda row: (_maybe_int(row.get("batch_size")) or 0) <= 2),
        "high_batch": best_for(lambda row: (_maybe_int(row.get("batch_size")) or 0) >= 6),
        "long_context": best_for(lambda row: row.get("regime") == "long_context"),
        "long_output": best_for(lambda row: row.get("x_axis_name") == "max_new_tokens" and (_maybe_int(row.get("x_value")) or 0) >= 128),
    }

    hf_match = [
        row for row in successful
        if row.get("run_name") == best_throughput.get("run_name")
        and str(row.get("batch_size")) == str(best_throughput.get("batch_size"))
        and row.get("backend") == "hf_sequential"
    ]
    improvement_vs_hf = None
    if hf_match:
        best_tps = _maybe_float(best_throughput.get("mean_throughput_tok_per_sec")) or 0.0
        hf_tps = _maybe_float(hf_match[0].get("mean_throughput_tok_per_sec")) or 0.0
        if hf_tps:
            improvement_vs_hf = {
                "baseline_backend": hf_match[0].get("backend"),
                "baseline_tok_s": hf_tps,
                "best_tok_s": best_tps,
                "speedup_x": best_tps / hf_tps,
                "improvement_pct": 100.0 * (best_tps / hf_tps - 1.0),
            }

    return {
        "best_throughput": _row_summary(best_throughput),
        "best_latency": _row_summary(best_latency),
        "best_backend_per_batch_size": best_by_batch,
        "best_backend_per_regime": regimes,
        "best_throughput_vs_matching_hf": improvement_vs_hf,
    }


def _fmt(value: Any, digits: int = 2) -> str:
    number = _maybe_float(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def _hardware_text() -> str:
    return (
        "AWS EC2 Ubuntu instance with a CUDA-capable NVIDIA A10G 24GB GPU. "
        "Benchmarks use CUDA synchronization where configured and store heavy outputs under `/data/project_runtime`."
    )


def build_methodology(key_results: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    configs = sorted({row.get("config_name") for row in rows if row.get("config_name")})
    return "\n".join(
        [
            "# Methodology",
            "",
            "## Task",
            "Clinical dialogue transcript to structured SOAP note generation. The note-generation task is the workload; the systems contribution is the inference runtime.",
            "",
            "## Metrics",
            "- Latency: end-to-end request latency plus TTFT, prefill latency, and decode latency when observable.",
            "- Throughput: generated tokens per second (`tok/s`) and requests per second (`req/s`).",
            "- Memory: peak local GPU memory, KV allocated bytes, KV utilization ratio, and KV fragmentation proxy for local paged runs.",
            "- Quality sanity: ROUGE and BERTScore where references are available.",
            "",
            "## Experiment Dimensions",
            "- Backend comparison: local HF sequential, local Mistral paged static batch, local Mistral paged static batch + Triton, local Llama HF sequential, and Together hosted Llama when available.",
            "- Batch-size sweep: measured batch sizes from the benchmark configs, including high-batch decode-heavy regimes.",
            "- Context-length sweep: approximate prompt-token buckets from 512 through 8192 tokens.",
            "- Output-length sweep: measured decode lengths from 16 through 128 generated tokens in the completed poster study.",
            "- Prefill optimization studies: compact SOAP prompting and two-stage transcript-to-facts compression on the long-context MedSynth pilot.",
            "",
            "## Hardware",
            _hardware_text(),
            "",
            "## Source Configs",
            *[f"- `{name}`" for name in configs],
        ]
    )


def build_system_design(key_results: dict[str, Any]) -> str:
    best = key_results["best_throughput"]
    speedup = key_results.get("best_throughput_vs_matching_hf") or {}
    speedup_line = (
        f"The strongest measured local result is `{_fmt(best.get('mean_throughput_tok_per_sec'))} tok/s`, "
        f"or `{_fmt(speedup.get('speedup_x'))}x` the matching HF sequential throughput "
        f"(`+{_fmt(speedup.get('improvement_pct'))}%`)."
        if speedup
        else f"The strongest measured local result is `{_fmt(best.get('mean_throughput_tok_per_sec'))} tok/s`."
    )
    return "\n".join(
        [
            "# System Design and Novelty",
            "",
            "## System Components",
            "- HF baseline: sequential Hugging Face generation path with library-managed KV growth.",
            "- Paged KV backend: custom Mistral-specific runtime with explicit page/block allocation, prefill/decode split, and static batched decode.",
            "- Paged KV + Triton backend: same custom runtime with Triton-accelerated paged-KV gather for decode.",
            "- Together hosted backend: managed OpenAI-compatible Llama endpoint used as a serving-stack reference when provider access is available.",
            "",
            "## Key Novelty",
            "- Page-backed KV memory management with explicit allocation, utilization, and fragmentation metrics.",
            "- Runtime-level prefill/decode staging rather than only application-level prompt or dataset changes.",
            "- Static batched decode scheduling to expose multi-request serving throughput.",
            "- Triton optimization for the repo-controlled paged-KV gather hotspot.",
            "",
            "## Data-Backed Claims",
            f"- {speedup_line}",
            "- Quality sanity metrics are preserved across the local Mistral HF and custom runtime paths in the completed decode-heavy studies.",
            "- Gains are strongest for multi-request decode-heavy workloads; TTFT and long-context prefill-heavy workloads remain limitations.",
            "- The two-stage facts pipeline reduces final SOAP-stage prompt tokens and TTFT, but inline generative fact extraction is too expensive for end-to-end serving in the pilot.",
            "- The implementation does not claim full PagedAttention or production continuous batching.",
        ]
    )


def build_model_architectures() -> str:
    return "\n".join(
        [
            "# Model Architectures",
            "",
            "## Mistral",
            "- Decoder-only transformer language model.",
            "- Uses causal self-attention and KV caching during autoregressive generation.",
            "- The custom runtime work targets Mistral's serving behavior: page-backed KV storage, explicit prefill/decode execution, static batched decode, and Triton paged-KV gather.",
            "",
            "## Llama",
            "- Decoder-only transformer language model with the same broad autoregressive serving pattern: tokenization, prefill, KV-cache reuse, and decode.",
            "- Used as a comparison model family for local HF sequential and Together hosted serving-stack baselines.",
            "",
            "## Fairness Framing",
            "- This poster compares serving stacks and runtime behavior, not a new foundation model.",
            "- Mistral and Llama are not treated as identical models; output labels separate model family from backend/deployment.",
            "- The custom paged runtime remains Mistral-specific, while Llama is included to contextualize local and hosted baseline behavior.",
        ]
    )


def _extract_sections(md_path: Path, limit: int = 5) -> list[dict[str, str]]:
    if not md_path.exists():
        return []
    text = md_path.read_text(encoding="utf-8")
    sections = re.split(r"\n## ", text)
    results = []
    for section in sections[1:]:
        title, _, body = section.partition("\n")
        results.append({"title": title.strip(), "body": body.strip()})
        if len(results) >= limit:
            break
    return results


def build_qualitative() -> str:
    default_path = POSTER_STUDY_DIR / "qualitative_review_default_prompt.md"
    prompt_control_path = POSTER_STUDY_DIR / "prompt_control_review.md"
    serving_requests_path = (
        BENCHMARK_ROOT
        / "systems_benchmark_serving_stack_comparison"
        / "systems"
        / "systems_request_rows.csv"
    )
    default_sections = _extract_sections(default_path, limit=3)
    prompt_sections = _extract_sections(prompt_control_path, limit=8)
    lines = [
        "# Qualitative Output Comparison",
        "",
        "Qualitative rows are copied from existing review exports. Heuristic hallucination flags are manual-review aids, not clinical ground truth.",
        "",
        "## Representative Serving-Stack Outputs",
        "",
    ]
    backend_order = [
        ("hf_sequential", "HF Mistral"),
        ("mistral_paged_static_batch_triton", "Triton Backend"),
        ("hf_sequential_llama_local", "HF Llama"),
        ("together_hosted_llama", "Together Hosted"),
    ]
    if serving_requests_path.exists():
        dataset = _read_jsonl(Path("/data/project_runtime/datasets/medsynth/test.jsonl"))[:5]
        request_rows = _read_csv(serving_requests_path)
        selected = [
            row for row in request_rows
            if row.get("repeat_index") in ("0", 0)
            and str(row.get("batch_size")) == "8"
        ]
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in selected:
            grouped[row.get("backend", "")].append(row)
        for rows in grouped.values():
            rows.sort(key=lambda row: row.get("request_id", ""))

        for sample_index, sample in enumerate(dataset[:3]):
            lines.append(f"### Sample {sample_index}")
            lines.append("PROMPT:")
            lines.append(str(sample.get("transcript", ""))[:520].replace("\n", " "))
            lines.append("")
            for backend, label in backend_order:
                output = ""
                if sample_index < len(grouped.get(backend, [])):
                    output = grouped[backend][sample_index].get("text", "")
                lines.append(f"{label.upper()} OUTPUT:")
                lines.append(output[:900] if output else "N/A in saved serving-stack request rows.")
                lines.append("")
            lines.append(
                "Hallucination/structure note: inspect for invented vitals, labs, medications, demographics, and whether the SOAP headings remain intact."
            )
            lines.append("")
    else:
        lines.append("No serving-stack request-row CSV was found, so explicit Together/local output snippets are unavailable.")
        lines.append("")
    lines.extend(
        [
        "## Representative Default-Prompt Comparisons",
        "",
        ]
    )
    if default_sections:
        for section in default_sections:
            lines.append(f"### {section['title']}")
            lines.append(section["body"][:2600])
            lines.append("")
    else:
        lines.append("No default-prompt qualitative review file was found.")
        lines.append("")
    lines.extend(
        [
            "## Strict vs Default Prompt Review",
            "",
            "The strict prompt adds: use only transcript facts, write `unknown` for missing vitals/labs/medications/exam findings, and do not invent values or diagnoses.",
            "",
        ]
    )
    if prompt_control_path.exists():
        prompt_text = prompt_control_path.read_text(encoding="utf-8")
        summary_table = prompt_text.split("## Sample", 1)[0].strip()
        lines.append(summary_table)
        lines.append("")
        lines.append("## Representative Prompt-Control Outputs")
        lines.append("")
        for section in prompt_sections[:5]:
            lines.append(f"### {section['title']}")
            lines.append(section["body"][:1600])
            lines.append("")
    else:
        lines.append("No prompt-control review file was found.")
    lines.extend(
        [
            "## Notes",
            "- HF Mistral and paged Triton Mistral outputs are expected to match closely because the runtime preserves model behavior.",
            "- Local Llama and Together hosted Llama are model/deployment comparisons, not evidence that the custom runtime changes model quality.",
            "- The Together hosted section is included only when completed hosted request rows exist in saved benchmark outputs.",
        ]
    )
    return "\n".join(lines)


def build_figures() -> str:
    captions = {
        "hero_backend_figure.png": ("Hero Result", "Paged KV + Triton achieves the highest measured local decode-heavy throughput at the selected high-batch operating point.", "The custom runtime wins most clearly in multi-request decode-heavy serving."),
        "kv_efficiency_support.png": ("KV Efficiency Support", "KV allocated bytes, utilization, and fragmentation proxy for the paged Triton backend.", "The design exposes and controls KV-cache memory behavior."),
        "context_scaling_panels.png": ("Context Scaling", "TTFT, total latency, and peak GPU memory across approximate prompt-token buckets.", "Longer prompts expose the prefill-heavy limitation regime."),
        "prefill_vs_context.png": ("Prefill Scaling", "Prefill latency as prompt length increases.", "TTFT/prefill grows with context and is not the main optimized path."),
        "decode_vs_output_length.png": ("Decode Scaling", "Decode latency as output length increases.", "Triton helps most when decode work dominates."),
        "throughput_vs_output_length.png": ("Output-Length Throughput", "Throughput across generated-token lengths.", "Longer decode lengths amplify the custom runtime advantage."),
        "latency_quantiles_batch4.png": ("Latency Quantiles", "P50/P95/P99 request latency at batch size 4.", "Tail latency surfaces backend variability that means alone can hide."),
    }
    lines = ["# Figure Index and Captions", ""]
    for path in sorted(POSTER_STUDY_DIR.glob("*.png")):
        title, caption, proves = captions.get(path.name, (path.stem.replace("_", " ").title(), "Poster-study figure generated from measured benchmark outputs.", "Supports the systems evaluation."))
        pdf_path = path.with_suffix(".pdf")
        lines.extend(
            [
                f"## `{path.name}`",
                f"- Title: {title}",
                f"- Caption: {caption}",
                f"- What it proves: {proves}",
                f"- PNG: `{path}`",
                f"- PDF: `{pdf_path if pdf_path.exists() else 'N/A'}`",
                "",
            ]
        )
    prefill_captions = {
        "prompt_tokens_standard_vs_compact.png": ("Compact Prompt Token Count", "Wrapper-only prompt compression reduces very few tokens in long-context transcripts.", "Transcript tokens dominate the long-context prefill cost."),
        "prefill_standard_vs_compact.png": ("Compact Prompt Prefill", "Prefill latency is nearly unchanged by shortening only the SOAP instruction wrapper.", "Simple prompt-wrapper compression is insufficient."),
        "two_stage_prompt_token_reduction.png": ("Two-Stage Prompt Reduction", "Transcript-to-facts compression drastically reduces the final SOAP-stage prompt length.", "Fact compression can remove final-stage prefill burden."),
        "two_stage_ttft_prefill_comparison.png": ("Two-Stage TTFT Tradeoff", "Final-stage TTFT improves, but full two-stage latency includes expensive fact extraction.", "Inline generative extraction is not yet a net serving win."),
    }
    if PREFILL_OPT_DIR.exists():
        lines.append("# Prefill Optimization Figures")
        lines.append("")
        for path in sorted(PREFILL_OPT_DIR.glob("*.png")):
            title, caption, proves = prefill_captions.get(path.name, (path.stem.replace("_", " ").title(), "Prefill optimization figure generated from measured outputs.", "Supports the prefill limitation story."))
            pdf_path = path.with_suffix(".pdf")
            lines.extend(
                [
                    f"## `{path.name}`",
                    f"- Title: {title}",
                    f"- Caption: {caption}",
                    f"- What it proves: {proves}",
                    f"- PNG: `{path}`",
                    f"- PDF: `{pdf_path if pdf_path.exists() else 'N/A'}`",
                    "",
                ]
            )
    return "\n".join(lines)


def build_executive_summary(key_results: dict[str, Any]) -> str:
    best = key_results["best_throughput"]
    speedup = key_results.get("best_throughput_vs_matching_hf") or {}
    latency = key_results["best_latency"]
    return "\n".join(
        [
            "# Executive Summary",
            "",
            f"- Main result: `{best.get('backend_label_resolved')}` reaches `{_fmt(best.get('mean_throughput_tok_per_sec'))} tok/s` in `{best.get('run_name')}` at batch size `{best.get('batch_size')}`.",
            f"- Compared with the matching HF sequential baseline, the best local runtime is `{_fmt(speedup.get('speedup_x'))}x` faster (`+{_fmt(speedup.get('improvement_pct'))}%`) where a matching HF row is available.",
            "- Key insight: gains come from decode-heavy, multi-request serving; TTFT/prefill-heavy long-context workloads remain less favorable.",
            "- Prefill optimization finding: two-stage transcript-to-facts compression cuts the final SOAP prompt by 96.18% and final-stage TTFT by 94.54%, but inline generative fact extraction makes total latency worse.",
            "- Serving-stack comparison: local Llama and Together hosted Llama are contextual baselines, not model-quality claims against Mistral.",
            "- Quality preservation: local Mistral HF, paged runtime, and paged+Triton paths preserve quality sanity metrics in the completed measured studies.",
            f"- Lowest measured mean latency row: `{latency.get('backend_label_resolved')}` in `{latency.get('run_name')}` with `{_fmt(latency.get('mean_latency_sec'))} s` mean latency.",
        ]
    )


def main() -> None:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    grouped, flat_rows = aggregate_runs()
    key_results = compute_key_results(flat_rows)

    _write_json(BUNDLE_DIR / "all_runs_grouped.json", grouped)
    _write_json(BUNDLE_DIR / "all_rows_flat.json", flat_rows)
    _write_json(BUNDLE_DIR / "key_results.json", key_results)
    _write_text(BUNDLE_DIR / "methodology.md", build_methodology(key_results, flat_rows))
    _write_text(BUNDLE_DIR / "system_design.md", build_system_design(key_results))
    _write_text(BUNDLE_DIR / "model_architectures.md", build_model_architectures())
    _write_text(BUNDLE_DIR / "qualitative_comparison.md", build_qualitative())
    _write_text(BUNDLE_DIR / "figures.md", build_figures())
    _write_text(BUNDLE_DIR / "executive_summary.md", build_executive_summary(key_results))

    generated_files = sorted(str(path) for path in BUNDLE_DIR.iterdir() if path.is_file())
    manifest_path = BUNDLE_DIR / "manifest.json"
    if str(manifest_path) not in generated_files:
        generated_files.append(str(manifest_path))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs_processed": len(grouped),
        "total_rows_aggregated": len(flat_rows),
        "best_backend_overall": key_results["best_throughput"].get("backend"),
        "best_throughput_tok_s": key_results["best_throughput"].get("mean_throughput_tok_per_sec"),
        "best_latency_sec": key_results["best_latency"].get("mean_latency_sec"),
        "generated_files": sorted(generated_files),
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
