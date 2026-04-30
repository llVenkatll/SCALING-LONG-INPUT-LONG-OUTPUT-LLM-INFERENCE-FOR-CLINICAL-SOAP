import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clinical_speech.benchmarking import aggregate_numeric_records, write_csv_rows
from clinical_speech.config import load_config
from clinical_speech.data.dataset import load_dataset
from clinical_speech.evaluation.metrics import maybe_score_references
from clinical_speech.kernels.paged_kv import TRITON_PAGED_KV_RUNTIME_USABLE
from clinical_speech.models.hosted_together import TogetherHostedGenerator
from clinical_speech.models.note_generator import NoteGenerator
from clinical_speech.pipeline.prompts import build_prompt_for_mode
from clinical_speech.runtime.engine import ManualBatchEngine
from clinical_speech.runtime.scheduler import StaticBatchScheduler
from clinical_speech.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def _release_backend_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _group_prompts(prompts: list[str], batch_size: int) -> list[list[str]]:
    scheduler = StaticBatchScheduler(batch_size)
    return [batch.prompts for batch in scheduler.schedule(prompts)]


def _write_summary_figures(summary_rows: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    by_backend: dict[str, list[dict]] = {}
    for row in summary_rows:
        by_backend.setdefault(row["backend"], []).append(row)

    def _plot(metric_key: str, filename: str, ylabel: str) -> None:
        plt.figure(figsize=(7, 4))
        for backend, rows in sorted(by_backend.items()):
            ordered = sorted(rows, key=lambda item: int(item["batch_size"]))
            label = ordered[0].get("backend_label", backend) if ordered else backend
            plt.plot(
                [int(item["batch_size"]) for item in ordered],
                [float(item[metric_key]) for item in ordered],
                marker="o",
                label=label,
            )
        plt.xlabel("Batch size")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    _plot("mean_throughput_tok_per_sec", "systems_throughput_vs_batch.png", "Throughput (tok/s)")
    _plot("mean_requests_per_sec", "systems_requests_vs_batch.png", "Requests / sec")


def _aggregate_request_summaries(request_rows: list[dict], *, backend: str, batch_size: int) -> dict[str, float | None]:
    rows = [row for row in request_rows if row["backend"] == backend and row["batch_size"] == batch_size]
    if not rows:
        return {}
    summary = aggregate_numeric_records(rows)
    return {
        "mean_ttft_sec": summary.get("ttft_sec", {}).get("mean"),
        "mean_prefill_latency_sec": summary.get("prefill_latency_sec", {}).get("mean"),
        "mean_decode_latency_sec": summary.get("decode_latency_sec", {}).get("mean"),
        "mean_latency_sec": summary.get("latency_sec", {}).get("mean"),
        "mean_peak_gpu_mem_gb": summary.get("peak_gpu_mem_gb", {}).get("mean"),
        "mean_prompt_tokens": summary.get("prompt_tokens", {}).get("mean"),
        "mean_completion_tokens": summary.get("completion_tokens", {}).get("mean"),
    }


def _infer_model_family(model_id: str) -> str:
    lowered = model_id.lower()
    if "mistral" in lowered:
        return "Mistral"
    if "llama" in lowered:
        return "Llama"
    return model_id


def _default_backend_metadata(cfg, backend_name: str, *, model_id: str) -> dict[str, str]:
    model_family = _infer_model_family(model_id)
    defaults = {
        "hf_sequential": {
            "backend_label": f"HF Sequential ({model_family}, local)",
            "provider": "local_hf",
            "deployment": "local",
            "serving_stack": "hf_sequential",
        },
        "mistral_paged_single": {
            "backend_label": f"Paged Single ({model_family}, local)",
            "provider": "local_runtime",
            "deployment": "local",
            "serving_stack": "paged_single",
        },
        "mistral_paged_static_batch": {
            "backend_label": f"Paged Static Batch ({model_family}, local)",
            "provider": "local_runtime",
            "deployment": "local",
            "serving_stack": "paged_static_batch",
        },
        "mistral_paged_static_batch_triton": {
            "backend_label": f"Paged Static Batch + Triton ({model_family}, ours)",
            "provider": "local_runtime",
            "deployment": "local",
            "serving_stack": "paged_static_batch_triton",
        },
    }
    return {
        "model_family": model_family,
        **defaults.get(
            backend_name,
            {
                "backend_label": backend_name,
                "provider": "local_hf",
                "deployment": "local",
                "serving_stack": backend_name,
            },
        ),
    }


def _resolve_backend_metadata(cfg, backend_name: str, *, model_config) -> dict[str, str]:
    spec = cfg.systems_benchmark.backend_specs.get(backend_name)
    metadata = _default_backend_metadata(cfg, backend_name, model_id=model_config.llm_model_id)
    if spec is None:
        return metadata
    if spec.label:
        metadata["backend_label"] = spec.label
    if spec.model_family:
        metadata["model_family"] = spec.model_family
    if spec.provider:
        metadata["provider"] = spec.provider
    if spec.deployment:
        metadata["deployment"] = spec.deployment
    if spec.serving_stack:
        metadata["serving_stack"] = spec.serving_stack
    return metadata


def _resolve_model_config(cfg, backend_name: str):
    spec = cfg.systems_benchmark.backend_specs.get(backend_name)
    if spec is None:
        return cfg.model
    model_config = cfg.model.model_copy(deep=True)
    if spec.llm_model_id:
        model_config.llm_model_id = spec.llm_model_id
    if spec.asr_model_id:
        model_config.asr_model_id = spec.asr_model_id
    if spec.device:
        model_config.device = spec.device
    if spec.dtype:
        model_config.dtype = spec.dtype
    if spec.load_in_8bit is not None:
        model_config.load_in_8bit = spec.load_in_8bit
    if spec.attn_implementation:
        model_config.attn_implementation = spec.attn_implementation
    return model_config


def _benchmark_hf_sequential(
    cfg,
    prompts: list[str],
    *,
    backend_name: str = "hf_sequential",
    generator=None,
    model_config=None,
    backend_metadata: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    model_config = model_config or cfg.model
    generator = generator or NoteGenerator(model_config, cfg.generation)
    backend_metadata = backend_metadata or _resolve_backend_metadata(cfg, backend_name, model_config=model_config)
    batch_rows: list[dict] = []
    request_rows: list[dict] = []
    quality_predictions: list[dict] = []
    for batch_size in cfg.systems_benchmark.batch_sizes:
        waves = _group_prompts(prompts, batch_size)
        for repeat_idx in range(cfg.systems_benchmark.repeat_batches + cfg.systems_benchmark.warmup_batches):
            record = repeat_idx >= cfg.systems_benchmark.warmup_batches
            for wave_index, wave_prompts in enumerate(waves):
                wave_start = time.perf_counter()
                wave_requests = []
                for local_index, prompt in enumerate(wave_prompts):
                    result = generator.generate(prompt, cfg.benchmark)
                    wave_requests.append(result)
                    if record:
                        request_id = f"{backend_name}_b{batch_size}_r{repeat_idx}_w{wave_index}_i{local_index}"
                        request_row = {
                            "backend": backend_name,
                            "batch_size": batch_size,
                            "repeat_index": repeat_idx,
                            "wave_index": wave_index,
                            "request_id": request_id,
                            "text": result.text,
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "total_tokens": result.total_tokens,
                            "prompt_token_source": result.prompt_token_source,
                            "completion_token_source": result.completion_token_source,
                            "tokenization_sec": result.tokenization_sec,
                            "device_transfer_sec": result.device_transfer_sec,
                            "prefill_latency_sec": result.prefill_latency_sec,
                            "ttft_sec": result.ttft_sec,
                            "decode_latency_sec": result.decode_latency_sec,
                            "latency_sec": result.latency_sec,
                            "total_inference_sec": result.total_inference_sec,
                            "peak_gpu_mem_gb": result.peak_gpu_mem_gb,
                            **backend_metadata,
                        }
                        request_rows.append(request_row)
                        if repeat_idx == cfg.systems_benchmark.warmup_batches and batch_size == max(cfg.systems_benchmark.batch_sizes):
                            quality_predictions.append(
                                {
                                    "id": request_id,
                                    "reference_note": None,
                                    "predicted_note": result.text,
                                }
                            )
                wave_latency = time.perf_counter() - wave_start
                if record:
                    completion_counts = [result.completion_tokens for result in wave_requests]
                    completion_observed = all(value is not None for value in completion_counts)
                    total_generated_tokens = (
                        sum(value for value in completion_counts if value is not None)
                        if completion_observed
                        else None
                    )
                    peak_values = [result.peak_gpu_mem_gb for result in wave_requests if result.peak_gpu_mem_gb is not None]
                    batch_rows.append(
                        {
                            "backend": backend_name,
                            "batch_size": batch_size,
                            "repeat_index": repeat_idx,
                            "wave_index": wave_index,
                            "batch_latency_sec": wave_latency,
                            "total_generated_tokens": total_generated_tokens,
                            "throughput_tok_per_sec": None if total_generated_tokens is None else total_generated_tokens / wave_latency,
                            "requests_per_sec": len(wave_requests) / wave_latency,
                            "peak_gpu_mem_gb": max(peak_values) if peak_values else None,
                            **backend_metadata,
                        }
                    )
    return batch_rows, request_rows, quality_predictions


def _benchmark_manual_runtime(
    cfg,
    prompts: list[str],
    *,
    backend_name: str,
    scheduler_mode: str,
    batch_sizes: list[int],
    triton_paged_kv_enabled: bool,
    backend_metadata: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    runtime_cfg = cfg.runtime.model_copy(deep=True)
    runtime_cfg.backend = "mistral_paged"
    runtime_cfg.scheduler_mode = scheduler_mode
    runtime_cfg.triton_paged_kv_enabled = triton_paged_kv_enabled
    engine = ManualBatchEngine(cfg.model, cfg.generation, runtime_cfg)
    backend_metadata = backend_metadata or _resolve_backend_metadata(cfg, backend_name, model_config=cfg.model)
    batch_rows: list[dict] = []
    request_rows: list[dict] = []
    quality_predictions: list[dict] = []
    for batch_size in batch_sizes:
        runtime_cfg.max_batch_size = batch_size
        engine.runtime_config.max_batch_size = batch_size
        waves = _group_prompts(prompts, batch_size)
        for repeat_idx in range(cfg.systems_benchmark.repeat_batches + cfg.systems_benchmark.warmup_batches):
            record = repeat_idx >= cfg.systems_benchmark.warmup_batches
            for wave_index, wave_prompts in enumerate(waves):
                request_ids = [f"{backend_name}_b{batch_size}_r{repeat_idx}_w{wave_index}_i{i}" for i in range(len(wave_prompts))]
                batch_result = engine.run_batch(request_ids, wave_prompts)
                if not record:
                    continue
                batch_row = {
                    "backend": backend_name,
                    "repeat_index": repeat_idx,
                    "wave_index": wave_index,
                    **batch_result.as_dict(),
                    **backend_metadata,
                }
                batch_rows.append(batch_row)
                for request_result in batch_result.requests:
                    request_row = {
                        "backend": backend_name,
                        "batch_size": batch_size,
                        "repeat_index": repeat_idx,
                        "wave_index": wave_index,
                        **request_result.as_dict(),
                        **backend_metadata,
                    }
                    request_rows.append(request_row)
                    if repeat_idx == cfg.systems_benchmark.warmup_batches and batch_size == max(cfg.systems_benchmark.batch_sizes):
                        quality_predictions.append(
                            {
                                "id": request_result.request_id,
                                "reference_note": None,
                                "predicted_note": request_result.text,
                            }
                        )
    return batch_rows, request_rows, quality_predictions


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if cfg.systems_benchmark.max_new_tokens is not None:
        cfg.generation.max_new_tokens = cfg.systems_benchmark.max_new_tokens
    resolved_backends = list(cfg.systems_benchmark.backends)
    backend_metadata = {
        backend_name: _resolve_backend_metadata(
            cfg,
            backend_name,
            model_config=_resolve_model_config(cfg, backend_name),
        )
        for backend_name in resolved_backends
    }
    print(
        "Resolved systems benchmark backends: "
        + ", ".join(resolved_backends)
        + f" | Triton runtime usable: {TRITON_PAGED_KV_RUNTIME_USABLE}"
        + f" | Triton backend enabled: {cfg.runtime.triton_paged_kv_enabled}",
        flush=True,
    )

    dataset = load_dataset(cfg.dataset.path, cfg.systems_benchmark.num_prompts)
    prompts = [
        build_prompt_for_mode(sample.transcript, cfg.systems_benchmark.prompt_mode)
        for sample in dataset
        if sample.transcript
    ]
    if not prompts:
        raise SystemExit("No transcript prompts were available for systems benchmarking.")

    benchmark_dir = cfg.experiment.benchmark_dir / "systems"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    repo_tables_dir = PROJECT_ROOT / "results" / "tables"
    repo_figures_dir = PROJECT_ROOT / "results" / "figures"
    repo_tables_dir.mkdir(parents=True, exist_ok=True)
    repo_figures_dir.mkdir(parents=True, exist_ok=True)

    all_batch_rows: list[dict] = []
    all_request_rows: list[dict] = []
    quality_by_backend: dict[str, dict] = {}
    backend_failures: dict[str, str] = {}
    enabled_backends = set(resolved_backends)

    if "hf_sequential" in enabled_backends:
        hf_model_config = _resolve_model_config(cfg, "hf_sequential")
        hf_batch_rows, hf_request_rows, _hf_quality_predictions = _benchmark_hf_sequential(
            cfg,
            prompts,
            backend_name="hf_sequential",
            model_config=hf_model_config,
            backend_metadata=backend_metadata["hf_sequential"],
        )
        all_batch_rows.extend(hf_batch_rows)
        all_request_rows.extend(hf_request_rows)
        _release_backend_memory()

    for backend_name in resolved_backends:
        if backend_name == "hf_sequential":
            continue
        if backend_name in {"mistral_paged_single", "mistral_paged_static_batch", "mistral_paged_static_batch_triton"}:
            continue
        spec = cfg.systems_benchmark.backend_specs.get(backend_name)
        if spec is None:
            continue
        if spec.provider == "local_hf":
            model_config = _resolve_model_config(cfg, backend_name)
            batch_rows, request_rows, _quality_predictions = _benchmark_hf_sequential(
                cfg,
                prompts,
                backend_name=backend_name,
                model_config=model_config,
                backend_metadata=backend_metadata[backend_name],
            )
            all_batch_rows.extend(batch_rows)
            all_request_rows.extend(request_rows)
            _release_backend_memory()
        elif spec.provider == "together_hosted":
            model_config = _resolve_model_config(cfg, backend_name)
            try:
                generator = TogetherHostedGenerator(
                    model_config=model_config,
                    generation_config=cfg.generation,
                    backend_spec=spec,
                )
                batch_rows, request_rows, _quality_predictions = _benchmark_hf_sequential(
                    cfg,
                    prompts,
                    backend_name=backend_name,
                    generator=generator,
                    model_config=model_config,
                    backend_metadata=backend_metadata[backend_name],
                )
                all_batch_rows.extend(batch_rows)
                all_request_rows.extend(request_rows)
                _release_backend_memory()
            except Exception as exc:
                backend_failures[backend_name] = str(exc)
                print(
                    f"Warning: backend {backend_name} failed and will be recorded as unavailable: {exc}",
                    flush=True,
                )

    if "mistral_paged_single" in enabled_backends:
        runtime_v0_batch_rows, runtime_v0_request_rows, _runtime_v0_quality_predictions = _benchmark_manual_runtime(
            cfg,
            prompts,
            backend_name="mistral_paged_single",
            scheduler_mode="none",
            batch_sizes=[1],
            triton_paged_kv_enabled=False,
            backend_metadata=backend_metadata["mistral_paged_single"],
        )
        all_batch_rows.extend(runtime_v0_batch_rows)
        all_request_rows.extend(runtime_v0_request_rows)
        _release_backend_memory()

    if "mistral_paged_static_batch" in enabled_backends:
        runtime_v1_batch_rows, runtime_v1_request_rows, _runtime_v1_quality_predictions = _benchmark_manual_runtime(
            cfg,
            prompts,
            backend_name="mistral_paged_static_batch",
            scheduler_mode="static_batch",
            batch_sizes=cfg.systems_benchmark.batch_sizes,
            triton_paged_kv_enabled=False,
            backend_metadata=backend_metadata["mistral_paged_static_batch"],
        )
        all_batch_rows.extend(runtime_v1_batch_rows)
        all_request_rows.extend(runtime_v1_request_rows)
        _release_backend_memory()

    if "mistral_paged_static_batch_triton" in enabled_backends and cfg.runtime.triton_paged_kv_enabled:
        if not TRITON_PAGED_KV_RUNTIME_USABLE:
            raise SystemExit(
                "systems_benchmark requested mistral_paged_static_batch_triton but Triton paged-KV runtime is unavailable"
            )
        triton_batch_rows, triton_request_rows, _triton_quality_predictions = _benchmark_manual_runtime(
            cfg,
            prompts,
            backend_name="mistral_paged_static_batch_triton",
            scheduler_mode="static_batch",
            batch_sizes=cfg.systems_benchmark.batch_sizes,
            triton_paged_kv_enabled=True,
            backend_metadata=backend_metadata["mistral_paged_static_batch_triton"],
        )
        all_batch_rows.extend(triton_batch_rows)
        all_request_rows.extend(triton_request_rows)
        _release_backend_memory()

    for backend in sorted({row["backend"] for row in all_request_rows}):
        backend_predictions = []
        for row, sample in zip(
            [record for record in all_request_rows if record["backend"] == backend][: len(dataset)],
            dataset,
            strict=False,
        ):
            backend_predictions.append(
                {
                    "id": row["request_id"],
                    "reference_note": sample.reference_note,
                    "predicted_note": row["text"],
                    "metadata": sample.metadata,
                }
            )
        quality_by_backend[backend] = maybe_score_references(backend_predictions)

    summary_rows: list[dict] = []
    for backend in sorted({row["backend"] for row in all_batch_rows}):
        for batch_size in cfg.systems_benchmark.batch_sizes:
            rows = [row for row in all_batch_rows if row["backend"] == backend and row["batch_size"] == batch_size]
            if not rows:
                continue
            summary = aggregate_numeric_records(rows)
            request_summary = _aggregate_request_summaries(all_request_rows, backend=backend, batch_size=batch_size)
            representative_row = rows[0]
            summary_rows.append(
                {
                    "backend": backend,
                    "batch_size": batch_size,
                    "backend_label": representative_row.get("backend_label"),
                    "model_family": representative_row.get("model_family"),
                    "provider": representative_row.get("provider"),
                    "deployment": representative_row.get("deployment"),
                    "serving_stack": representative_row.get("serving_stack"),
                    "mean_batch_latency_sec": summary.get("batch_latency_sec", {}).get("mean"),
                    "mean_throughput_tok_per_sec": summary.get("throughput_tok_per_sec", {}).get("mean"),
                    "mean_requests_per_sec": summary.get("requests_per_sec", {}).get("mean"),
                    "mean_peak_gpu_mem_gb": request_summary.get("mean_peak_gpu_mem_gb"),
                    "mean_ttft_sec": request_summary.get("mean_ttft_sec"),
                    "mean_prefill_latency_sec": request_summary.get("mean_prefill_latency_sec"),
                    "mean_decode_latency_sec": request_summary.get("mean_decode_latency_sec"),
                    "mean_latency_sec": request_summary.get("mean_latency_sec"),
                    "mean_prompt_tokens": request_summary.get("mean_prompt_tokens"),
                    "mean_completion_tokens": request_summary.get("mean_completion_tokens"),
                    "mean_kv_allocated_bytes": summary.get("kv_allocated_bytes", {}).get("mean"),
                    "mean_kv_utilization_ratio": summary.get("kv_utilization_ratio", {}).get("mean"),
                    "mean_kv_fragmentation_ratio": summary.get("kv_fragmentation_ratio", {}).get("mean"),
                }
            )

    report = {
        "config": str(args.config),
        "num_prompts": len(prompts),
        "batch_sizes": cfg.systems_benchmark.batch_sizes,
        "enabled_backends": sorted(enabled_backends),
        "prompt_mode": cfg.systems_benchmark.prompt_mode,
        "backend_failures": backend_failures,
        "backend_metadata": backend_metadata,
        "triton_paged_kv_runtime_usable": TRITON_PAGED_KV_RUNTIME_USABLE,
        "summary_rows": summary_rows,
        "quality_by_backend": quality_by_backend,
    }
    write_json(benchmark_dir / "systems_benchmark_report.json", report)
    write_json(benchmark_dir / "systems_quality.json", quality_by_backend)
    write_json(benchmark_dir / "systems_batch_rows.json", {"rows": all_batch_rows})
    write_json(benchmark_dir / "systems_request_rows.json", {"rows": all_request_rows})
    write_csv_rows(benchmark_dir / "systems_summary.csv", summary_rows)
    write_csv_rows(benchmark_dir / "systems_batch_rows.csv", all_batch_rows)
    write_csv_rows(benchmark_dir / "systems_request_rows.csv", all_request_rows)

    write_csv_rows(repo_tables_dir / "systems_summary.csv", summary_rows)
    write_csv_rows(repo_tables_dir / "systems_batch_rows.csv", all_batch_rows)
    write_csv_rows(repo_tables_dir / "systems_request_rows.csv", all_request_rows)
    if cfg.systems_benchmark.save_figures:
        _write_summary_figures(summary_rows, repo_figures_dir)
        _write_summary_figures(summary_rows, benchmark_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
