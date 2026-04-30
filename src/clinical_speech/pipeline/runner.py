import argparse
import logging
import time
from pathlib import Path

from tqdm import tqdm

from clinical_speech.benchmarking import safe_rate, write_csv_rows
from clinical_speech.config import AppConfig, load_config
from clinical_speech.data.dataset import load_dataset
from clinical_speech.evaluation.metrics import aggregate_runtime_metrics, maybe_score_references
from clinical_speech.models.asr import ASRModel
from clinical_speech.models.note_generator import GenerationResult, NoteGenerator
from clinical_speech.pipeline.chunking import chunk_text_by_words
from clinical_speech.pipeline.prompts import (
    build_chunk_summary_prompt,
    build_final_note_from_summaries_prompt,
    build_note_prompt,
)
from clinical_speech.storage import (
    DATA_MOUNT_ROOT,
    PROJECT_ROOT,
    cleanup_stale_temp_dirs,
    configure_rotating_log,
    disk_status,
    format_disk_status,
    managed_temp_dir,
    prune_old_files,
    resolve_managed_path,
    validate_storage_layout,
)
from clinical_speech.utils.io import write_json, write_jsonl

LOGGER = logging.getLogger("clinical_speech")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def _sum_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present)


def _max_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return max(present)


def _append_generation_benchmark_rows(
    rows: list[dict],
    *,
    sample_id: str,
    stage: str,
    result: GenerationResult,
    chunk_index: int | None = None,
) -> None:
    for run_index, run_metrics in enumerate(result.benchmark_runs, start=1):
        row = {
            "sample_id": sample_id,
            "stage": stage,
            "run_index": run_index,
            "chunk_index": chunk_index,
        }
        row.update(run_metrics.as_dict())
        rows.append(row)


def run_sample(
    sample,
    cfg: AppConfig,
    note_generator: NoteGenerator,
    asr_model: ASRModel | None,
) -> tuple[dict, dict | None, dict, list[dict]]:
    sample_start = time.perf_counter()

    transcript = sample.transcript
    reference_transcript = sample.transcript if cfg.dataset.task == "audio_to_note" else None
    generated_transcript = None
    asr_latency = None
    if cfg.dataset.task == "audio_to_note":
        if not sample.audio_path:
            raise ValueError(f"Sample {sample.id} is missing audio_path")
        if asr_model is None:
            raise ValueError("ASR model was not initialized")
        asr_start = time.perf_counter()
        transcript = asr_model.transcribe(sample.audio_path)
        asr_latency = time.perf_counter() - asr_start
        generated_transcript = transcript

    if not transcript:
        raise ValueError(f"Sample {sample.id} has no transcript")

    benchmark_rows: list[dict] = []
    if cfg.chunking.enabled:
        chunking_start = time.perf_counter()
        chunks = chunk_text_by_words(
            transcript,
            chunk_words=cfg.chunking.chunk_words,
            overlap_words=cfg.chunking.overlap_words,
        )
        chunking_latency_sec = time.perf_counter() - chunking_start

        chunk_summaries = []
        summary_results: list[GenerationResult] = []
        summarization_start = time.perf_counter()
        for chunk_index, chunk in enumerate(chunks):
            summary_prompt = build_chunk_summary_prompt(chunk, cfg.chunking.summary_prompt)
            summary_result = note_generator.generate(summary_prompt, cfg.benchmark)
            chunk_summaries.append(summary_result.text)
            summary_results.append(summary_result)
            _append_generation_benchmark_rows(
                benchmark_rows,
                sample_id=sample.id,
                stage="summarization",
                result=summary_result,
                chunk_index=chunk_index,
            )
        summarization_latency_sec = time.perf_counter() - summarization_start
        prompt = build_final_note_from_summaries_prompt(chunk_summaries)
    else:
        prompt = build_note_prompt(transcript)
        chunk_summaries = []
        summary_results = []
        chunking_latency_sec = 0.0
        summarization_latency_sec = 0.0

    note_result = note_generator.generate(prompt, cfg.benchmark)
    _append_generation_benchmark_rows(
        benchmark_rows,
        sample_id=sample.id,
        stage="final_note",
        result=note_result,
    )
    total_runtime = time.perf_counter() - sample_start

    summary_peak_gpu_mem_gb = _max_optional([result.peak_gpu_mem_gb for result in summary_results])
    peak_gpu_mem_gb = _max_optional([summary_peak_gpu_mem_gb, note_result.peak_gpu_mem_gb])
    summarization_generation_latency_sec = sum(result.latency_sec for result in summary_results)
    summarization_total_inference_sec = sum(result.total_inference_sec for result in summary_results)
    summarization_tokenization_sec = sum(result.tokenization_sec for result in summary_results)
    summarization_device_transfer_sec = sum(result.device_transfer_sec for result in summary_results)
    summarization_prefill_latency_sec = _sum_optional([result.prefill_latency_sec for result in summary_results])
    summarization_ttft_sec = _sum_optional([result.ttft_sec for result in summary_results])
    summarization_decode_latency_sec = _sum_optional([result.decode_latency_sec for result in summary_results])
    summarization_prompt_tokens = sum(result.prompt_tokens for result in summary_results)
    summarization_completion_tokens = sum(result.completion_tokens for result in summary_results)
    summarization_total_tokens = sum(result.total_tokens for result in summary_results)

    prediction = {
        "id": sample.id,
        "transcript": transcript,
        "reference_transcript": reference_transcript,
        "generated_transcript": generated_transcript,
        "reference_note": sample.reference_note,
        "predicted_note": note_result.text,
        "audio_path": sample.audio_path,
        "metadata": sample.metadata,
        "runtime": {
            "preprocessing_latency_sec": asr_latency if asr_latency is not None else 0.0,
            "asr_latency_sec": asr_latency,
            "chunking_latency_sec": chunking_latency_sec,
            "summarization_latency_sec": summarization_latency_sec,
            "summarization_generation_latency_sec": summarization_generation_latency_sec,
            "summarization_total_inference_sec": summarization_total_inference_sec,
            "summarization_tokenization_sec": summarization_tokenization_sec,
            "summarization_device_transfer_sec": summarization_device_transfer_sec,
            "summarization_prefill_latency_sec": summarization_prefill_latency_sec,
            "summarization_ttft_sec": summarization_ttft_sec,
            "summarization_decode_latency_sec": summarization_decode_latency_sec,
            "summarization_prompt_tokens": summarization_prompt_tokens,
            "summarization_completion_tokens": summarization_completion_tokens,
            "summarization_total_tokens": summarization_total_tokens,
            "summarization_chunk_count": len(summary_results),
            "final_tokenization_sec": note_result.tokenization_sec,
            "final_device_transfer_sec": note_result.device_transfer_sec,
            "final_prefill_latency_sec": note_result.prefill_latency_sec,
            "final_generation_latency_sec": note_result.latency_sec,
            "final_decode_latency_sec": note_result.decode_latency_sec,
            "final_total_inference_sec": note_result.total_inference_sec,
            "generation_latency_sec": note_result.latency_sec,
            "ttft_sec": note_result.ttft_sec,
            "prefill_latency_sec": note_result.prefill_latency_sec,
            "decode_latency_sec": note_result.decode_latency_sec,
            "prompt_tokens": note_result.prompt_tokens,
            "completion_tokens": note_result.completion_tokens,
            "total_tokens": note_result.total_tokens,
            "throughput_tok_per_sec": safe_rate(note_result.completion_tokens, note_result.latency_sec),
            "peak_gpu_mem_gb": peak_gpu_mem_gb,
            "end_to_end_latency_sec": total_runtime,
        },
    }
    raw_intermediate = None
    if cfg.output.save_raw_intermediates and chunk_summaries:
        raw_intermediate = {
            "id": sample.id,
            "chunk_summaries": chunk_summaries,
        }
    sample_benchmark_row = {
        "sample_id": sample.id,
        **prediction["runtime"],
    }
    return prediction, raw_intermediate, sample_benchmark_row, benchmark_rows


def _write_progress_checkpoint(
    cfg: AppConfig,
    predictions: list[dict],
    raw_intermediates: list[dict],
) -> None:
    if cfg.output.keep_latest_checkpoint:
        latest_path = cfg.experiment.checkpoint_dir / "latest_predictions.jsonl"
        latest_state = cfg.experiment.checkpoint_dir / "latest_state.json"
        write_jsonl(latest_path, predictions)
        write_json(
            latest_state,
            {
                "completed_samples": len(predictions),
                "latest_predictions_path": str(latest_path),
            },
        )
        if raw_intermediates and cfg.output.save_raw_intermediates:
            write_jsonl(cfg.experiment.checkpoint_dir / "latest_raw_intermediates.jsonl", raw_intermediates)

    snapshot_path = cfg.experiment.checkpoint_dir / f"predictions_step_{len(predictions):05d}.jsonl"
    write_jsonl(snapshot_path, predictions)
    if raw_intermediates and cfg.output.save_raw_intermediates:
        write_jsonl(
            cfg.experiment.checkpoint_dir / f"raw_intermediates_step_{len(predictions):05d}.jsonl",
            raw_intermediates,
        )
    prune_old_files(
        cfg.experiment.checkpoint_dir,
        pattern="predictions_step_*.jsonl",
        keep=cfg.output.max_checkpoints_to_keep,
    )
    if cfg.output.save_raw_intermediates:
        prune_old_files(
            cfg.experiment.checkpoint_dir,
            pattern="raw_intermediates_step_*.jsonl",
            keep=cfg.output.max_checkpoints_to_keep,
        )


def _maybe_override_output_dir(cfg: AppConfig, output_dir: Path | None) -> None:
    if output_dir is None:
        return
    cfg.experiment.output_dir = resolve_managed_path(
        output_dir,
        base_dir=cfg.storage.outputs_dir,
        strip_prefixes=("outputs",),
    )
    cfg.experiment.output_dir.mkdir(parents=True, exist_ok=True)


def _run_preflight(cfg: AppConfig) -> None:
    warnings = validate_storage_layout(
        cfg.managed_paths(),
        runtime_root=cfg.storage.runtime_root,
        min_data_free_gb=cfg.preflight.min_free_data_gb,
        min_root_free_gb=cfg.preflight.min_free_root_gb,
    )
    LOGGER.info("Storage preflight passed: %s", format_disk_status(disk_status(DATA_MOUNT_ROOT)))
    LOGGER.info("Storage preflight passed: %s", format_disk_status(disk_status(Path("/"))))
    for warning in warnings:
        LOGGER.warning(warning)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, runtime_root=args.runtime_root)
    _maybe_override_output_dir(cfg, args.output_dir)
    if args.benchmark_only:
        cfg.benchmark.benchmark_only = True

    log_path = configure_rotating_log(
        cfg.experiment.log_dir,
        max_bytes=cfg.output.max_log_bytes,
        backup_count=cfg.output.log_backup_count,
    )
    LOGGER.info("Project root: %s", PROJECT_ROOT)
    LOGGER.info("Logging to %s", log_path)
    LOGGER.info("Experiment outputs: %s", cfg.experiment.output_dir)
    LOGGER.info("Dataset path: %s", cfg.dataset.path)

    stale_temp_dirs = cleanup_stale_temp_dirs(
        cfg.storage.tmp_dir,
        older_than_hours=cfg.output.stale_tmp_hours,
    )
    if stale_temp_dirs:
        LOGGER.info("Removed %d stale temp directories from %s", len(stale_temp_dirs), cfg.storage.tmp_dir)

    if cfg.preflight.enabled and not args.skip_preflight:
        _run_preflight(cfg)

    dataset = load_dataset(cfg.dataset.path, cfg.dataset.max_samples)

    note_generator = NoteGenerator(cfg.model, cfg.generation)
    asr_model = ASRModel(cfg.model) if cfg.dataset.task == "audio_to_note" else None

    predictions = []
    raw_intermediates = []
    sample_benchmark_rows: list[dict] = []
    generation_benchmark_rows: list[dict] = []
    with managed_temp_dir(
        cfg.storage.tmp_dir,
        prefix=f"{cfg.experiment.name}-",
        cleanup=cfg.output.cleanup_temp_on_success,
    ) as run_tmp_dir:
        LOGGER.info("Using managed temp directory %s", run_tmp_dir)
        for sample in tqdm(dataset, desc=cfg.experiment.name):
            prediction, raw_intermediate, sample_benchmark_row, generation_rows = run_sample(
                sample,
                cfg,
                note_generator,
                asr_model,
            )
            predictions.append(prediction)
            sample_benchmark_rows.append(sample_benchmark_row)
            generation_benchmark_rows.extend(generation_rows)
            if raw_intermediate is not None:
                raw_intermediates.append(raw_intermediate)

            if cfg.output.save_frequency > 0 and len(predictions) % cfg.output.save_frequency == 0:
                _write_progress_checkpoint(cfg, predictions, raw_intermediates)
                LOGGER.info("Saved recovery checkpoint after %d samples", len(predictions))

        if cfg.output.save_frequency > 0 and predictions:
            _write_progress_checkpoint(cfg, predictions, raw_intermediates)

        runtime_metrics = aggregate_runtime_metrics(predictions)
        quality_metrics = (
            {
                "metrics": {},
                "supported_metrics": [],
                "unsupported_metrics": [],
                "warnings": ["Quality metrics skipped because benchmark_only mode is enabled."],
            }
            if cfg.benchmark.benchmark_only
            else maybe_score_references(predictions)
        )
        metrics = {
            "experiment_name": cfg.experiment.name,
            "num_samples": len(predictions),
            "runtime": runtime_metrics,
            "quality": quality_metrics,
            "benchmark": {
                "enabled": cfg.benchmark.enabled,
                "warmup_runs": cfg.benchmark.warmup_runs,
                "repeat_runs": cfg.benchmark.repeat_runs,
                "synchronize_cuda": cfg.benchmark.synchronize_cuda,
                "benchmark_only": cfg.benchmark.benchmark_only,
            },
            "storage": {
                "output_dir": str(cfg.experiment.output_dir),
                "log_dir": str(cfg.experiment.log_dir),
                "checkpoint_dir": str(cfg.experiment.checkpoint_dir),
                "benchmark_dir": str(cfg.experiment.benchmark_dir),
                "profiler_dir": str(cfg.experiment.profiler_dir),
                "tmp_dir": str(run_tmp_dir),
            },
        }

        if cfg.output.save_predictions:
            write_jsonl(cfg.experiment.output_dir / "predictions.jsonl", predictions)
        if cfg.output.save_metrics:
            write_json(cfg.experiment.output_dir / "metrics.json", metrics)
            write_json(
                cfg.experiment.benchmark_dir / "runtime_metrics.json",
                {
                    "experiment_name": cfg.experiment.name,
                    "num_samples": len(predictions),
                    "runtime": runtime_metrics,
                    "quality": quality_metrics,
                    "benchmark": metrics["benchmark"],
                },
            )
            write_json(
                cfg.experiment.benchmark_dir / "sample_runtime_rows.json",
                {"rows": sample_benchmark_rows},
            )
            if cfg.benchmark.save_per_run_metrics:
                write_json(
                    cfg.experiment.benchmark_dir / "generation_run_rows.json",
                    {"rows": generation_benchmark_rows},
                )
            if cfg.benchmark.save_csv:
                write_csv_rows(
                    cfg.experiment.benchmark_dir / "sample_runtime_rows.csv",
                    sample_benchmark_rows,
                )
                if cfg.benchmark.save_per_run_metrics:
                    write_csv_rows(
                        cfg.experiment.benchmark_dir / "generation_run_rows.csv",
                        generation_benchmark_rows,
                    )
        if cfg.output.save_raw_intermediates and raw_intermediates:
            write_jsonl(cfg.experiment.output_dir / "raw_intermediates.jsonl", raw_intermediates)

    LOGGER.info("Completed %d samples", len(predictions))
    print(metrics)
