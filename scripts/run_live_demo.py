from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clinical_speech.config import GenerationConfig, ModelConfig, RuntimeConfig
from clinical_speech.pipeline.prompts import build_note_prompt
from clinical_speech.runtime.engine import ManualBatchEngine
from clinical_speech.utils.io import read_jsonl


@dataclass(frozen=True)
class DemoSystemSpec:
    key: str
    label: str
    description: str
    llm_model_id: str


TRITON_SYSTEM = DemoSystemSpec(
    key="mistral_triton",
    label="Mistral 7B Paged + Triton",
    description="Custom paged-KV runtime with Triton paged gather enabled.",
    llm_model_id="mistralai/Mistral-7B-Instruct-v0.3",
)

EXAMPLE_DATASET_CANDIDATES = [
    Path("/data/project_runtime/datasets/medsynth/test.jsonl"),
    Path("/data/project_runtime/datasets/meddialog/test.jsonl"),
    PROJECT_ROOT / "data" / "fixtures" / "smoke_notes_long.jsonl",
    PROJECT_ROOT / "data" / "fixtures" / "smoke_notes.jsonl",
]

_ACTIVE_ENGINE: ManualBatchEngine | None = None

DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_NEW_TOKENS = 384
WARMUP_BATCH_SIZE = 8
WARMUP_MAX_NEW_TOKENS = 16
WARMUP_TRANSCRIPT = "Patient reports mild cough for one day and wants advice."
REAL_INFERENCE_LABEL = (
    "Real inference: model is preloaded/warmed, outputs are generated live for the submitted transcript."
)
UI_METRIC_KEYS = (
    "batch_size",
    "active_requests",
    "total_generated_tokens",
    "ttft_sec_request_1",
    "batch_latency_sec",
    "throughput_tok_per_sec",
    "peak_gpu_mem_gb",
)


def _example_dataset_path() -> Path | None:
    for candidate in EXAMPLE_DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _model_config() -> ModelConfig:
    return ModelConfig(
        asr_model_id="openai/whisper-small",
        llm_model_id=TRITON_SYSTEM.llm_model_id,
        device="cuda",
        dtype="float16",
        load_in_8bit=False,
        attn_implementation="sdpa",
    )


def _generation_config(max_new_tokens: int) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=int(max_new_tokens),
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
    )


def _runtime_config(batch_size: int) -> RuntimeConfig:
    resolved_batch_size = max(1, int(batch_size))
    return RuntimeConfig(
        backend="mistral_paged",
        scheduler_mode="static_batch",
        max_batch_size=resolved_batch_size,
        max_concurrent_requests=resolved_batch_size,
        max_cache_budget_gb=4.0,
        triton_enabled=True,
        triton_paged_kv_enabled=True,
    )


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _release_active_engine() -> None:
    global _ACTIVE_ENGINE

    engine = _ACTIVE_ENGINE
    _ACTIVE_ENGINE = None
    if engine is None:
        return
    if hasattr(engine, "model"):
        delattr(engine, "model")
    if hasattr(engine, "tokenizer"):
        delattr(engine, "tokenizer")
    del engine
    _release_cuda_memory()


def _load_engine(*, max_new_tokens: int, batch_size: int) -> ManualBatchEngine:
    global _ACTIVE_ENGINE

    if _ACTIVE_ENGINE is None:
        _ACTIVE_ENGINE = ManualBatchEngine(
            _model_config(),
            _generation_config(max_new_tokens),
            _runtime_config(batch_size),
        )
        return _ACTIVE_ENGINE

    _ACTIVE_ENGINE.generation_config.max_new_tokens = int(max_new_tokens)
    _ACTIVE_ENGINE.runtime_config.max_batch_size = max(1, int(batch_size))
    _ACTIVE_ENGINE.runtime_config.max_concurrent_requests = max(1, int(batch_size))
    return _ACTIVE_ENGINE


def _preload_and_warm_backend() -> dict[str, Any]:
    """Compile/load the real backend path once; discard the generated warmup text."""
    _, metrics = _run_triton_batch(
        WARMUP_TRANSCRIPT,
        max_new_tokens=WARMUP_MAX_NEW_TOKENS,
        batch_size=WARMUP_BATCH_SIZE,
    )
    return metrics


def load_example(index: int = 0) -> str:
    dataset_path = _example_dataset_path()
    if dataset_path is None:
        return ""
    rows = read_jsonl(dataset_path)
    if not rows:
        return ""
    return str(rows[index % len(rows)].get("transcript", "")).strip()


def _resolve_transcript(transcript_text: str) -> tuple[str, str]:
    cleaned_text = (transcript_text or "").strip()
    if not cleaned_text:
        raise ValueError("Paste or load a transcript before running the demo.")
    return cleaned_text, "Using pasted or loaded transcript input."


def _run_triton_batch(transcript: str, *, max_new_tokens: int, batch_size: int) -> tuple[str, dict[str, Any]]:
    prompt = build_note_prompt(transcript)
    resolved_batch_size = max(1, int(batch_size))
    prompts = [prompt for _ in range(resolved_batch_size)]
    request_ids = [f"demo_req_{index:02d}" for index in range(resolved_batch_size)]

    engine = _load_engine(max_new_tokens=max_new_tokens, batch_size=resolved_batch_size)
    start = time.perf_counter()
    batch = engine.run_batch(request_ids=request_ids, prompts=prompts)
    wall_clock_sec = time.perf_counter() - start

    primary_request = batch.requests[0]
    total_completion_tokens = sum(request.completion_tokens for request in batch.requests)
    mean_completion_tokens = total_completion_tokens / len(batch.requests)
    likely_truncated = any(request.completion_tokens >= int(max_new_tokens) for request in batch.requests)

    metrics = {
        "system": TRITON_SYSTEM.label,
        "batch_size": batch.batch_size,
        "active_requests": batch.batch_size,
        "prompt_tokens_request_1": primary_request.prompt_tokens,
        "completion_tokens_request_1": primary_request.completion_tokens,
        "mean_completion_tokens": mean_completion_tokens,
        "total_generated_tokens": batch.total_generated_tokens,
        "ttft_sec_request_1": primary_request.ttft_sec,
        "prefill_latency_sec_batch": batch.prefill_latency_sec,
        "decode_latency_sec_request_1": primary_request.decode_latency_sec,
        "latency_sec_request_1": primary_request.latency_sec,
        "batch_latency_sec": batch.batch_latency_sec,
        "throughput_tok_per_sec": batch.throughput_tok_per_sec,
        "requests_per_sec": batch.requests_per_sec,
        "peak_gpu_mem_gb": batch.peak_gpu_mem_gb,
        "scheduler_steps": batch.scheduler_steps,
        "wall_clock_sec": wall_clock_sec,
        "likely_truncated_at_max_new_tokens": likely_truncated,
        **batch.kv_cache_metrics,
    }
    return primary_request.text, metrics


def _format_metrics(metrics: dict[str, Any], *, compact: bool = True) -> str:
    lines = ["| Metric | Value |", "| --- | --- |"]
    keys = UI_METRIC_KEYS if compact else tuple(metrics.keys())
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, float):
            lines.append(f"| `{key}` | {value:.4f} |")
        else:
            lines.append(f"| `{key}` | {value} |")
    return "\n".join(lines)


def _run_demo_outputs(
    transcript_text: str,
    batch_size: int,
    max_new_tokens: int,
) -> tuple[str, str, str, str]:
    try:
        transcript, source_message = _resolve_transcript(transcript_text)
        output_text, metrics = _run_triton_batch(
            transcript,
            max_new_tokens=int(max_new_tokens),
            batch_size=int(batch_size),
        )
    except Exception as exc:
        fallback_transcript = (transcript_text or "").strip()
        message = str(exc)
        if "out of memory" in message.lower():
            message = (
                "Run failed: CUDA out of memory. "
                "Reduce batch size or max new tokens, or use a shorter transcript on this A10G."
            )
        else:
            message = f"Run failed: {exc}"
        return fallback_transcript, message, "", "_No metrics yet._"

    status_message = (
        f"{source_message} Ran `{TRITON_SYSTEM.label}` with batch size `{batch_size}`. "
        f"Active requests: `{batch_size}`. "
        f"{'Output likely hit the max token limit; raise `Max new tokens` for a fuller note.' if metrics['likely_truncated_at_max_new_tokens'] else 'Output completed within the current token budget.'}"
    )
    print(
        "Live run complete: "
        f"batch_size={metrics['batch_size']}, "
        f"max_new_tokens={max_new_tokens}, "
        f"batch_latency_sec={metrics['batch_latency_sec']:.2f}, "
        f"throughput_tok_per_sec={metrics['throughput_tok_per_sec']:.2f}",
        flush=True,
    )
    return transcript, status_message, output_text, _format_metrics(metrics)


def _clear_outputs() -> tuple[str, str, str, str]:
    return "", "_Results cleared._", "", "_No metrics yet._"


def launch_gradio(*, server_name: str, server_port: int) -> None:
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError("Gradio is not installed. Install it in the project venv before launching the demo.") from exc

    print(
        f"Preloading and warming {TRITON_SYSTEM.label} "
        f"(batch_size={WARMUP_BATCH_SIZE}, max_new_tokens={WARMUP_MAX_NEW_TOKENS})...",
        flush=True,
    )
    warmup_metrics = _preload_and_warm_backend()
    print(
        "Warmup complete: "
        f"throughput={warmup_metrics['throughput_tok_per_sec']:.2f} tok/s, "
        f"peak_gpu_mem_gb={warmup_metrics['peak_gpu_mem_gb']}",
        flush=True,
    )

    examples = [load_example(0), load_example(1), load_example(2)]

    custom_theme = gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="slate",
        neutral_hue="zinc",
    )
    custom_css = """
        .app-shell {max-width: 1360px; margin: 0 auto;}
        .hero-card, .control-card, .results-card {border: 1px solid #e7e5e4; border-radius: 18px;}
        .hero-card {background: linear-gradient(135deg, #fff7ed 0%, #ffffff 48%, #f8fafc 100%);}
        .metric-card {background: #fafaf9; border-radius: 14px; padding: 12px;}
        .results-card textarea {font-size: 15px; line-height: 1.45;}
        """

    with gr.Blocks(title="Clinical Note Live Demo") as demo:
        gr.Markdown(
            "# Triton Clinical Note Demo\n"
            "Paste one transcript, run the Triton backend only, and scale the batch size to show true batched throughput.\n\n"
            f"**{REAL_INFERENCE_LABEL}**",
            elem_classes=["app-shell", "hero-card"],
        )
        with gr.Row(elem_classes=["app-shell"]):
            with gr.Column(scale=7, elem_classes=["control-card"]):
                gr.Markdown("## Transcript")
                transcript_input = gr.Textbox(
                    label="Paste Clinical Transcript",
                    value=load_example(0),
                    lines=12,
                    max_lines=18,
                    placeholder="Paste the clinical conversation or prepared demo transcript here.",
                )
                gr.Examples(examples=examples, inputs=transcript_input, label="Quick examples")
                with gr.Row():
                    run_btn = gr.Button("Run Triton Batch", variant="primary")
                    clear_btn = gr.Button("Clear")
                with gr.Row():
                    example_idx = gr.Slider(0, 7, value=0, step=1, label="Example index")
                    load_btn = gr.Button("Load Example")
            with gr.Column(scale=5, elem_classes=["control-card"]):
                gr.Markdown("## Run Controls")
                backend_name = gr.Textbox(label="Backend", value=TRITON_SYSTEM.label, interactive=False)
                batch_size = gr.Slider(1, 8, value=DEFAULT_BATCH_SIZE, step=1, label="Batch size")
                max_tokens = gr.Slider(
                    128,
                    768,
                    value=DEFAULT_MAX_NEW_TOKENS,
                    step=64,
                    label="Max new tokens",
                    info="384 is the complete-note default; lower only if you need a faster stage run.",
                )
                resolved_transcript = gr.Textbox(label="Transcript used", lines=12, max_lines=18)
                status = gr.Markdown(
                    f"_Ready. {REAL_INFERENCE_LABEL}_",
                    elem_classes=["metric-card"],
                )

        with gr.Row(elem_classes=["app-shell"]):
            with gr.Column(scale=7, elem_classes=["results-card"]):
                gr.Markdown(f"## Output\n### {TRITON_SYSTEM.label}")
                output_text = gr.Textbox(
                    label="Generated SOAP note (request 1)",
                    lines=24,
                    max_lines=32,
                    autoscroll=False,
                )
            with gr.Column(scale=5, elem_classes=["results-card"]):
                gr.Markdown("## Metrics")
                metrics_md = gr.Markdown("_No metrics yet._", elem_classes=["metric-card"])

        load_btn.click(lambda idx: load_example(int(idx)), inputs=example_idx, outputs=transcript_input)
        run_btn.click(
            _run_demo_outputs,
            inputs=[transcript_input, batch_size, max_tokens],
            outputs=[resolved_transcript, status, output_text, metrics_md],
            concurrency_limit=1,
        )
        clear_btn.click(
            _clear_outputs,
            outputs=[resolved_transcript, status, output_text, metrics_md],
        )

    demo.launch(server_name=server_name, server_port=server_port, theme=custom_theme, css=custom_css)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Run a one-shot CLI demo instead of launching Gradio.")
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument("--transcript-file", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.cli:
        launch_gradio(server_name=args.server_name, server_port=args.server_port)
        return

    transcript = args.transcript_file.read_text(encoding="utf-8") if args.transcript_file else load_example(args.example_index)
    text, metrics = _run_triton_batch(
        transcript,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    print(f"SYSTEM: {TRITON_SYSTEM.label}")
    print(f"BATCH SIZE: {metrics['batch_size']}")
    print(f"ACTIVE REQUESTS: {metrics['active_requests']}")
    print("SOAP NOTE (REQUEST 1)")
    print(text)
    print("\nMETRICS")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
