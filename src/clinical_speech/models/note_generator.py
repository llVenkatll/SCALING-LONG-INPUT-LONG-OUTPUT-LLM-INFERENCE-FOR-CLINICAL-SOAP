import gc
import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.generation.streamers import BaseStreamer

from clinical_speech.benchmarking import aggregate_numeric_records
from clinical_speech.config import BenchmarkConfig, GenerationConfig, ModelConfig
from clinical_speech.models.factory import load_causal_lm_bundle


@dataclass
class GenerationRunMetrics:
    tokenization_sec: float
    device_transfer_sec: float
    prefill_latency_sec: float | None
    ttft_sec: float | None
    decode_latency_sec: float | None
    latency_sec: float
    total_inference_sec: float
    peak_gpu_mem_gb: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    prompt_token_source: str | None = None
    completion_token_source: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "tokenization_sec": self.tokenization_sec,
            "device_transfer_sec": self.device_transfer_sec,
            "prefill_latency_sec": self.prefill_latency_sec,
            "ttft_sec": self.ttft_sec,
            "decode_latency_sec": self.decode_latency_sec,
            "latency_sec": self.latency_sec,
            "total_inference_sec": self.total_inference_sec,
            "peak_gpu_mem_gb": self.peak_gpu_mem_gb,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "prompt_token_source": self.prompt_token_source,
            "completion_token_source": self.completion_token_source,
        }


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    tokenization_sec: float
    device_transfer_sec: float
    prefill_latency_sec: float | None
    latency_sec: float
    ttft_sec: float | None
    decode_latency_sec: float | None
    total_inference_sec: float
    peak_gpu_mem_gb: float | None
    benchmark_runs: list[GenerationRunMetrics]
    benchmark_summary: dict[str, dict[str, float | int]]
    prompt_token_source: str | None = None
    completion_token_source: str | None = None


@dataclass
class BatchGenerationResult:
    texts: list[str]
    prompt_tokens: list[int]
    completion_tokens: list[int]
    total_tokens: list[int]
    active_requests: int
    total_completion_tokens: int
    tokenization_sec: float
    device_transfer_sec: float
    prefill_latency_sec: float | None
    latency_sec: float
    ttft_sec: float | None
    decode_latency_sec: float | None
    total_inference_sec: float
    throughput_tok_per_sec: float | None
    peak_gpu_mem_gb: float | None
    benchmark_runs: list[GenerationRunMetrics]
    benchmark_summary: dict[str, dict[str, float | int]]
    prompt_token_source: str | None = None
    completion_token_source: str | None = None


def _synchronize_if_needed(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class FirstStepTimingProcessor(LogitsProcessor):
    """Records when first-token logits are produced, which is our closest observable prefill boundary."""

    def __init__(self, *, synchronize_cuda: bool):
        self.synchronize_cuda = synchronize_cuda
        self.first_step_time: float | None = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.first_step_time is None:
            _synchronize_if_needed(self.synchronize_cuda)
            self.first_step_time = time.perf_counter()
        return scores


class FirstTokenTimingStreamer(BaseStreamer):
    """Records when the first generated token is emitted while skipping the prompt tokens sent to the streamer."""

    def __init__(self, *, skip_prompt: bool = True, synchronize_cuda: bool):
        self.skip_prompt = skip_prompt
        self.synchronize_cuda = synchronize_cuda
        self.next_tokens_are_prompt = skip_prompt
        self.first_token_time: float | None = None

    def put(self, value) -> None:
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("FirstTokenTimingStreamer only supports batch size 1")
        if len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        if self.first_token_time is None and value.numel() > 0:
            _synchronize_if_needed(self.synchronize_cuda)
            self.first_token_time = time.perf_counter()

    def end(self) -> None:
        return None

class NoteGenerator:
    def __init__(self, model_config: ModelConfig, generation_config: GenerationConfig):
        self.model_config = model_config
        self.generation_config = generation_config
        bundle = load_causal_lm_bundle(model_config, padding_side="right")
        self.tokenizer = bundle.tokenizer
        self.model = bundle.model

    def _build_generation_kwargs(
        self,
        *,
        streamer: BaseStreamer | None = None,
        timing_processor: LogitsProcessor | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.generation_config.max_new_tokens,
            "do_sample": self.generation_config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if streamer is not None:
            kwargs["streamer"] = streamer
        if timing_processor is not None:
            kwargs["logits_processor"] = LogitsProcessorList([timing_processor])
        if self.generation_config.do_sample:
            kwargs["temperature"] = self.generation_config.temperature
            kwargs["top_p"] = self.generation_config.top_p
        return kwargs

    def generate(self, prompt: str, benchmark_config: BenchmarkConfig) -> GenerationResult:
        """
        Measure generation in a way that stays honest about what we can observe in the current HF stack.

        - tokenization/device_transfer are timed separately on the host side
        - prefill_latency_sec is approximated by the first logits-processor callback
        - ttft_sec is measured from `.generate()` start until the first generated token reaches the streamer
        - latency_sec is full decoding latency from `.generate()` start to completion
        """
        measured_runs: list[GenerationRunMetrics] = []
        last_text = ""
        measured_run_count = benchmark_config.repeat_runs if benchmark_config.enabled else 1
        total_runs = (benchmark_config.warmup_runs if benchmark_config.enabled else 0) + measured_run_count

        for run_index in range(total_runs):
            record_run = run_index >= (benchmark_config.warmup_runs if benchmark_config.enabled else 0)
            _release_cuda_memory()

            tokenization_start = time.perf_counter()
            inputs = self.tokenizer(prompt, return_tensors="pt")
            tokenization_end = time.perf_counter()

            device_transfer_start = time.perf_counter()
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            _synchronize_if_needed(benchmark_config.synchronize_cuda)
            device_transfer_end = time.perf_counter()

            prompt_tokens = int(inputs["input_ids"].shape[-1])
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            timing_processor = FirstStepTimingProcessor(synchronize_cuda=benchmark_config.synchronize_cuda)
            streamer = FirstTokenTimingStreamer(
                skip_prompt=True,
                synchronize_cuda=benchmark_config.synchronize_cuda,
            )

            _synchronize_if_needed(benchmark_config.synchronize_cuda)
            generation_start = time.perf_counter()
            generation_kwargs = self._build_generation_kwargs(
                streamer=streamer,
                timing_processor=timing_processor,
            )
            try:
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                _release_cuda_memory()
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            _synchronize_if_needed(benchmark_config.synchronize_cuda)
            generation_end = time.perf_counter()

            generated_ids = outputs[0][prompt_tokens:]
            last_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            completion_tokens = int(generated_ids.shape[-1])
            total_tokens = prompt_tokens + completion_tokens

            ttft_sec = (
                streamer.first_token_time - generation_start if streamer.first_token_time is not None else None
            )
            prefill_latency_sec = (
                timing_processor.first_step_time - generation_start
                if timing_processor.first_step_time is not None
                else None
            )
            latency_sec = generation_end - generation_start
            decode_latency_sec = (
                generation_end - streamer.first_token_time if streamer.first_token_time is not None else None
            )
            peak_gpu_mem_gb = None
            if torch.cuda.is_available():
                peak_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

            if record_run:
                measured_runs.append(
                    GenerationRunMetrics(
                        tokenization_sec=tokenization_end - tokenization_start,
                        device_transfer_sec=device_transfer_end - device_transfer_start,
                        prefill_latency_sec=prefill_latency_sec,
                        ttft_sec=ttft_sec,
                        decode_latency_sec=decode_latency_sec,
                        latency_sec=latency_sec,
                        total_inference_sec=(generation_end - tokenization_start),
                        peak_gpu_mem_gb=peak_gpu_mem_gb,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        prompt_token_source="local_tokenizer",
                        completion_token_source="local_tokenizer",
                    )
                )
            del outputs
            del generated_ids
            del inputs
            _release_cuda_memory()

        if not measured_runs:
            raise RuntimeError("No measured generation runs were recorded. Check benchmark.repeat_runs.")

        primary_run = measured_runs[0]
        benchmark_summary = aggregate_numeric_records([run.as_dict() for run in measured_runs])

        return GenerationResult(
            text=last_text,
            prompt_tokens=primary_run.prompt_tokens,
            completion_tokens=primary_run.completion_tokens,
            total_tokens=primary_run.total_tokens,
            tokenization_sec=primary_run.tokenization_sec,
            device_transfer_sec=primary_run.device_transfer_sec,
            prefill_latency_sec=primary_run.prefill_latency_sec,
            latency_sec=primary_run.latency_sec,
            ttft_sec=primary_run.ttft_sec if primary_run.ttft_sec is not None else primary_run.latency_sec,
            decode_latency_sec=primary_run.decode_latency_sec,
            total_inference_sec=primary_run.total_inference_sec,
            peak_gpu_mem_gb=primary_run.peak_gpu_mem_gb,
            benchmark_runs=measured_runs,
            benchmark_summary=benchmark_summary,
            prompt_token_source=primary_run.prompt_token_source,
            completion_token_source=primary_run.completion_token_source,
        )

    def generate_batch(self, prompts: list[str], benchmark_config: BenchmarkConfig) -> BatchGenerationResult:
        """
        True batched HF generation for multiple prompts in a single `.generate()` call.

        This is intentionally batch-oriented:
        - all prompts are tokenized together with padding/truncation
        - inputs move to device together
        - throughput reflects total generated tokens across the batch
        - TTFT/prefill/decode split is left as `None` because the current single-request
          streamer/timing hooks are not batch-safe
        """
        if not prompts:
            raise ValueError("generate_batch requires at least one prompt")

        measured_runs: list[GenerationRunMetrics] = []
        last_texts: list[str] = []
        last_prompt_tokens: list[int] = []
        last_completion_tokens: list[int] = []
        last_total_tokens: list[int] = []

        measured_run_count = benchmark_config.repeat_runs if benchmark_config.enabled else 1
        total_runs = (benchmark_config.warmup_runs if benchmark_config.enabled else 0) + measured_run_count

        for run_index in range(total_runs):
            record_run = run_index >= (benchmark_config.warmup_runs if benchmark_config.enabled else 0)
            _release_cuda_memory()

            tokenization_start = time.perf_counter()
            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokenization_end = time.perf_counter()

            device_transfer_start = time.perf_counter()
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            _synchronize_if_needed(benchmark_config.synchronize_cuda)
            device_transfer_end = time.perf_counter()

            prompt_lengths = [int(length) for length in inputs["attention_mask"].sum(dim=1).tolist()]
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            _synchronize_if_needed(benchmark_config.synchronize_cuda)
            generation_start = time.perf_counter()
            generation_kwargs = self._build_generation_kwargs()
            try:
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation_kwargs,
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                _release_cuda_memory()
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation_kwargs,
                )
            _synchronize_if_needed(benchmark_config.synchronize_cuda)
            generation_end = time.perf_counter()

            run_texts: list[str] = []
            run_prompt_tokens: list[int] = []
            run_completion_tokens: list[int] = []
            run_total_tokens: list[int] = []
            pad_token_id = self.tokenizer.pad_token_id

            for row_index, prompt_length in enumerate(prompt_lengths):
                generated_ids = outputs[row_index][prompt_length:]
                if pad_token_id is not None:
                    non_pad_mask = generated_ids.ne(pad_token_id)
                    generated_ids = generated_ids[non_pad_mask]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                completion_tokens = int(generated_ids.shape[-1])
                run_texts.append(generated_text)
                run_prompt_tokens.append(prompt_length)
                run_completion_tokens.append(completion_tokens)
                run_total_tokens.append(prompt_length + completion_tokens)

            last_texts = run_texts
            last_prompt_tokens = run_prompt_tokens
            last_completion_tokens = run_completion_tokens
            last_total_tokens = run_total_tokens

            peak_gpu_mem_gb = None
            if torch.cuda.is_available():
                peak_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

            if record_run:
                total_completion_tokens = sum(run_completion_tokens)
                measured_runs.append(
                    GenerationRunMetrics(
                        tokenization_sec=tokenization_end - tokenization_start,
                        device_transfer_sec=device_transfer_end - device_transfer_start,
                        prefill_latency_sec=None,
                        ttft_sec=None,
                        decode_latency_sec=None,
                        latency_sec=generation_end - generation_start,
                        total_inference_sec=(generation_end - tokenization_start),
                        peak_gpu_mem_gb=peak_gpu_mem_gb,
                        prompt_tokens=sum(run_prompt_tokens),
                        completion_tokens=total_completion_tokens,
                        total_tokens=sum(run_total_tokens),
                        prompt_token_source="local_tokenizer",
                        completion_token_source="local_tokenizer",
                    )
                )
            del outputs
            del inputs
            _release_cuda_memory()

        if not measured_runs:
            raise RuntimeError("No measured batch generation runs were recorded. Check benchmark.repeat_runs.")

        primary_run = measured_runs[0]
        benchmark_summary = aggregate_numeric_records([run.as_dict() for run in measured_runs])
        throughput_tok_per_sec = (
            None if primary_run.latency_sec == 0 else sum(last_completion_tokens) / primary_run.latency_sec
        )

        return BatchGenerationResult(
            texts=last_texts,
            prompt_tokens=last_prompt_tokens,
            completion_tokens=last_completion_tokens,
            total_tokens=last_total_tokens,
            active_requests=len(prompts),
            total_completion_tokens=sum(last_completion_tokens),
            tokenization_sec=primary_run.tokenization_sec,
            device_transfer_sec=primary_run.device_transfer_sec,
            prefill_latency_sec=primary_run.prefill_latency_sec,
            latency_sec=primary_run.latency_sec,
            ttft_sec=primary_run.ttft_sec,
            decode_latency_sec=primary_run.decode_latency_sec,
            total_inference_sec=primary_run.total_inference_sec,
            throughput_tok_per_sec=throughput_tok_per_sec,
            peak_gpu_mem_gb=primary_run.peak_gpu_mem_gb,
            benchmark_runs=measured_runs,
            benchmark_summary=benchmark_summary,
            prompt_token_source=primary_run.prompt_token_source,
            completion_token_source=primary_run.completion_token_source,
        )
