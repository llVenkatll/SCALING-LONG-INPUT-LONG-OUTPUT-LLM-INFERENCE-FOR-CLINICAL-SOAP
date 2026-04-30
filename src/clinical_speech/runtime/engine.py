from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from clinical_speech.config import GenerationConfig, ModelConfig, RuntimeConfig
from clinical_speech.kernels import pack_left_padded_sequences
from clinical_speech.runtime.block_manager import KVCacheBlockManager
from clinical_speech.runtime.kv_cache import (
    PagedKVCache,
    append_attention_tokens,
    infer_mistral_kv_layout,
    position_ids_from_attention_mask,
)
from clinical_speech.models.factory import load_causal_lm_bundle


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class RuntimeRequestResult:
    request_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokenization_sec: float
    device_transfer_sec: float
    prefill_latency_sec: float
    ttft_sec: float
    decode_latency_sec: float
    latency_sec: float
    total_inference_sec: float
    peak_gpu_mem_gb: float | None

    def as_dict(self) -> dict[str, float | int | str | None]:
        return {
            "request_id": self.request_id,
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tokenization_sec": self.tokenization_sec,
            "device_transfer_sec": self.device_transfer_sec,
            "prefill_latency_sec": self.prefill_latency_sec,
            "ttft_sec": self.ttft_sec,
            "decode_latency_sec": self.decode_latency_sec,
            "latency_sec": self.latency_sec,
            "total_inference_sec": self.total_inference_sec,
            "peak_gpu_mem_gb": self.peak_gpu_mem_gb,
        }


@dataclass
class RuntimeBatchResult:
    batch_size: int
    tokenization_sec: float
    device_transfer_sec: float
    prefill_latency_sec: float
    batch_latency_sec: float
    total_generated_tokens: int
    throughput_tok_per_sec: float
    requests_per_sec: float
    peak_gpu_mem_gb: float | None
    kv_cache_metrics: dict[str, float | int]
    requests: list[RuntimeRequestResult]
    scheduler_steps: int
    admission_wait_sec: float | None = None
    queue_depth_at_admit: int | None = None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "batch_size": self.batch_size,
            "tokenization_sec": self.tokenization_sec,
            "device_transfer_sec": self.device_transfer_sec,
            "prefill_latency_sec": self.prefill_latency_sec,
            "batch_latency_sec": self.batch_latency_sec,
            "total_generated_tokens": self.total_generated_tokens,
            "throughput_tok_per_sec": self.throughput_tok_per_sec,
            "requests_per_sec": self.requests_per_sec,
            "peak_gpu_mem_gb": self.peak_gpu_mem_gb,
            "scheduler_steps": self.scheduler_steps,
            "admission_wait_sec": self.admission_wait_sec,
            "queue_depth_at_admit": self.queue_depth_at_admit,
            **{f"kv_{key}": value for key, value in self.kv_cache_metrics.items()},
        }


class ManualBatchEngine:
    def __init__(self, model_config: ModelConfig, generation_config: GenerationConfig, runtime_config: RuntimeConfig):
        self.model_config = model_config
        self.generation_config = generation_config
        self.runtime_config = runtime_config
        bundle = load_causal_lm_bundle(model_config, padding_side="left")
        self.tokenizer = bundle.tokenizer
        self.model = bundle.model
        self.block_manager: KVCacheBlockManager | None = None
        self.paged_cache: PagedKVCache | None = None
        self.eos_token_id = self.tokenizer.eos_token_id

        if runtime_config.backend == "mistral_paged" and getattr(self.model.config, "model_type", None) != "mistral":
            raise ValueError("The mistral_paged runtime is only supported for Mistral-family checkpoints")

    def _sample_next_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        if self.generation_config.do_sample:
            temperature = max(self.generation_config.temperature, 1e-5)
            probs = torch.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        return torch.argmax(logits, dim=-1)

    def _prepare_batch_inputs(self, prompts: list[str]) -> tuple[dict[str, torch.Tensor], list[int], float, float]:
        tokenization_start = time.perf_counter()
        encoded = [
            self.tokenizer(prompt, add_special_tokens=True, return_attention_mask=False)["input_ids"]
            for prompt in prompts
        ]
        prompt_lengths = [len(ids) for ids in encoded]
        tokenization_end = time.perf_counter()

        device_transfer_start = time.perf_counter()
        sequences = [torch.tensor(ids, dtype=torch.long) for ids in encoded]
        input_ids, attention_mask = pack_left_padded_sequences(
            sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            device=self.model.device,
            pad_to_multiple_of=self.runtime_config.pad_to_multiple_of,
            use_triton=self.runtime_config.triton_enabled,
        )
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        _sync_cuda()
        device_transfer_end = time.perf_counter()
        return batch, prompt_lengths, tokenization_end - tokenization_start, device_transfer_end - device_transfer_start

    def _ensure_runtime_cache(self) -> PagedKVCache:
        if self.paged_cache is not None:
            return self.paged_cache
        layout = infer_mistral_kv_layout(self.model)
        budget_bytes = int(self.runtime_config.max_cache_budget_gb * (1024 ** 3))
        self.block_manager = KVCacheBlockManager(
            layout=layout,
            block_size_tokens=self.runtime_config.block_size_tokens,
            max_cache_budget_bytes=budget_bytes,
        )
        self.paged_cache = PagedKVCache(
            layout=layout,
            block_manager=self.block_manager,
            block_size_tokens=self.runtime_config.block_size_tokens,
            use_triton_gather=self.runtime_config.triton_paged_kv_enabled,
        )
        return self.paged_cache

    def _record_peak_snapshot(self, current_peak: dict[str, float | int], paged_cache: PagedKVCache) -> dict[str, float | int]:
        snapshot = paged_cache.snapshot()
        if snapshot["allocated_bytes"] > current_peak["allocated_bytes"]:
            return snapshot
        return current_peak

    def run_batch(self, request_ids: list[str], prompts: list[str]) -> RuntimeBatchResult:
        if len(prompts) == 0:
            raise ValueError("run_batch requires at least one prompt")
        if len(prompts) > self.runtime_config.max_batch_size:
            raise ValueError(
                f"Batch of size {len(prompts)} exceeds runtime.max_batch_size={self.runtime_config.max_batch_size}"
            )
        if self.runtime_config.scheduler_mode == "none" and len(prompts) != 1:
            raise ValueError("scheduler_mode=none only supports a single request")

        paged_cache = self._ensure_runtime_cache()
        batch_inputs, prompt_lengths, tokenization_sec, device_transfer_sec = self._prepare_batch_inputs(prompts)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        generation_start = time.perf_counter()
        request_text_tokens: dict[str, list[int]] = {request_id: [] for request_id in request_ids}
        finish_times: dict[str, float] = {}
        scheduler_steps = 0

        with torch.inference_mode():
            _sync_cuda()
            prefill_start = time.perf_counter()
            prefill_position_ids = position_ids_from_attention_mask(batch_inputs["attention_mask"])
            prefill_cache_lengths = [int(batch_inputs["input_ids"].shape[-1])] * len(request_ids)
            paged_cache.begin_forward(
                request_ids=request_ids,
                token_lengths=prefill_cache_lengths,
                attention_mask_length=int(batch_inputs["attention_mask"].shape[-1]),
            )
            prefill_outputs = self.model(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                position_ids=prefill_position_ids,
                past_key_values=paged_cache,
                use_cache=True,
                return_dict=True,
            )
            paged_cache.clear_batch_view()
            _sync_cuda()
            prefill_end = time.perf_counter()

            peak_kv_snapshot = paged_cache.snapshot()
            next_tokens = self._sample_next_tokens(prefill_outputs.logits[:, -1, :])
            ttft_sec = prefill_end - generation_start

            for row_index, request_id in enumerate(request_ids):
                token_id = int(next_tokens[row_index].item())
                request_text_tokens[request_id].append(token_id)
                if (
                    token_id == self.eos_token_id
                    or len(request_text_tokens[request_id]) >= self.generation_config.max_new_tokens
                ):
                    finish_times[request_id] = prefill_end

            active_request_ids = list(request_ids)
            active_attention_mask = append_attention_tokens(batch_inputs["attention_mask"], 1)

            while True:
                finished_request_ids = [request_id for request_id in active_request_ids if request_id in finish_times]
                if finished_request_ids:
                    paged_cache.free_requests(finished_request_ids)
                unfinished_indices = [
                    index
                    for index, request_id in enumerate(active_request_ids)
                    if request_id not in finish_times
                ]
                if not unfinished_indices:
                    break

                scheduler_steps += 1
                active_request_ids = [active_request_ids[index] for index in unfinished_indices]
                next_tokens = next_tokens[unfinished_indices]
                active_attention_mask = active_attention_mask[unfinished_indices]

                decode_position_ids = position_ids_from_attention_mask(active_attention_mask)[:, -1:]
                paged_cache.begin_forward(
                    request_ids=active_request_ids,
                    token_lengths=[1] * len(active_request_ids),
                    attention_mask_length=int(active_attention_mask.shape[-1]),
                )
                _sync_cuda()
                step_outputs = self.model(
                    input_ids=next_tokens.unsqueeze(-1),
                    attention_mask=active_attention_mask,
                    position_ids=decode_position_ids,
                    past_key_values=paged_cache,
                    use_cache=True,
                    return_dict=True,
                )
                paged_cache.clear_batch_view()
                _sync_cuda()
                step_time = time.perf_counter()
                peak_kv_snapshot = self._record_peak_snapshot(peak_kv_snapshot, paged_cache)

                next_tokens = self._sample_next_tokens(step_outputs.logits[:, -1, :])
                for row_index, request_id in enumerate(active_request_ids):
                    token_id = int(next_tokens[row_index].item())
                    request_text_tokens[request_id].append(token_id)
                    if (
                        token_id == self.eos_token_id
                        or len(request_text_tokens[request_id]) >= self.generation_config.max_new_tokens
                    ):
                        finish_times[request_id] = step_time

                active_attention_mask = append_attention_tokens(active_attention_mask, 1)

        generation_end = max(finish_times.values()) if finish_times else prefill_end
        peak_gpu_mem_gb = None
        if torch.cuda.is_available():
            peak_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        request_results: list[RuntimeRequestResult] = []
        total_generated_tokens = 0
        for request_id, prompt_length in zip(request_ids, prompt_lengths, strict=True):
            token_ids = request_text_tokens[request_id]
            if token_ids and token_ids[-1] == self.eos_token_id:
                decoded_ids = token_ids[:-1]
            else:
                decoded_ids = token_ids
            text = self.tokenizer.decode(decoded_ids, skip_special_tokens=True).strip()
            completion_tokens = len(token_ids)
            total_generated_tokens += completion_tokens
            latency_sec = finish_times[request_id] - generation_start
            total_inference_sec = tokenization_sec + device_transfer_sec + latency_sec
            request_results.append(
                RuntimeRequestResult(
                    request_id=request_id,
                    text=text,
                    prompt_tokens=prompt_length,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_length + completion_tokens,
                    tokenization_sec=tokenization_sec,
                    device_transfer_sec=device_transfer_sec,
                    prefill_latency_sec=prefill_end - prefill_start,
                    ttft_sec=ttft_sec,
                    decode_latency_sec=max(0.0, latency_sec - ttft_sec),
                    latency_sec=latency_sec,
                    total_inference_sec=total_inference_sec,
                    peak_gpu_mem_gb=peak_gpu_mem_gb,
                )
            )
            paged_cache.free_requests([request_id])

        batch_latency_sec = generation_end - generation_start
        return RuntimeBatchResult(
            batch_size=len(prompts),
            tokenization_sec=tokenization_sec,
            device_transfer_sec=device_transfer_sec,
            prefill_latency_sec=prefill_end - prefill_start,
            batch_latency_sec=batch_latency_sec,
            total_generated_tokens=total_generated_tokens,
            throughput_tok_per_sec=0.0 if batch_latency_sec == 0 else total_generated_tokens / batch_latency_sec,
            requests_per_sec=0.0 if batch_latency_sec == 0 else len(prompts) / batch_latency_sec,
            peak_gpu_mem_gb=peak_gpu_mem_gb,
            kv_cache_metrics=peak_kv_snapshot,
            requests=request_results,
            scheduler_steps=scheduler_steps,
        )
