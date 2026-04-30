from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from transformers import AutoTokenizer

from clinical_speech.benchmarking import aggregate_numeric_records
from clinical_speech.config import BenchmarkConfig, GenerationConfig, ModelConfig, SystemsBackendSpecConfig
from clinical_speech.models.note_generator import GenerationResult, GenerationRunMetrics
from clinical_speech.storage import get_runtime_paths


DEFAULT_TOGETHER_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_TOGETHER_USER_AGENT = "clinical-speech-bench/0.1 (openai-compatible httpx client)"


@dataclass
class HostedUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    prompt_token_source: str | None = None
    completion_token_source: str | None = None


class TogetherHostedGenerator:
    def __init__(
        self,
        *,
        model_config: ModelConfig,
        generation_config: GenerationConfig,
        backend_spec: SystemsBackendSpecConfig,
    ) -> None:
        self.model_config = model_config
        self.generation_config = generation_config
        self.backend_spec = backend_spec
        self.base_url = self._resolve_base_url(backend_spec)
        self.api_key = self._resolve_api_key(backend_spec)
        self._tokenizer = None

    def _resolve_api_key(self, backend_spec: SystemsBackendSpecConfig) -> str:
        env_name = backend_spec.api_key_env or "TOGETHER_API_KEY"
        api_key = os.environ.get(env_name)
        if not api_key:
            raise RuntimeError(
                f"Together hosted backend requires environment variable {env_name} to be set"
            )
        return api_key

    def _resolve_base_url(self, backend_spec: SystemsBackendSpecConfig) -> str:
        if backend_spec.base_url:
            return backend_spec.base_url.rstrip("/")
        if backend_spec.base_url_env:
            override = os.environ.get(backend_spec.base_url_env)
            if override:
                return override.rstrip("/")
        return os.environ.get("TOGETHER_BASE_URL", DEFAULT_TOGETHER_BASE_URL).rstrip("/")

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        runtime_paths = get_runtime_paths()
        tokenizer_model_id = self.backend_spec.tokenizer_model_id or self.model_config.llm_model_id
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_id,
            cache_dir=str(runtime_paths.hf),
        )
        return self._tokenizer

    def _count_completion_tokens(self, text: str) -> int | None:
        try:
            tokenizer = self._get_tokenizer()
            ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        except Exception:
            return None
        return len(ids)

    def _make_payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.backend_spec.api_model_id or self.model_config.llm_model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.generation_config.max_new_tokens,
            "temperature": self.generation_config.temperature,
            "top_p": self.generation_config.top_p,
            "stream": self.backend_spec.stream,
        }
        if self.backend_spec.stream and self.backend_spec.stream_include_usage:
            payload["stream_options"] = {"include_usage": True}
        if not self.generation_config.do_sample:
            payload["temperature"] = 0.0
            payload["top_p"] = 1.0
        return payload

    def _make_headers(self, *, accept: str | None = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": DEFAULT_TOGETHER_USER_AGENT,
        }
        if accept:
            headers["Accept"] = accept
        return headers

    def _format_http_error(self, response: httpx.Response) -> str:
        try:
            detail = response.text.strip()
        except httpx.ResponseNotRead:
            detail = response.read().decode("utf-8", errors="replace").strip()
        lowered = detail.lower()
        if "cloudflare" in lowered and ("error 1010" in lowered or "access denied" in lowered):
            ray_match = re.search(r"Ray ID:\s*([A-Za-z0-9]+)", detail)
            ray_suffix = f" (Ray ID: {ray_match.group(1)})" if ray_match else ""
            return (
                f"Together hosted request was blocked by Cloudflare (HTTP {response.status_code}{ray_suffix}). "
                "This is usually an IP/network or WAF restriction rather than a benchmark bug. "
                "Verify TOGETHER_API_KEY, TOGETHER_BASE_URL, and the configured Together model ID. "
                "If those are correct, retry from a different network or IP."
            )
        if not detail:
            return f"Together hosted request failed with HTTP {response.status_code}"
        compact_detail = re.sub(r"\s+", " ", detail)
        if compact_detail.startswith("<!doctype html"):
            compact_detail = "Received an HTML error page from Together instead of JSON/SSE."
        if len(compact_detail) > 240:
            compact_detail = compact_detail[:237] + "..."
        return f"Together hosted request failed with HTTP {response.status_code}: {compact_detail}"

    def _stream_completion(self, prompt: str) -> tuple[str, float | None, float, HostedUsage]:
        url = f"{self.base_url}/chat/completions"
        payload = self._make_payload(prompt)
        text_parts: list[str] = []
        ttft_sec: float | None = None
        usage = HostedUsage(prompt_tokens=None, completion_tokens=None, total_tokens=None)
        start = time.perf_counter()
        try:
            with httpx.Client(
                timeout=self.backend_spec.timeout_sec,
                headers=self._make_headers(accept="text/event-stream"),
            ) as client:
                with client.stream("POST", url, json=payload) as response:
                    if response.status_code >= 400:
                        response.read()
                        raise RuntimeError(self._format_http_error(response))
                    for line in response.iter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        if chunk.get("usage"):
                            usage = HostedUsage(
                                prompt_tokens=chunk["usage"].get("prompt_tokens"),
                                completion_tokens=chunk["usage"].get("completion_tokens"),
                                total_tokens=chunk["usage"].get("total_tokens"),
                                prompt_token_source="provider_usage",
                                completion_token_source="provider_usage",
                            )
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            if ttft_sec is None:
                                ttft_sec = time.perf_counter() - start
                            text_parts.append(content)
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Together hosted request timed out after {self.backend_spec.timeout_sec:.1f}s"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Together hosted request failed: {exc}") from exc
        latency_sec = time.perf_counter() - start
        return "".join(text_parts).strip(), ttft_sec, latency_sec, usage

    def _non_stream_completion(self, prompt: str) -> tuple[str, float | None, float, HostedUsage]:
        url = f"{self.base_url}/chat/completions"
        payload = self._make_payload(prompt)
        payload["stream"] = False
        payload.pop("stream_options", None)
        start = time.perf_counter()
        try:
            with httpx.Client(timeout=self.backend_spec.timeout_sec, headers=self._make_headers()) as client:
                response = client.post(url, json=payload)
                if response.status_code >= 400:
                    raise RuntimeError(self._format_http_error(response))
                body = response.json()
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Together hosted request timed out after {self.backend_spec.timeout_sec:.1f}s"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Together hosted request failed: {exc}") from exc
        latency_sec = time.perf_counter() - start
        usage_payload = body.get("usage") or {}
        usage = HostedUsage(
            prompt_tokens=usage_payload.get("prompt_tokens"),
            completion_tokens=usage_payload.get("completion_tokens"),
            total_tokens=usage_payload.get("total_tokens"),
            prompt_token_source="provider_usage" if usage_payload.get("prompt_tokens") is not None else None,
            completion_token_source="provider_usage" if usage_payload.get("completion_tokens") is not None else None,
        )
        content = ((body.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        return content.strip(), None, latency_sec, usage

    def generate(self, prompt: str, benchmark_config: BenchmarkConfig) -> GenerationResult:
        if self.backend_spec.stream:
            text, ttft_sec, latency_sec, usage = self._stream_completion(prompt)
        else:
            text, ttft_sec, latency_sec, usage = self._non_stream_completion(prompt)

        completion_tokens = usage.completion_tokens
        completion_token_source = usage.completion_token_source
        if completion_tokens is None:
            completion_tokens = self._count_completion_tokens(text)
            completion_token_source = "estimated_local_tokenizer" if completion_tokens is not None else "unavailable"
        prompt_token_source = usage.prompt_token_source or ("unavailable" if usage.prompt_tokens is None else "provider_usage")
        total_tokens = usage.total_tokens
        if total_tokens is None and usage.prompt_tokens is not None and completion_tokens is not None:
            total_tokens = usage.prompt_tokens + completion_tokens

        run = GenerationRunMetrics(
            tokenization_sec=0.0,
            device_transfer_sec=0.0,
            prefill_latency_sec=None,
            ttft_sec=ttft_sec,
            decode_latency_sec=None if ttft_sec is None else max(0.0, latency_sec - ttft_sec),
            latency_sec=latency_sec,
            total_inference_sec=latency_sec,
            peak_gpu_mem_gb=None,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_token_source=prompt_token_source,
            completion_token_source=completion_token_source,
        )
        return GenerationResult(
            text=text,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            total_tokens=run.total_tokens,
            tokenization_sec=run.tokenization_sec,
            device_transfer_sec=run.device_transfer_sec,
            prefill_latency_sec=run.prefill_latency_sec,
            latency_sec=run.latency_sec,
            ttft_sec=run.ttft_sec,
            decode_latency_sec=run.decode_latency_sec,
            total_inference_sec=run.total_inference_sec,
            peak_gpu_mem_gb=run.peak_gpu_mem_gb,
            benchmark_runs=[run],
            benchmark_summary=aggregate_numeric_records([run.as_dict()]),
            prompt_token_source=run.prompt_token_source,
            completion_token_source=run.completion_token_source,
        )


class TogetherHostedModel:
    """
    Small convenience wrapper for ad-hoc hosted inference outside the benchmark runner.

    This keeps the benchmark-facing TogetherHostedGenerator unchanged while supporting
    a simpler interface for quick smoke tests:

        model = TogetherHostedModel(spec)
        text = model.generate("prompt")
    """

    def __init__(
        self,
        backend_spec: SystemsBackendSpecConfig,
        *,
        model_config: ModelConfig | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        resolved_model_config = model_config or ModelConfig(
            asr_model_id="openai/whisper-small",
            llm_model_id=backend_spec.llm_model_id or backend_spec.api_model_id or "meta-llama/Meta-Llama-3-8B-Instruct",
            device="cpu",
        )
        resolved_generation_config = generation_config or GenerationConfig(
            max_new_tokens=512,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        self._generator = TogetherHostedGenerator(
            model_config=resolved_model_config,
            generation_config=resolved_generation_config,
            backend_spec=backend_spec,
        )

    def generate_result(
        self,
        prompt: str,
        benchmark_config: BenchmarkConfig | None = None,
    ) -> GenerationResult:
        return self._generator.generate(prompt, benchmark_config or BenchmarkConfig(enabled=False))

    def generate(
        self,
        prompt: str,
        benchmark_config: BenchmarkConfig | None = None,
    ) -> str:
        return self.generate_result(prompt, benchmark_config).text
