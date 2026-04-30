import os
import unittest
from unittest import mock

from clinical_speech.config import BackendSpec, BenchmarkConfig, GenerationConfig, ModelConfig, SystemsBackendSpecConfig
from clinical_speech.models.hosted_together import TogetherHostedGenerator, TogetherHostedModel


class _FakeStreamingResponse:
    def __init__(self, *, status_code: int = 200, lines: list[str] | None = None, text: str = ""):
        self.status_code = status_code
        self._lines = lines or []
        self.text = text

    def iter_lines(self):
        return iter(self._lines)

    def read(self) -> bytes:
        return self.text.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeJSONResponse:
    def __init__(self, payload: dict, *, status_code: int = 200, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else ""

    def json(self):
        return self._payload


class _FakeHTTPXClient:
    def __init__(self, *, stream_response=None, post_response=None, **kwargs):
        self.stream_response = stream_response
        self.post_response = post_response
        self.kwargs = kwargs

    def stream(self, method: str, url: str, json: dict):
        del method, url, json
        return self.stream_response

    def post(self, url: str, json: dict):
        del url, json
        return self.post_response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TogetherHostedGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model_config = ModelConfig(
            asr_model_id="openai/whisper-small",
            llm_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            device="cuda",
            dtype="float16",
            load_in_8bit=False,
            attn_implementation="sdpa",
        )
        self.generation_config = GenerationConfig(max_new_tokens=16, temperature=0.0, top_p=1.0, do_sample=False)
        self.benchmark_config = BenchmarkConfig(enabled=False)
        self.env_patch = mock.patch.dict(os.environ, {"TOGETHER_API_KEY": "test-key"}, clear=False)
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()

    def test_streaming_response_uses_provider_usage_when_available(self) -> None:
        spec = SystemsBackendSpecConfig(
            provider="together_hosted",
            stream=True,
            stream_include_usage=True,
            api_key_env="TOGETHER_API_KEY",
        )
        generator = TogetherHostedGenerator(
            model_config=self.model_config,
            generation_config=self.generation_config,
            backend_spec=spec,
        )
        lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15}}',
            "data: [DONE]",
        ]
        client = _FakeHTTPXClient(stream_response=_FakeStreamingResponse(lines=lines))
        with mock.patch("clinical_speech.models.hosted_together.httpx.Client", return_value=client):
            result = generator.generate("prompt", self.benchmark_config)

        self.assertEqual(result.text, "Hello world")
        self.assertIsNotNone(result.ttft_sec)
        self.assertEqual(result.prompt_tokens, 12)
        self.assertEqual(result.completion_tokens, 3)
        self.assertEqual(result.total_tokens, 15)
        self.assertIsNone(result.peak_gpu_mem_gb)
        self.assertEqual(result.prompt_token_source, "provider_usage")
        self.assertEqual(result.completion_token_source, "provider_usage")

    def test_streaming_response_falls_back_to_local_completion_token_count(self) -> None:
        spec = SystemsBackendSpecConfig(
            provider="together_hosted",
            stream=True,
            stream_include_usage=False,
            api_key_env="TOGETHER_API_KEY",
        )
        generator = TogetherHostedGenerator(
            model_config=self.model_config,
            generation_config=self.generation_config,
            backend_spec=spec,
        )
        lines = [
            'data: {"choices":[{"delta":{"content":"hello there"}}]}',
            "data: [DONE]",
        ]

        class _FakeTokenizer:
            def __call__(self, text, add_special_tokens=False, return_attention_mask=False):
                self.last_text = text
                return {"input_ids": [1, 2, 3, 4]}

        client = _FakeHTTPXClient(stream_response=_FakeStreamingResponse(lines=lines))
        with mock.patch("clinical_speech.models.hosted_together.httpx.Client", return_value=client):
            with mock.patch.object(generator, "_get_tokenizer", return_value=_FakeTokenizer()):
                result = generator.generate("prompt", self.benchmark_config)

        self.assertEqual(result.text, "hello there")
        self.assertIsNone(result.prompt_tokens)
        self.assertEqual(result.completion_tokens, 4)
        self.assertIsNone(result.total_tokens)
        self.assertEqual(result.prompt_token_source, "unavailable")
        self.assertEqual(result.completion_token_source, "estimated_local_tokenizer")

    def test_cloudflare_block_message_is_compact_and_actionable(self) -> None:
        spec = SystemsBackendSpecConfig(
            provider="together_hosted",
            stream=True,
            stream_include_usage=True,
            api_key_env="TOGETHER_API_KEY",
        )
        generator = TogetherHostedGenerator(
            model_config=self.model_config,
            generation_config=self.generation_config,
            backend_spec=spec,
        )
        html = """
        <!doctype html>
        <html>
          <head><title>Access denied | api.together.xyz used Cloudflare</title></head>
          <body>
            <h1>Error 1010</h1>
            <span>Ray ID: 9f08e5d0fea51972</span>
            <h2>Access denied</h2>
          </body>
        </html>
        """
        client = _FakeHTTPXClient(
            stream_response=_FakeStreamingResponse(status_code=403, text=html),
        )
        with mock.patch("clinical_speech.models.hosted_together.httpx.Client", return_value=client):
            with self.assertRaises(RuntimeError) as ctx:
                generator.generate("prompt", self.benchmark_config)

        message = str(ctx.exception)
        self.assertIn("blocked by Cloudflare", message)
        self.assertIn("Ray ID: 9f08e5d0fea51972", message)
        self.assertNotIn("<!doctype html>", message)

    def test_compatibility_wrapper_supports_simple_generate_api(self) -> None:
        spec = BackendSpec(
            provider="together_hosted",
            api_model_id="example/model",
            llm_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            api_key_env="TOGETHER_API_KEY",
            stream=False,
        )
        model = TogetherHostedModel(spec)
        with mock.patch.object(
            model._generator,
            "_non_stream_completion",
            return_value=("hello", None, 0.25, mock.Mock(prompt_tokens=1, completion_tokens=1, total_tokens=2, prompt_token_source="provider_usage", completion_token_source="provider_usage")),
        ):
            text = model.generate("prompt")
        self.assertEqual(text, "hello")


if __name__ == "__main__":
    unittest.main()
