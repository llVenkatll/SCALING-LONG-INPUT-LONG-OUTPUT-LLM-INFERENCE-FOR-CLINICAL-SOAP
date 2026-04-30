import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from clinical_speech.config import BenchmarkConfig, GenerationConfig, ModelConfig
from clinical_speech.models.note_generator import NoteGenerator


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, prompt, return_tensors: str = "pt", padding: bool = False, truncation: bool = False):
        del return_tensors, truncation
        if isinstance(prompt, list):
            rows = []
            for index, _item in enumerate(prompt, start=1):
                rows.append([10 + index, 20 + index])
            input_ids = torch.tensor(rows, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            if padding:
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
        return {
            "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "decoded-" + "-".join(str(int(token)) for token in token_ids)


class _FlakyGenerateModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.calls = 0

    def generate(self, **kwargs):
        del kwargs
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("CUDA out of memory")
        return torch.tensor([[11, 12, 21, 22]], dtype=torch.long)


class _BatchGenerateModel:
    def __init__(self):
        self.device = torch.device("cpu")

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch_size = int(input_ids.shape[0])
        suffixes = []
        for row_index in range(batch_size):
            suffixes.append(torch.tensor([31 + row_index, 41 + row_index], dtype=torch.long))
        return torch.stack(
            [
                torch.cat([input_ids[row_index], suffixes[row_index]])
                for row_index in range(batch_size)
            ],
            dim=0,
        )


class NoteGeneratorRetryTest(unittest.TestCase):
    def test_retries_once_after_oom(self) -> None:
        fake_bundle = SimpleNamespace(tokenizer=_FakeTokenizer(), model=_FlakyGenerateModel())
        model_cfg = ModelConfig(asr_model_id="asr", llm_model_id="llm", device="cpu")
        generation_cfg = GenerationConfig(max_new_tokens=4, do_sample=False)
        benchmark_cfg = BenchmarkConfig(enabled=False)

        with mock.patch("clinical_speech.models.note_generator.load_causal_lm_bundle", return_value=fake_bundle):
            with mock.patch("clinical_speech.models.note_generator.torch.cuda.is_available", return_value=False):
                generator = NoteGenerator(model_cfg, generation_cfg)
                result = generator.generate("prompt", benchmark_cfg)

        self.assertEqual(fake_bundle.model.calls, 2)
        self.assertEqual(result.prompt_tokens, 2)
        self.assertEqual(result.completion_tokens, 2)
        self.assertEqual(result.text, "decoded-21-22")

    def test_generate_batch_uses_true_batched_generate(self) -> None:
        fake_bundle = SimpleNamespace(tokenizer=_FakeTokenizer(), model=_BatchGenerateModel())
        model_cfg = ModelConfig(asr_model_id="asr", llm_model_id="llm", device="cpu")
        generation_cfg = GenerationConfig(max_new_tokens=4, do_sample=False)
        benchmark_cfg = BenchmarkConfig(enabled=False)

        with mock.patch("clinical_speech.models.note_generator.load_causal_lm_bundle", return_value=fake_bundle):
            with mock.patch("clinical_speech.models.note_generator.torch.cuda.is_available", return_value=False):
                generator = NoteGenerator(model_cfg, generation_cfg)
                result = generator.generate_batch(["prompt-a", "prompt-b", "prompt-c"], benchmark_cfg)

        self.assertEqual(result.active_requests, 3)
        self.assertEqual(result.prompt_tokens, [2, 2, 2])
        self.assertEqual(result.completion_tokens, [2, 2, 2])
        self.assertEqual(result.total_completion_tokens, 6)
        self.assertEqual(result.texts[0], "decoded-31-41")
        self.assertEqual(result.texts[1], "decoded-32-42")
        self.assertEqual(result.texts[2], "decoded-33-43")
        self.assertGreater(result.throughput_tok_per_sec or 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
