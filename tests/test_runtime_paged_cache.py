from types import SimpleNamespace
from unittest import mock
import unittest

import torch

from clinical_speech.config import GenerationConfig, ModelConfig, RuntimeConfig
from clinical_speech.runtime.block_manager import KVCacheBlockManager, KVCacheLayout
from clinical_speech.runtime.engine import ManualBatchEngine
from clinical_speech.runtime.kv_cache import PagedKVCache


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __call__(self, text: str, add_special_tokens: bool = True, return_attention_mask: bool = False, return_tensors=None):
        del add_special_tokens, return_attention_mask, return_tensors
        ids = [int(token) for token in text.split()] if text else [1]
        return {"input_ids": ids}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        values = []
        for token_id in token_ids:
            value = int(token_id)
            if skip_special_tokens and value == self.eos_token_id:
                continue
            values.append(str(value))
        return " ".join(values)


class FakeMistralConfig:
    model_type = "mistral"
    num_hidden_layers = 1
    num_attention_heads = 1
    num_key_value_heads = 1
    hidden_size = 2


class FakeMistralModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = FakeMistralConfig()
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        return_dict: bool = True,
    ):
        del attention_mask, position_ids, use_cache, return_dict
        batch_size, seq_len = input_ids.shape
        kv_states = input_ids.to(torch.float32).view(batch_size, 1, seq_len, 1).repeat(1, 1, 1, 2)
        if past_key_values is not None:
            past_key_values.update(kv_states, kv_states + 100.0, 0)

        vocab_size = 16
        logits = torch.zeros((batch_size, seq_len, vocab_size), dtype=torch.float32)
        next_tokens = (input_ids[:, -1] + 1) % vocab_size
        for row_index, token in enumerate(next_tokens):
            logits[row_index, -1, int(token.item())] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


class PagedKVCacheRuntimeTest(unittest.TestCase):
    def test_paged_cache_decode_matches_expected_dense_gather(self) -> None:
        layout = KVCacheLayout(num_layers=1, num_key_value_heads=1, head_dim=2, bytes_per_element=4)
        manager = KVCacheBlockManager(
            layout=layout,
            block_size_tokens=2,
            max_cache_budget_bytes=layout.bytes_per_token * 2 * 4,
        )
        cache = PagedKVCache(layout=layout, block_manager=manager, block_size_tokens=2)

        prefill_keys = torch.tensor(
            [
                [[[-1.0, -1.0], [10.0, 10.0], [11.0, 11.0], [12.0, 12.0]]],
                [[[-2.0, -2.0], [-2.0, -2.0], [20.0, 20.0], [21.0, 21.0]]],
            ]
        )
        cache.begin_forward(request_ids=["req_a", "req_b"], token_lengths=[3, 2], attention_mask_length=4)
        dense_prefill_keys, _dense_prefill_values = cache.update(prefill_keys, prefill_keys + 100.0, 0)
        cache.clear_batch_view()
        self.assertTrue(torch.equal(dense_prefill_keys, prefill_keys))

        decode_keys = torch.tensor([[[[13.0, 13.0]]], [[[22.0, 22.0]]]])
        cache.begin_forward(request_ids=["req_a", "req_b"], token_lengths=[1, 1], attention_mask_length=5)
        dense_decode_keys, _dense_decode_values = cache.update(decode_keys, decode_keys + 100.0, 0)
        cache.clear_batch_view()

        expected = torch.tensor(
            [
                [[[0.0, 0.0], [10.0, 10.0], [11.0, 11.0], [12.0, 12.0], [13.0, 13.0]]],
                [[[0.0, 0.0], [0.0, 0.0], [20.0, 20.0], [21.0, 21.0], [22.0, 22.0]]],
            ]
        )
        self.assertTrue(torch.equal(dense_decode_keys, expected))

    def test_manual_engine_smoke_uses_paged_runtime_and_metrics_schema(self) -> None:
        fake_bundle = SimpleNamespace(tokenizer=FakeTokenizer(), model=FakeMistralModel())
        with mock.patch("clinical_speech.runtime.engine.load_causal_lm_bundle", return_value=fake_bundle):
            engine = ManualBatchEngine(
                ModelConfig(asr_model_id="fake/asr", llm_model_id="fake/mistral", device="cpu", dtype="float32"),
                GenerationConfig(max_new_tokens=2, do_sample=False),
                RuntimeConfig(
                    backend="mistral_paged",
                    scheduler_mode="static_batch",
                    block_size_tokens=2,
                    max_batch_size=2,
                    max_concurrent_requests=2,
                    max_cache_budget_gb=0.001,
                    triton_enabled=False,
                    triton_paged_kv_enabled=False,
                    pad_to_multiple_of=1,
                ),
            )
            result = engine.run_batch(["req_a", "req_b"], ["1 2 3", "4 5"])

        self.assertEqual(result.scheduler_steps, 1)
        self.assertEqual([request.text for request in result.requests], ["4 5", "6 7"])
        self.assertIn("kv_allocated_bytes", result.as_dict())
        self.assertIn("scheduler_steps", result.as_dict())


if __name__ == "__main__":
    unittest.main()
