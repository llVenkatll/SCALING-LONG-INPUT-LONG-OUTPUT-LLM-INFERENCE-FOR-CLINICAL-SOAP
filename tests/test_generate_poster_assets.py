import importlib.util
import sys
import unittest
from pathlib import Path


def _load_module(script_name: str, module_name: str):
    script_path = Path("/data/project/scripts") / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


poster_assets = _load_module("generate_poster_assets.py", "generate_poster_assets")


class PosterAssetSelectionTests(unittest.TestCase):
    def test_resolve_main_comparison_prefers_full_three_backend_batch(self) -> None:
        rows_by_key = {
            ("hf_sequential", 1): {},
            ("mistral_paged_static_batch", 1): {},
            ("mistral_paged_static_batch_triton", 1): {},
            ("hf_sequential", 2): {},
            ("mistral_paged_static_batch", 2): {},
            ("mistral_paged_static_batch_triton", 2): {},
        }
        batch_size, backends = poster_assets._resolve_main_comparison(rows_by_key)
        self.assertEqual(batch_size, 2)
        self.assertEqual(
            backends,
            [
                "hf_sequential",
                "mistral_paged_static_batch",
                "mistral_paged_static_batch_triton",
            ],
        )

    def test_resolve_main_comparison_honors_preferred_batch_when_available(self) -> None:
        rows_by_key = {
            ("hf_sequential", 2): {},
            ("mistral_paged_static_batch", 2): {},
            ("mistral_paged_static_batch_triton", 2): {},
            ("hf_sequential", 4): {},
            ("mistral_paged_static_batch", 4): {},
            ("mistral_paged_static_batch_triton", 4): {},
        }
        batch_size, backends = poster_assets._resolve_main_comparison(rows_by_key, preferred_batch_size=2)
        self.assertEqual(batch_size, 2)
        self.assertEqual(len(backends), 3)

    def test_resolve_main_comparison_includes_hosted_backends_when_present(self) -> None:
        rows_by_key = {
            ("hf_sequential", 4): {"backend_label": "HF Sequential (Mistral, local)", "mean_throughput_tok_per_sec": "20.0", "mean_requests_per_sec": "0.3"},
            ("mistral_paged_static_batch", 4): {"backend_label": "Paged Static Batch (Mistral, local)", "mean_throughput_tok_per_sec": "30.0", "mean_requests_per_sec": "0.5"},
            ("mistral_paged_static_batch_triton", 4): {"backend_label": "Paged Static Batch + Triton (Mistral, ours)", "mean_throughput_tok_per_sec": "40.0", "mean_requests_per_sec": "0.6"},
            ("hf_sequential_llama_local", 4): {"backend_label": "HF Sequential (Llama, local)", "mean_throughput_tok_per_sec": "18.0", "mean_requests_per_sec": "0.28"},
            ("together_hosted_llama", 4): {"backend_label": "Together Hosted (Llama)", "mean_throughput_tok_per_sec": "22.0", "mean_requests_per_sec": "0.31"},
        }
        batch_size, backends = poster_assets._resolve_main_comparison(rows_by_key)
        self.assertEqual(batch_size, 4)
        self.assertEqual(
            backends,
            [
                "hf_sequential",
                "hf_sequential_llama_local",
                "mistral_paged_static_batch",
                "mistral_paged_static_batch_triton",
                "together_hosted_llama",
            ],
        )

    def test_resolve_main_comparison_prefers_batch_with_observable_metrics(self) -> None:
        rows_by_key = {
            ("hf_sequential", 4): {"mean_throughput_tok_per_sec": "20.0", "mean_requests_per_sec": "0.3"},
            ("mistral_paged_static_batch", 4): {"mean_throughput_tok_per_sec": "30.0", "mean_requests_per_sec": "0.5"},
            ("together_hosted_llama", 4): {"mean_throughput_tok_per_sec": "", "mean_requests_per_sec": ""},
            ("hf_sequential", 2): {"mean_throughput_tok_per_sec": "19.0", "mean_requests_per_sec": "0.28"},
            ("mistral_paged_static_batch", 2): {"mean_throughput_tok_per_sec": "26.0", "mean_requests_per_sec": "0.4"},
            ("together_hosted_llama", 2): {"mean_throughput_tok_per_sec": "21.0", "mean_requests_per_sec": "0.32"},
        }
        batch_size, backends = poster_assets._resolve_main_comparison(rows_by_key)
        self.assertEqual(batch_size, 2)
        self.assertEqual(
            backends,
            [
                "hf_sequential",
                "mistral_paged_static_batch",
                "together_hosted_llama",
            ],
        )


if __name__ == "__main__":
    unittest.main()
