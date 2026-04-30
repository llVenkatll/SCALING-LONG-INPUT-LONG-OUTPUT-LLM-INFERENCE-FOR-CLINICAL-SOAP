import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from clinical_speech.config import load_config


class ConfigResolutionTests(unittest.TestCase):
    def test_inherited_config_resolves_runtime_paths(self) -> None:
        runtime_root = Path("/data/project_runtime/test_config_runtime")
        cfg = load_config("configs/naive_baseline_benchmark.yaml", runtime_root=runtime_root)

        self.assertEqual(cfg.experiment.name, "naive_baseline_benchmark")
        self.assertEqual(cfg.benchmark.warmup_runs, 1)
        self.assertEqual(cfg.benchmark.repeat_runs, 3)
        self.assertTrue(cfg.benchmark.benchmark_only)
        self.assertTrue(str(cfg.experiment.output_dir).startswith(str(runtime_root / "outputs")))
        self.assertTrue(str(cfg.experiment.benchmark_dir).startswith(str(runtime_root / "benchmarks")))

    def test_smoke_fixture_path_stays_repo_relative(self) -> None:
        cfg = load_config("configs/smoke_naive.yaml", runtime_root="/data/project_runtime/test_smoke_config")
        self.assertTrue(str(cfg.dataset.path).endswith("data/fixtures/smoke_notes.jsonl"))

    def test_systems_benchmark_backends_can_be_overridden_by_config(self) -> None:
        cfg = load_config("configs/systems_benchmark_batch4_pilot.yaml", runtime_root="/data/project_runtime/test_systems_cfg")
        self.assertTrue(cfg.runtime.triton_paged_kv_enabled)
        self.assertEqual(
            cfg.systems_benchmark.backends,
            [
                "hf_sequential",
                "mistral_paged_single",
                "mistral_paged_static_batch",
                "mistral_paged_static_batch_triton",
            ],
        )

    def test_long_context_pilot_resolves_triton_backend_list(self) -> None:
        cfg = load_config(
            "configs/systems_benchmark_long_context_pilot.yaml",
            runtime_root="/data/project_runtime/test_long_context_systems_cfg",
        )
        self.assertTrue(cfg.runtime.triton_paged_kv_enabled)
        self.assertEqual(
            cfg.systems_benchmark.backends,
            [
                "hf_sequential",
                "mistral_paged_static_batch",
                "mistral_paged_static_batch_triton",
            ],
        )

    def test_legacy_enabled_backends_key_is_normalized(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "legacy_systems.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "extends: /data/project/configs/systems_benchmark_pilot.yaml",
                        "runtime:",
                        "  triton_paged_kv_enabled: true",
                        "systems_benchmark:",
                        "  enabled_backends:",
                        "    - hf_sequential",
                        "    - mistral_paged_static_batch",
                        "    - mistral_paged_static_batch_triton",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_config(config_path, runtime_root="/data/project_runtime/test_legacy_enabled_backends")
        self.assertEqual(
            cfg.systems_benchmark.backends,
            [
                "hf_sequential",
                "mistral_paged_static_batch",
                "mistral_paged_static_batch_triton",
            ],
        )

    def test_serving_stack_comparison_config_parses_backend_specs(self) -> None:
        cfg = load_config(
            "configs/systems_benchmark_serving_stack_comparison.yaml",
            runtime_root="/data/project_runtime/test_serving_stack_comparison_cfg",
        )
        self.assertEqual(
            cfg.systems_benchmark.backends,
            [
                "hf_sequential",
                "mistral_paged_static_batch",
                "mistral_paged_static_batch_triton",
                "hf_sequential_llama_local",
                "together_hosted_llama",
            ],
        )
        self.assertEqual(
            cfg.systems_benchmark.backend_specs["hf_sequential_llama_local"].llm_model_id,
            "meta-llama/Meta-Llama-3-8B-Instruct",
        )
        self.assertEqual(
            cfg.systems_benchmark.backend_specs["together_hosted_llama"].provider,
            "together_hosted",
        )
        self.assertEqual(
            cfg.systems_benchmark.backend_specs["together_hosted_llama"].api_model_id,
            "venkat011003_84d6/togethercomputer/meta-llama-3.1-8B-Instruct-AWQ-INT4-69f5dd64",
        )

    def test_long_context_compact_prompt_mode_config_parses(self) -> None:
        cfg = load_config(
            "configs/systems_benchmark_long_context_compact_prompt.yaml",
            runtime_root="/data/project_runtime/test_long_context_compact_prompt_cfg",
        )
        self.assertEqual(cfg.systems_benchmark.prompt_mode, "compact_soap")
        self.assertEqual(
            cfg.systems_benchmark.backends,
            [
                "hf_sequential",
                "mistral_paged_static_batch",
                "mistral_paged_static_batch_triton",
            ],
        )

    def test_prefill_two_stage_config_parses(self) -> None:
        cfg = load_config(
            "configs/prefill_two_stage_long_context.yaml",
            runtime_root="/data/project_runtime/test_prefill_two_stage_cfg",
        )
        self.assertEqual(cfg.systems_benchmark.prompt_mode, "two_stage_facts")
        self.assertEqual(cfg.systems_benchmark.stage1_max_new_tokens, 256)
        self.assertEqual(cfg.systems_benchmark.stage2_max_new_tokens, 16)
        self.assertEqual(cfg.systems_benchmark.backends, ["hf_sequential"])


if __name__ == "__main__":
    unittest.main()
