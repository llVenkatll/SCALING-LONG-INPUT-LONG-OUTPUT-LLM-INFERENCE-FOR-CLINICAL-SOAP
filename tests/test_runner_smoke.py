import json
import sys
import unittest
from pathlib import Path
from unittest import mock

from clinical_speech.benchmarking import aggregate_numeric_records
from clinical_speech.models.note_generator import GenerationResult, GenerationRunMetrics
from clinical_speech.pipeline import runner


class FakeNoteGenerator:
    def __init__(self, *_args, **_kwargs):
        self.call_count = 0

    def generate(self, prompt: str, benchmark_config) -> GenerationResult:
        self.call_count += 1
        run = GenerationRunMetrics(
            tokenization_sec=0.001,
            device_transfer_sec=0.002,
            prefill_latency_sec=0.01,
            ttft_sec=0.02,
            decode_latency_sec=0.03,
            latency_sec=0.05,
            total_inference_sec=0.053,
            peak_gpu_mem_gb=1.25,
            prompt_tokens=len(prompt.split()),
            completion_tokens=12,
            total_tokens=len(prompt.split()) + 12,
        )
        return GenerationResult(
            text=f"fake note {self.call_count}",
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
        )


class RunnerSmokeTests(unittest.TestCase):
    def test_runner_writes_storage_safe_smoke_outputs(self) -> None:
        runtime_root = Path("/data/project_runtime/test_runner_smoke")
        argv = [
            "run_experiment.py",
            "--config",
            "configs/smoke_naive.yaml",
            "--runtime-root",
            str(runtime_root),
            "--benchmark-only",
            "--skip-preflight",
        ]

        with mock.patch.object(runner, "NoteGenerator", FakeNoteGenerator):
            with mock.patch.object(sys, "argv", argv):
                runner.main()

        metrics_path = runtime_root / "outputs" / "smoke_naive" / "metrics.json"
        benchmark_csv = runtime_root / "benchmarks" / "smoke_naive" / "sample_runtime_rows.csv"
        predictions_path = runtime_root / "outputs" / "smoke_naive" / "predictions.jsonl"

        self.assertTrue(metrics_path.exists())
        self.assertTrue(benchmark_csv.exists())
        self.assertTrue(predictions_path.exists())

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertTrue(metrics["benchmark"]["benchmark_only"])
        self.assertIn("skipped", metrics["quality"]["warnings"][0].lower())


if __name__ == "__main__":
    unittest.main()
