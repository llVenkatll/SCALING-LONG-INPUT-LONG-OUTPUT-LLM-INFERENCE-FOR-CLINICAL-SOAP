import importlib
import unittest
from unittest import mock


demo = importlib.import_module("scripts.run_live_demo")


def _metrics(*, batch_size: int = 8, truncated: bool = False) -> dict:
    return {
        "batch_size": batch_size,
        "active_requests": batch_size,
        "total_generated_tokens": batch_size * 16,
        "ttft_sec_request_1": 0.25,
        "batch_latency_sec": 1.5,
        "throughput_tok_per_sec": 85.0,
        "peak_gpu_mem_gb": 17.5,
        "likely_truncated_at_max_new_tokens": truncated,
        "completion_tokens_request_1": 16,
        "debug_extra_metric": "keep-internal-only",
    }


class LiveDemoPreloadTest(unittest.TestCase):
    def test_preload_warm_backend_uses_real_triton_batch_path(self) -> None:
        with mock.patch.object(demo, "_run_triton_batch", return_value=("discarded note", _metrics())) as run_batch:
            result = demo._preload_and_warm_backend()

        run_batch.assert_called_once_with(
            demo.WARMUP_TRANSCRIPT,
            max_new_tokens=demo.WARMUP_MAX_NEW_TOKENS,
            batch_size=demo.WARMUP_BATCH_SIZE,
        )
        self.assertEqual(result["batch_size"], 8)

    def test_run_demo_outputs_calls_backend_for_each_submission(self) -> None:
        def fake_run(transcript: str, *, max_new_tokens: int, batch_size: int):
            del max_new_tokens
            return f"live note for {transcript}", _metrics(batch_size=batch_size)

        with mock.patch.object(demo, "_run_triton_batch", side_effect=fake_run) as run_batch:
            first = demo._run_demo_outputs("first transcript", 8, 384)
            second = demo._run_demo_outputs("second transcript", 8, 384)

        self.assertEqual(run_batch.call_count, 2)
        self.assertEqual(first[2], "live note for first transcript")
        self.assertEqual(second[2], "live note for second transcript")
        self.assertNotEqual(first[2], second[2])

    def test_compact_metrics_show_only_judge_facing_fields(self) -> None:
        rendered = demo._format_metrics(_metrics(), compact=True)

        for key in demo.UI_METRIC_KEYS:
            self.assertIn(f"`{key}`", rendered)
        self.assertNotIn("debug_extra_metric", rendered)

    def test_live_demo_defaults_are_stage_friendly(self) -> None:
        self.assertEqual(demo.DEFAULT_BATCH_SIZE, 8)
        self.assertEqual(demo.DEFAULT_MAX_NEW_TOKENS, 384)
        self.assertEqual(demo.WARMUP_BATCH_SIZE, 8)
        self.assertEqual(demo.WARMUP_MAX_NEW_TOKENS, 16)
        self.assertEqual(
            demo.REAL_INFERENCE_LABEL,
            "Real inference: model is preloaded/warmed, outputs are generated live for the submitted transcript.",
        )


if __name__ == "__main__":
    unittest.main()
