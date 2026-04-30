import unittest

from clinical_speech.evaluation.metrics import aggregate_runtime_metrics, evaluation_warnings


class RuntimeMetricTests(unittest.TestCase):
    def test_runtime_aggregation_includes_std_and_count(self) -> None:
        predictions = [
            {"runtime": {"generation_latency_sec": 2.0, "ttft_sec": 0.5, "prompt_tokens": 10}},
            {"runtime": {"generation_latency_sec": 4.0, "ttft_sec": 1.5, "prompt_tokens": 14}},
        ]
        runtime = aggregate_runtime_metrics(predictions)
        self.assertEqual(runtime["generation_latency_sec"]["count"], 2)
        self.assertIn("std", runtime["generation_latency_sec"])
        self.assertEqual(runtime["prompt_tokens"]["max"], 14.0)

    def test_evaluation_warnings_flag_tiny_and_smoke_datasets(self) -> None:
        predictions = [
            {"metadata": {"smoke_test": True}},
            {"metadata": {"smoke_test": True}},
        ]
        warnings = evaluation_warnings(predictions)
        self.assertEqual(len(warnings), 2)
        self.assertIn("smoke-test", warnings[0].lower() + warnings[1].lower())


if __name__ == "__main__":
    unittest.main()
