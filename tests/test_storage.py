import unittest
from pathlib import Path

from clinical_speech.storage import build_runtime_paths, resolve_managed_path, validate_storage_layout


class StoragePathTests(unittest.TestCase):
    def test_resolve_managed_output_path_strips_outputs_prefix(self) -> None:
        runtime_paths = build_runtime_paths("/data/project_runtime/test_storage_runtime")
        resolved = resolve_managed_path(
            Path("outputs/example_run"),
            base_dir=runtime_paths.outputs,
            strip_prefixes=("outputs",),
        )
        self.assertEqual(resolved, runtime_paths.outputs / "example_run")

    def test_runtime_paths_live_under_data_mount(self) -> None:
        runtime_paths = build_runtime_paths("/data/project_runtime/test_storage_runtime")
        self.assertTrue(str(runtime_paths.outputs).startswith("/data/"))
        self.assertTrue(str(runtime_paths.tmp).startswith("/data/"))

    def test_repo_fixture_dataset_is_warning_not_error(self) -> None:
        runtime_paths = build_runtime_paths("/data/project_runtime/test_storage_runtime")
        warnings = validate_storage_layout(
            [("dataset_path", Path("/data/project/data/fixtures/smoke_notes.jsonl"))],
            runtime_root=runtime_paths.root,
        )
        self.assertTrue(any("repository checkout" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
