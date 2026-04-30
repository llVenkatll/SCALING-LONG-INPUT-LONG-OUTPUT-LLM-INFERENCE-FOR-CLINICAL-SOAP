import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clinical_speech.storage import (
    PROJECT_ROOT as PACKAGE_PROJECT_ROOT,
    bootstrap_storage_env,
    resolve_existing_input_path,
    resolve_managed_path,
    validate_storage_layout,
)
from clinical_speech.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from clinical_speech.evaluation.metrics import evaluate_predictions_file
    except ModuleNotFoundError as exc:
        missing = exc.name or "project dependencies"
        raise SystemExit(
            f"Evaluation requires {missing}. Install the project dependencies, then rerun this command."
        ) from exc

    runtime_paths = bootstrap_storage_env(args.runtime_root, override=args.runtime_root is not None)
    predictions_path = resolve_existing_input_path(
        args.predictions,
        candidate_bases=[
            runtime_paths.outputs,
            PACKAGE_PROJECT_ROOT,
        ],
        strip_prefixes=("outputs",),
    )
    output_path = resolve_managed_path(
        args.output,
        base_dir=runtime_paths.outputs,
        strip_prefixes=("outputs",),
    )
    if not args.skip_preflight:
        validate_storage_layout(
            [
                ("predictions_path", predictions_path),
                ("metrics_output", output_path),
            ],
            runtime_root=runtime_paths.root,
        )
    metrics = evaluate_predictions_file(predictions_path)
    write_json(output_path, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
