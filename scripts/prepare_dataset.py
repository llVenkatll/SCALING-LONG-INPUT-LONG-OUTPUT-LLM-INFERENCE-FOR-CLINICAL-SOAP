import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

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
from clinical_speech.utils.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize a real dataset into the clinical_speech JSONL schema.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--transcript-field", default="transcript")
    parser.add_argument("--reference-note-field", default="reference_note")
    parser.add_argument("--audio-path-field", default="audio_path")
    parser.add_argument("--metadata-field", default="metadata")
    parser.add_argument("--dataset-name", default="external_dataset")
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _parse_metadata(value: Any, dataset_name: str) -> dict[str, Any]:
    if value in (None, "", {}):
        return {"source": dataset_name}
    if isinstance(value, dict):
        metadata = dict(value)
    else:
        try:
            metadata = json.loads(value)
            if not isinstance(metadata, dict):
                metadata = {"raw_metadata": metadata}
        except (json.JSONDecodeError, TypeError):
            metadata = {"raw_metadata": value}
    metadata.setdefault("source", dataset_name)
    return metadata


def normalize_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    normalized = []
    for index, row in enumerate(rows):
        sample_id = row.get(args.id_field) or f"{args.dataset_name}-{index}"
        transcript = row.get(args.transcript_field)
        reference_note = row.get(args.reference_note_field)
        audio_path = row.get(args.audio_path_field)
        metadata = _parse_metadata(row.get(args.metadata_field), args.dataset_name)

        normalized.append(
            {
                "id": str(sample_id),
                "transcript": transcript,
                "reference_note": reference_note,
                "audio_path": audio_path or None,
                "metadata": metadata,
            }
        )
    return normalized


def main() -> None:
    args = parse_args()
    runtime_paths = bootstrap_storage_env(args.runtime_root, override=args.runtime_root is not None)
    input_path = resolve_existing_input_path(
        args.input,
        candidate_bases=[
            runtime_paths.datasets,
            PACKAGE_PROJECT_ROOT,
        ],
    )
    output_path = resolve_managed_path(
        args.output,
        base_dir=runtime_paths.datasets,
        strip_prefixes=("datasets", "data"),
    )
    if not args.skip_preflight:
        validate_storage_layout(
            [
                ("dataset_input", input_path),
                ("dataset_output", output_path),
            ],
            runtime_root=runtime_paths.root,
        )

    rows = _load_rows(input_path)
    normalized = normalize_rows(rows, args)
    write_jsonl(output_path, normalized)
    print(f"Normalized {len(normalized)} records to {output_path}")


if __name__ == "__main__":
    main()
