import argparse
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
from clinical_speech.utils.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-words", required=True, type=int)
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def merge_records(records: list[dict[str, Any]], target_words: int) -> list[dict[str, Any]]:
    merged = []
    bucket = []
    word_count = 0
    index = 0
    for record in records:
        transcript = record.get("transcript", "")
        words = len(transcript.split())
        bucket.append(record)
        word_count += words
        if word_count >= target_words:
            merged.append(
                {
                    "id": f"merged-{index}",
                    "transcript": "\n\n".join(item.get("transcript", "") for item in bucket),
                    "reference_note": "\n\n".join(item.get("reference_note", "") for item in bucket),
                    "audio_path": None,
                    "metadata": {"merged_ids": [item.get("id") for item in bucket]},
                }
            )
            index += 1
            bucket = []
            word_count = 0
    if bucket:
        merged.append(
            {
                "id": f"merged-{index}",
                "transcript": "\n\n".join(item.get("transcript", "") for item in bucket),
                "reference_note": "\n\n".join(item.get("reference_note", "") for item in bucket),
                "audio_path": None,
                "metadata": {"merged_ids": [item.get("id") for item in bucket]},
            }
        )
    return merged


def main() -> None:
    args = parse_args()
    runtime_paths = bootstrap_storage_env(args.runtime_root, override=args.runtime_root is not None)
    input_path = resolve_existing_input_path(
        args.input,
        candidate_bases=[
            runtime_paths.datasets,
            PACKAGE_PROJECT_ROOT,
        ],
        strip_prefixes=("datasets",),
    )
    output_path = resolve_managed_path(
        args.output,
        base_dir=runtime_paths.datasets,
        strip_prefixes=("datasets", "data"),
    )
    if not args.skip_preflight:
        validate_storage_layout(
            [
                ("long_context_input", input_path),
                ("long_context_output", output_path),
            ],
            runtime_root=runtime_paths.root,
        )
    records = read_jsonl(input_path)
    merged = merge_records(records, target_words=args.target_words)
    write_jsonl(output_path, merged)
    print(f"Wrote {len(merged)} long-context samples to {output_path}")


if __name__ == "__main__":
    main()
