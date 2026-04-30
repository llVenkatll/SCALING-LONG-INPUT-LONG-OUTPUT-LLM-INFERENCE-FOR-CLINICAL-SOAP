from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clinical_speech.storage import bootstrap_storage_env
from clinical_speech.utils.io import write_json, write_jsonl


DATASET_REGISTRY = {
    "medsynth": {
        "hf_dataset": "Ahmad0067/MedSynth",
        "config": "default",
        "split": "train",
        "default_output_dir": "medsynth",
        "source_url": "https://huggingface.co/datasets/Ahmad0067/MedSynth",
    }
}

DATASETS_SERVER_BASE = "https://datasets-server.huggingface.co/rows"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a public reference dataset and normalize it into the clinical_speech JSONL schema."
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_REGISTRY.keys()), default="medsynth")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--runtime-root", type=Path, default=None)
    return parser.parse_args()


def _fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def _build_rows_url(dataset: str, config: str, split: str, offset: int, length: int) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    return f"{DATASETS_SERVER_BASE}?{query}"


def _fetch_all_rows(dataset: str, config: str, split: str, page_size: int, max_rows: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    total = None
    while True:
        length = page_size
        if max_rows is not None:
            remaining = max_rows - len(rows)
            if remaining <= 0:
                break
            length = min(length, remaining)
        payload = _fetch_json(_build_rows_url(dataset, config, split, offset, length))
        batch = payload.get("rows", [])
        if total is None:
            total = payload.get("num_rows_total")
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if len(batch) < length:
            break
        if total is not None and offset >= total:
            break
    return rows


def _normalize_medsynth_rows(rows: list[dict[str, Any]], dataset_name: str, hf_dataset: str, hf_split: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for entry in rows:
        row = entry["row"]
        row_idx = int(entry["row_idx"])
        transcript = str(row.get("Dialogue", "")).strip()
        reference_note = str(row.get(" Note", "")).strip()
        normalized.append(
            {
                "id": f"{dataset_name}-{row_idx:05d}",
                "transcript": transcript,
                "reference_note": reference_note,
                "audio_path": None,
                "metadata": {
                    "source": dataset_name,
                    "hf_dataset": hf_dataset,
                    "hf_split": hf_split,
                    "original_row_idx": row_idx,
                    "icd10": row.get("ICD10"),
                    "icd10_desc": row.get("ICD10_desc"),
                },
            }
        )
    return normalized


def _split_rows(rows: list[dict[str, Any]], valid_fraction: float, test_fraction: float, seed: int) -> dict[str, list[dict[str, Any]]]:
    if valid_fraction < 0 or test_fraction < 0 or valid_fraction + test_fraction >= 1:
        raise ValueError("valid_fraction and test_fraction must be non-negative and sum to less than 1")

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    valid_count = int(total * valid_fraction)
    test_count = int(total * test_fraction)
    train_count = total - valid_count - test_count
    return {
        "train": shuffled[:train_count],
        "valid": shuffled[train_count : train_count + valid_count],
        "test": shuffled[train_count + valid_count :],
    }


def main() -> None:
    args = parse_args()
    spec = DATASET_REGISTRY[args.dataset]
    runtime_paths = bootstrap_storage_env(args.runtime_root, override=args.runtime_root is not None)
    output_dir = args.output_dir or (runtime_paths.datasets / spec["default_output_dir"])
    if not output_dir.is_absolute():
        output_dir = runtime_paths.datasets / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = _fetch_all_rows(
        dataset=spec["hf_dataset"],
        config=spec["config"],
        split=spec["split"],
        page_size=args.page_size,
        max_rows=args.max_rows,
    )
    if args.dataset == "medsynth":
        normalized = _normalize_medsynth_rows(
            raw_rows,
            dataset_name=args.dataset,
            hf_dataset=spec["hf_dataset"],
            hf_split=spec["split"],
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    splits = _split_rows(normalized, valid_fraction=args.valid_fraction, test_fraction=args.test_fraction, seed=args.seed)

    for split_name, split_rows in splits.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", split_rows)

    manifest = {
        "dataset": args.dataset,
        "source_url": spec["source_url"],
        "hf_dataset": spec["hf_dataset"],
        "hf_config": spec["config"],
        "hf_split": spec["split"],
        "seed": args.seed,
        "valid_fraction": args.valid_fraction,
        "test_fraction": args.test_fraction,
        "counts": {name: len(items) for name, items in splits.items()},
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "manifest.json", manifest)

    print(f"Fetched {len(normalized)} rows from {spec['hf_dataset']}")
    for split_name, split_rows in splits.items():
        print(f"Wrote {len(split_rows)} rows to {output_dir / f'{split_name}.jsonl'}")
    print(f"Wrote manifest to {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
