from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transformers import AutoTokenizer

from clinical_speech.pipeline.prompts import build_note_prompt
from clinical_speech.utils.io import read_jsonl, write_jsonl


BUCKET_TARGETS = [512, 1024, 2048, 4096, 8192]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/data/project_runtime/datasets/medsynth/test.jsonl"),
    )
    parser.add_argument(
        "--long-source",
        type=Path,
        default=Path("/data/project_runtime/datasets/medsynth/long_context_4000.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/project_runtime/datasets/medsynth/context_buckets"),
    )
    parser.add_argument(
        "--tokenizer-model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
    )
    parser.add_argument("--samples-per-bucket", type=int, default=8)
    return parser.parse_args()


def _prompt_len(tokenizer, transcript: str) -> int:
    prompt = build_note_prompt(transcript)
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def _truncate_transcript_to_target(tokenizer, transcript: str, target_tokens: int) -> str:
    words = transcript.split()
    if not words:
        return transcript
    lo = 1
    hi = len(words)
    best_words = words[:hi]
    best_gap = abs(_prompt_len(tokenizer, " ".join(best_words)) - target_tokens)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = words[:mid]
        candidate_text = " ".join(candidate)
        candidate_len = _prompt_len(tokenizer, candidate_text)
        gap = abs(candidate_len - target_tokens)
        if gap < best_gap:
            best_gap = gap
            best_words = candidate
        if candidate_len < target_tokens:
            lo = mid + 1
        else:
            hi = mid - 1
    return " ".join(best_words)


def _build_small_bucket_rows(tokenizer, rows: list[dict], target_tokens: int, samples: int) -> list[dict]:
    records: list[dict] = []
    def _priority(row: dict) -> tuple[int, int]:
        prompt_tokens = _prompt_len(tokenizer, row.get("transcript", ""))
        return (0 if prompt_tokens >= target_tokens else 1, abs(prompt_tokens - target_tokens))

    ordered = sorted(rows, key=_priority)
    for index, row in enumerate(ordered):
        transcript = row.get("transcript", "")
        adjusted = _truncate_transcript_to_target(tokenizer, transcript, target_tokens)
        records.append(
            {
                "id": f"bucket-{target_tokens}-{index}",
                "transcript": adjusted,
                "reference_note": row.get("reference_note", ""),
                "audio_path": None,
                "metadata": {
                    **(row.get("metadata") or {}),
                    "bucket_target_tokens": target_tokens,
                    "source_mode": "truncate_single",
                    "source_id": row.get("id"),
                    "prompt_tokens": _prompt_len(tokenizer, adjusted),
                },
            }
        )
        if len(records) >= samples:
            break
    return records


def _merge_until_target(tokenizer, rows: list[dict], start_index: int, target_tokens: int) -> tuple[dict, int]:
    merged_rows: list[dict] = []
    index = start_index
    prompt_tokens = 0
    while index < len(rows) and prompt_tokens < target_tokens:
        merged_rows.append(rows[index])
        candidate_text = "\n\n".join(item.get("transcript", "") for item in merged_rows)
        prompt_tokens = _prompt_len(tokenizer, candidate_text)
        index += 1

    if not merged_rows:
        raise ValueError("Could not build merged context bucket row")

    candidate_text = "\n\n".join(item.get("transcript", "") for item in merged_rows)
    if prompt_tokens > target_tokens and merged_rows:
        last = merged_rows[-1]
        prefix = "\n\n".join(item.get("transcript", "") for item in merged_rows[:-1])
        remaining_budget = target_tokens - _prompt_len(tokenizer, prefix) if prefix else target_tokens
        if remaining_budget > 0:
            adjusted_last = _truncate_transcript_to_target(tokenizer, last.get("transcript", ""), remaining_budget)
            pieces = [item.get("transcript", "") for item in merged_rows[:-1]]
            pieces.append(adjusted_last)
            candidate_text = "\n\n".join(piece for piece in pieces if piece)
            prompt_tokens = _prompt_len(tokenizer, candidate_text)

    merged = {
        "transcript": candidate_text,
        "reference_note": "\n\n".join(item.get("reference_note", "") for item in merged_rows),
        "metadata": {
            "bucket_target_tokens": target_tokens,
            "source_mode": "merge_sequence",
            "source_ids": [item.get("id") for item in merged_rows],
            "prompt_tokens": prompt_tokens,
        },
    }
    return merged, index


def _build_merged_bucket_rows(tokenizer, rows: list[dict], target_tokens: int, samples: int) -> list[dict]:
    records: list[dict] = []
    cursor = 0
    while cursor < len(rows) and len(records) < samples:
        merged, next_cursor = _merge_until_target(tokenizer, rows, cursor, target_tokens)
        records.append(
            {
                "id": f"bucket-{target_tokens}-{len(records)}",
                "transcript": merged["transcript"],
                "reference_note": merged["reference_note"],
                "audio_path": None,
                "metadata": merged["metadata"],
            }
        )
        cursor = next_cursor
    return records


def _build_long_bucket_rows(tokenizer, rows: list[dict], target_tokens: int, samples: int) -> list[dict]:
    scored = []
    for row in rows:
        prompt_tokens = _prompt_len(tokenizer, row.get("transcript", ""))
        scored.append((abs(prompt_tokens - target_tokens), prompt_tokens, row))
    scored.sort(key=lambda item: item[0])
    records: list[dict] = []
    for index, (_gap, prompt_tokens, row) in enumerate(scored[:samples]):
        records.append(
            {
                "id": f"bucket-{target_tokens}-{index}",
                "transcript": row.get("transcript", ""),
                "reference_note": row.get("reference_note", ""),
                "audio_path": None,
                "metadata": {
                    **(row.get("metadata") or {}),
                    "bucket_target_tokens": target_tokens,
                    "source_mode": "select_long_context",
                    "source_id": row.get("id"),
                    "prompt_tokens": prompt_tokens,
                },
            }
        )
    return records


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, cache_dir="/data/project_runtime/cache/hf")
    base_rows = read_jsonl(args.source)
    long_rows = read_jsonl(args.long_source) if args.long_source.exists() else []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, int | str]] = []

    for target in BUCKET_TARGETS:
        if target <= 1024:
            rows = _build_small_bucket_rows(tokenizer, base_rows, target, args.samples_per_bucket)
        elif target < 8192:
            rows = _build_merged_bucket_rows(tokenizer, base_rows, target, args.samples_per_bucket)
        else:
            source_rows = long_rows if long_rows else _build_merged_bucket_rows(tokenizer, base_rows, target, args.samples_per_bucket)
            rows = _build_long_bucket_rows(tokenizer, source_rows, target, args.samples_per_bucket) if long_rows else source_rows

        output_path = args.output_dir / f"bucket_{target}.jsonl"
        write_jsonl(output_path, rows)
        prompt_lengths = [row["metadata"]["prompt_tokens"] for row in rows]
        manifest.append(
            {
                "bucket": target,
                "count": len(rows),
                "min_prompt_tokens": min(prompt_lengths) if prompt_lengths else 0,
                "max_prompt_tokens": max(prompt_lengths) if prompt_lengths else 0,
                "path": str(output_path),
            }
        )
        print(f"Wrote {len(rows)} rows to {output_path}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(__import__("json").dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
