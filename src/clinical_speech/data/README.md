# Dataset Format

The experiment runner expects JSONL records with the following structure:

```json
{"id":"sample-1","transcript":"...","reference_note":"...","audio_path":"data/raw/audio/sample.wav","metadata":{"source":"medsynth"}}
```

## Required for transcript-to-note

- `id`
- `transcript`
- `reference_note`

## Required for audio-to-note

- `id`
- `audio_path`
- `reference_note`

Recommended:

- `transcript` as gold transcript for WER evaluation

## Suggested Splits

For the hardened EC2 workflow, place them under `/data/project_runtime/datasets`:

- `/data/project_runtime/datasets/medsynth/train.jsonl`
- `/data/project_runtime/datasets/medsynth/valid.jsonl`
- `/data/project_runtime/datasets/medsynth/test.jsonl`
- `/data/project_runtime/datasets/primock57/test.jsonl`

## Smoke Fixtures

Tiny repo fixtures for infrastructure-only testing live in:

- `data/fixtures/smoke_notes.jsonl`
- `data/fixtures/smoke_notes_long.jsonl`

These are not scientific datasets.

## Normalizing Real Data

Use `scripts/prepare_dataset.py` to convert an external CSV or JSONL source into the expected schema while keeping the output under `/data/project_runtime/datasets`.
