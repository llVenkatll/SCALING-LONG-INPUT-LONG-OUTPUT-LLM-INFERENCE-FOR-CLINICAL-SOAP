# Evaluation

## Supported Quality Metrics

Current quality metrics are computed in `src/clinical_speech/evaluation/metrics.py`.

Supported today:

- ROUGE
- BERTScore F1 mean
- WER, but only when every prediction contains both:
  - `reference_transcript`
  - `generated_transcript`

Unsupported metrics are reported explicitly in the saved quality report instead of being implied.

## Evaluation Honesty Rules

- If reference notes are missing, note-generation quality metrics are marked unsupported.
- If transcript pairs required for WER are missing, WER is marked unsupported.
- If the evaluation set is tiny, a warning is emitted.
- If prediction metadata marks the dataset as `smoke_test`, a warning is emitted.

## Smoke-Test Warning

The fixture datasets in `data/fixtures/` are only for infrastructure checks.

Do not use:

- `configs/smoke_naive.yaml`
- `configs/smoke_chunked.yaml`

for paper tables, ablations, or scientific claims.

## Prediction Schema Notes

Prediction records may now include:

- `reference_transcript`
- `generated_transcript`

These are used to support honest WER computation for audio-to-note runs when the source dataset contains gold transcripts.

## Preparing A Real Dataset

Use `scripts/prepare_dataset.py` to normalize an external CSV or JSONL source into the expected schema:

```json
{"id":"sample-1","transcript":"...","reference_note":"...","audio_path":null,"metadata":{"source":"dataset-name"}}
```

Example:

```bash
python3 scripts/prepare_dataset.py \
  --input /data/project_runtime/datasets/raw/source.csv \
  --output /data/project_runtime/datasets/meddialog/test.jsonl \
  --id-field visit_id \
  --transcript-field conversation \
  --reference-note-field soap_note \
  --audio-path-field wav_path \
  --metadata-field metadata_json \
  --dataset-name meddialog
```

## Recommended Reporting Practice

- Report the dataset name and number of evaluated samples.
- State whether the run was transcript-to-note or audio-to-note.
- Report runtime and quality metrics separately.
- State explicitly when a run used smoke fixtures, benchmark-only mode, or approximate prefill timing.
