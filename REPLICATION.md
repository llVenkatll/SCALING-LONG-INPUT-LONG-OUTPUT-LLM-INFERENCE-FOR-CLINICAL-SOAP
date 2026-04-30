# Public Replication Guide

This guide is the teammate-ready path for reproducing the public, GitHub-shareable version of the project.

## What This Covers

- open transcript-to-note runs using MedSynth
- open audio/transcript/note runs using PriMock57
- local baselines, chunked runs, and systems benchmarks
- exact dataset prep commands and required environment variables

## What This Does Not Claim

- It does not claim that every archived poster/report artifact in this repo is regenerated bit-for-bit by these public datasets.
- It does not treat the public MedDialog corpus as a drop-in note-generation benchmark, because that corpus does not ship gold SOAP notes in the format this repo expects.

## Dataset Sources

### 1. MedSynth

- Dataset page: https://huggingface.co/datasets/Ahmad0067/MedSynth
- Paper page: https://huggingface.co/papers/2508.01401
- Use in this repo: open transcript-to-note experiments and systems benchmarking
- Why it fits: it provides dialogue-note pairs that can be normalized directly into this repo’s `transcript` + `reference_note` schema

### 2. PriMock57

- GitHub: https://github.com/babylonhealth/primock57
- Paper: https://arxiv.org/abs/2204.00333
- Use in this repo: audio/transcript/note experiments
- Why it fits: it includes audio, manual transcripts, and consultation notes

### 3. MedDialog

- Paper: https://arxiv.org/abs/2004.03329
- Public HF mirror: https://huggingface.co/datasets/OpenMed/MedDialog
- Use in this repo: optional background/long-context conversation source only
- Important: do not treat public MedDialog as the gold-note evaluation dataset for this repository

## Model Access

### Open local default

- `mistralai/Mistral-7B-Instruct-v0.3`
- Used by the default public transcript-to-note configs

### Gated local model

- `meta-llama/Meta-Llama-3-8B-Instruct`
- Official page: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
- This model is gated. Request access on Hugging Face and authenticate with `huggingface-cli login` or `HF_TOKEN`.

### Optional hosted path

- Together API is only needed for the hosted serving comparison configs.
- Required env var: `TOGETHER_API_KEY`

## One-Time Setup

```bash
git clone <your-github-url>
cd clinical-speech-experiments
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
source scripts/setup_storage_env.sh
```

If you need the lighter install path:

```bash
pip install --no-cache-dir -r requirements-lite.txt
```

Optional environment file:

```bash
cp .env.example .env.local
```

Then edit `.env.local` and source it when needed:

```bash
source .env.local
```

## Dataset Preparation

### A. Fetch MedSynth

This writes deterministic `train.jsonl`, `valid.jsonl`, and `test.jsonl` files under `/data/project_runtime/datasets/medsynth`.

```bash
source scripts/setup_storage_env.sh
python3 scripts/fetch_reference_datasets.py --dataset medsynth
```

### B. Build the long-context MedSynth set used by the chunked baseline

```bash
source scripts/setup_storage_env.sh
python3 scripts/prepare_long_context.py \
  --input /data/project_runtime/datasets/medsynth/test.jsonl \
  --output /data/project_runtime/datasets/medsynth/test_long.jsonl \
  --target-words 4000
```

### C. Build the context-bucket MedSynth sets used by the systems benchmarks

```bash
source scripts/setup_storage_env.sh
python3 scripts/prepare_long_context.py \
  --input /data/project_runtime/datasets/medsynth/test.jsonl \
  --output /data/project_runtime/datasets/medsynth/long_context_4000.jsonl \
  --target-words 4000

python3 scripts/build_context_bucket_datasets.py
```

### D. Prepare PriMock57

PriMock57 stores audio with Git LFS, so `git lfs pull` matters.

```bash
source scripts/setup_storage_env.sh
git lfs install
git clone https://github.com/babylonhealth/primock57.git /data/project_runtime/datasets_raw/primock57
cd /data/project_runtime/datasets_raw/primock57
git lfs pull
cd /data/project
python3 scripts/prepare_primock57.py \
  --input-root /data/project_runtime/datasets_raw/primock57
```

That command writes:

- `/data/project_runtime/datasets/primock57/test.jsonl`
- `/data/project_runtime/datasets/primock57/audio_mixed/*.wav`

The mixed audio files are stereo WAVs built from the doctor and patient channels so the current ASR path can consume them as a single file.

## Validation Order

### 1. Smoke test

```bash
source scripts/setup_storage_env.sh
python3 scripts/run_experiment.py --config configs/smoke_naive.yaml
```

### 2. Public transcript-to-note baseline

```bash
source scripts/setup_storage_env.sh
python3 scripts/preflight_storage.py --config configs/naive_baseline.yaml
python3 scripts/run_experiment.py --config configs/naive_baseline.yaml
```

### 3. Public chunked baseline

```bash
source scripts/setup_storage_env.sh
python3 scripts/preflight_storage.py --config configs/chunked_baseline.yaml
python3 scripts/run_experiment.py --config configs/chunked_baseline.yaml
```

### 4. Systems benchmark

```bash
source scripts/setup_storage_env.sh
python3 scripts/preflight_storage.py --config configs/systems_benchmark.yaml
python3 scripts/run_systems_benchmark.py --config configs/systems_benchmark.yaml
```

### 5. Audio-to-note run on PriMock57

Before this run, make sure you have Meta Llama access if you want to use the default config unchanged.

```bash
source scripts/setup_storage_env.sh
source .env.local
python3 scripts/preflight_storage.py --config configs/proposed_system.yaml
python3 scripts/run_experiment.py --config configs/proposed_system.yaml
```

If you do not have gated Meta Llama access yet, copy the config and swap `model.llm_model_id` to an accessible model such as `mistralai/Mistral-7B-Instruct-v0.3`.

## Evaluation

```bash
source scripts/setup_storage_env.sh
python3 scripts/evaluate_run.py \
  --predictions /data/project_runtime/outputs/naive_baseline/predictions.jsonl \
  --output /data/project_runtime/outputs/naive_baseline/metrics.json
```

## Reference Files In The Repo

If you want a quick shape check before running anything expensive, inspect:

- `reference_outputs/smoke_naive/`
- `reference_outputs/systems_benchmark/systems_benchmark_report_excerpt.json`

These are reference-only examples so teammates can compare field names and output layout.

## Recommended Commit Scope

Commit:

- `src/`
- `scripts/`
- `configs/`
- `tests/`
- docs
- smoke fixtures

Do not commit:

- `/data/project_runtime`
- cloned raw datasets
- mixed audio artifacts
- model caches
- generated outputs
- secrets in `.env.local`

## Honest Replication Notes

- The public replication path is centered on MedSynth and PriMock57 because they match the schema this repository needs.
- The public MedDialog corpus is useful as a conversation corpus, but it is not the public gold-note dataset for this repo.
- If you regenerate public results on a different GPU, driver stack, or model revision, runtime numbers will differ.
