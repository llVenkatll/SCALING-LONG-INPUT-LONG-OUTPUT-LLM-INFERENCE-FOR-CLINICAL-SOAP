# Clinical Speech LLM Experiment Scaffold

This codebase is a practical scaffold for the project in `final_proposal.pdf`:

- long-input clinical speech/transcript processing,
- long-output SOAP note generation,
- baseline, chunked, and proposed experiment modes,
- benchmarking for latency, memory, and throughput,
- evaluation hooks for WER, ROUGE-L, and BERTScore.

It is designed to get you to a runnable experiment structure quickly. Some parts are fully usable now, while dataset-specific preparation and final model access still depend on what you provide.

For the public GitHub-shareable replication path in this repository, use:

- `MedSynth` for transcript-to-note runs
- `PriMock57` for audio/transcript/note runs

See `REPLICATION.md` for exact source links and commands.

This repository now distinguishes explicitly between:

- smoke-test runs that validate infrastructure and storage safety,
- benchmark runs that produce paper-usable runtime artifacts,
- real evaluation runs on a properly prepared dataset.

Do not treat smoke-test numbers as scientific results.

## What You Need To Provide

### 1. Datasets

You need data in one of these forms:

#### Option A: Transcript-first experiments

Best if you want to get note-generation experiments running first.

Provide JSONL files with this schema:

```json
{"id":"sample-1","transcript":"doctor patient conversation ...","reference_note":"SOAP note text ...","audio_path":null,"metadata":{"source":"medsynth"}}
```

Required fields:

- `id`
- `transcript`
- `reference_note`

Optional fields:

- `audio_path`
- `metadata`

Recommended public open-data locations on this machine:

- `/data/project_runtime/datasets/medsynth/train.jsonl`
- `/data/project_runtime/datasets/medsynth/valid.jsonl`
- `/data/project_runtime/datasets/medsynth/test.jsonl`
- `/data/project_runtime/datasets/primock57/test.jsonl`

For quick infrastructure validation, the repo also ships tiny fixture datasets:

- `data/fixtures/smoke_notes.jsonl`
- `data/fixtures/smoke_notes_long.jsonl`

These are intentionally tiny and are only for smoke tests.

#### Option B: Audio + transcript experiments

Best if you want to evaluate the full ASR-to-note pipeline.

Provide JSONL with:

```json
{"id":"sample-1","audio_path":"data/raw/audio/sample1.wav","transcript":"ground truth transcript","reference_note":"SOAP note text ...","metadata":{"source":"primock57"}}
```

Required:

- `id`
- `audio_path`
- `reference_note`

Strongly recommended:

- `transcript` for WER evaluation

### 2. Models

You need to decide what you will actually run:

#### ASR model

Recommended default:

- `openai/whisper-small`

If you need better quality and have more GPU:

- `openai/whisper-medium`
- `openai/whisper-large-v3`

#### Note generation model

The proposal uses `LLaMA3-8B`. For this scaffold, you can set:

- `meta-llama/Meta-Llama-3-8B-Instruct` if you have access

If you do not have gated access yet, use a fallback:

- `mistralai/Mistral-7B-Instruct-v0.3`
- `Qwen/Qwen2.5-7B-Instruct`

### 3. Prompt format

You need to confirm the note style you want to generate. This scaffold assumes SOAP note generation by default.

### 4. Hardware details

You should record:

- GPU type,
- GPU memory,
- batch size used,
- dtype / quantization mode,
- sequence length limits.

## Config Layout

Configs now support inheritance through a top-level `extends:` field.

Useful configs:

- `configs/base.yaml`
- `configs/naive_baseline.yaml`
- `configs/chunked_baseline.yaml`
- `configs/proposed_system.yaml`
- `configs/naive_baseline_8bit.yaml`
- `configs/proposed_system_fp16.yaml`
- `configs/naive_baseline_benchmark.yaml`
- `configs/systems_benchmark.yaml`
- `configs/smoke_naive.yaml`
- `configs/smoke_chunked.yaml`

## What Experiments You Should Run

Start with these in order:

1. `Transcript -> Note` naive baseline
2. `Transcript -> Note` chunked baseline
3. `Audio -> ASR -> Note` baseline
4. Quantized note-generation baseline
5. Chunk-size / overlap ablation

## Recommended First Deliverable

For the mid-term, the safest minimum set is:

1. Use transcript-first JSONL data.
2. Run a note-generation baseline on 20 to 50 examples.
3. Measure:
   - generation latency,
   - time to first token,
   - peak GPU memory,
   - throughput,
   - ROUGE-L,
   - BERTScore.
4. If audio is available, run Whisper on a smaller subset and report WER.

For journal-quality work, move beyond the smoke fixtures immediately and normalize a real dataset into the expected schema with `scripts/prepare_dataset.py`.

## Setup

```bash
cd /data/project
source scripts/setup_storage_env.sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you hit a quota error during install, try the lighter path first:

```bash
source scripts/setup_storage_env.sh
rm -rf ~/.cache/huggingface ~/.cache/torch ~/.cache/wandb
pip install --no-cache-dir -r requirements-lite.txt
pip install --no-cache-dir torch transformers accelerate
```

On macOS, `bitsandbytes` is skipped automatically.

On quota-limited Linux clusters, `pip install -r requirements.txt` may try to install multi-GB CUDA wheels into your home Conda env. In that case:

```bash
pip install --no-cache-dir -r requirements-cluster-lite.txt
```

Then install PyTorch separately only after moving the environment to scratch or project storage.

## Example Commands

Prepare a synthetic long-context split by concatenating transcripts:

```bash
python3 scripts/prepare_long_context.py \
  --input /data/project_runtime/datasets/medsynth/test.jsonl \
  --output /data/project_runtime/datasets/medsynth/test_long.jsonl \
  --target-words 5000
```

Normalize a real dataset into the required JSONL schema:

```bash
python3 scripts/prepare_dataset.py \
  --input /data/project_runtime/datasets/raw/source.jsonl \
  --output /data/project_runtime/datasets/medsynth/test.jsonl \
  --id-field id \
  --transcript-field transcript \
  --reference-note-field reference_note \
  --audio-path-field audio_path \
  --metadata-field metadata \
  --dataset-name medsynth
```

Run the tiny smoke test:

```bash
python3 scripts/run_experiment.py --config configs/smoke_naive.yaml
```

Run the naive baseline:

```bash
python3 scripts/preflight_storage.py --config configs/naive_baseline.yaml
python3 scripts/run_experiment.py --config configs/naive_baseline.yaml
```

Run the chunked baseline:

```bash
python3 scripts/run_experiment.py --config configs/chunked_baseline.yaml
```

Run the proposed-system scaffold:

```bash
python3 scripts/run_experiment.py --config configs/proposed_system.yaml
```

Run repeated benchmarking without scoring:

```bash
python3 scripts/run_experiment.py --config configs/naive_baseline_benchmark.yaml
```

Run the custom systems benchmark:

```bash
python3 scripts/run_systems_benchmark.py --config configs/systems_benchmark.yaml
```

Score predictions:

```bash
python3 scripts/evaluate_run.py \
  --predictions /data/project_runtime/outputs/naive_baseline/predictions.jsonl \
  --output /data/project_runtime/outputs/naive_baseline/metrics.json
```

## Safe EC2 Workflow

- Keep the repo itself in `/data/project`.
- Put datasets under `/data/project_runtime/datasets`.
- Keep outputs, logs, checkpoints, caches, and temp files under `/data/project_runtime`.
- Use the CLI scripts, not the legacy notebooks, for storage-safe EC2 runs.
- Use `REPLICATION.md` for the public dataset links and exact prep commands.
- Use `configs/smoke_*.yaml` only for smoke testing.
- Use `configs/*benchmark*.yaml` when you want repeated runtime measurements.
- See `RUN_SAFE.md` for the full operating guide.
- See `BENCHMARKING.md` and `EVALUATION.md` for paper-facing guidance.

## What Is Implemented vs Placeholder

Implemented:

- config loading
- JSONL dataset loading
- SOAP prompting
- naive transcript-to-note generation runner
- chunked transcript processing runner
- synchronized runtime timing with tokenization/device-transfer/prefill/TTFT/decode breakdowns
- aggregate metric computation hooks
- repeated benchmark artifact output in JSON and CSV
- config inheritance for ablation-style runs
- smoke-test fixture configs

Placeholder / project-specific:

- dataset-specific raw-data preparation for MedSynth / PriMock57 and any optional external conversation corpora
- true streaming ASR
- true sliding-window KV-cache reuse
- LoRA fine-tuning pipeline

Those are intentionally left as clean extension points so you can get baseline experiments running first.
