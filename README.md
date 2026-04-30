# Clinical Speech Experiments

This repository contains the code, configs, scripts, and tests for the clinical speech LLM benchmarking scaffold in `src/clinical_speech`.

The repo is set up so teammates can reproduce experiments without storing datasets, model caches, checkpoints, or benchmark outputs inside Git.

## What To Commit

- `src/`
- `scripts/`
- `configs/`
- `tests/`
- `data/fixtures/`
- `pyproject.toml`
- `requirements*.txt`
- setup and usage docs such as `README_EXPERIMENTS.md`, `RUN_SAFE.md`, `BENCHMARKING.md`, and `EVALUATION.md`

## What Not To Commit

- `.venv/`, `__pycache__/`, notebook checkpoints, and local editor noise
- real datasets or patient-sensitive data
- API keys, `.env` files, SSH keys, or provider tokens
- runtime outputs under `results/`, `poster_download_bundle/`, `poster_final_bundle/`, and generated poster assets
- local caches, checkpoints, logs, and archives generated during runs

If a large binary is truly source material and must live with the repo, use Git LFS and document why it is needed. Otherwise, prefer regenerating it from source or attaching it to a release.

## Quick Start

```bash
git clone <your-github-url>
cd clinical-speech-experiments
python3 -m venv .venv
source .venv/bin/activate
source scripts/setup_storage_env.sh
pip install -r requirements.txt
```

If you are on a storage-constrained machine, use the lighter install path:

```bash
pip install --no-cache-dir -r requirements-lite.txt
```

## Reproducible Teammate Workflow

1. Clone the repository and create a fresh virtual environment.
2. Run `source scripts/setup_storage_env.sh` in every new shell before installs or experiments.
3. Keep datasets under `/data/project_runtime/datasets`, not in the repository.
4. Run a preflight check before long jobs:

```bash
python3 scripts/preflight_storage.py --config configs/naive_baseline.yaml
```

5. Validate the environment with the tiny smoke run:

```bash
python3 scripts/run_experiment.py --config configs/smoke_naive.yaml
```

6. Move to real experiments only after the smoke run passes.

## Useful Commands

Run the naive baseline:

```bash
python3 scripts/run_experiment.py --config configs/naive_baseline.yaml
```

Run repeated benchmarking:

```bash
python3 scripts/run_experiment.py --config configs/naive_baseline_benchmark.yaml
python3 scripts/run_systems_benchmark.py --config configs/systems_benchmark.yaml
```

Run tests:

```bash
PYTHONPATH=src pytest -q
```

## More Detail

- Repo upload checklist: `GITHUB_UPLOAD_CHECKLIST.md`
- Public replication path: `REPLICATION.md`
- Experiment setup: `README_EXPERIMENTS.md`
- Safe runtime usage: `RUN_SAFE.md`
- Benchmark interpretation: `BENCHMARKING.md`
- Evaluation rules: `EVALUATION.md`
