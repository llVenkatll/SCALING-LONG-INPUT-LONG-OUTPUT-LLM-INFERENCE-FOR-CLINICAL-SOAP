# Safe Run Guide

## Where To Work

- Keep the repo under `/data/project`.
- Keep every heavy runtime artifact under `/data/project_runtime`.
- Do not use `/home/ubuntu`, `~`, `/tmp`, or the repo root for datasets, model downloads, checkpoints, logs, profiler traces, or caches.

## One-Time Setup

Run this in every new shell before installing or launching jobs:

```bash
cd /data/project
source scripts/setup_storage_env.sh
```

Optional sanity check:

```bash
python scripts/preflight_storage.py --config configs/naive_baseline.yaml
```

## Safe Directory Layout

```text
/data/project_runtime/
    datasets/
    checkpoints/
    logs/
    cache/
    hf/
    torch/
    profiler/
    outputs/
    tmp/
    benchmarks/
```

## Where To Put Data

Place input datasets under `/data/project_runtime/datasets`, for example:

```text
/data/project_runtime/datasets/medsynth/test.jsonl
/data/project_runtime/datasets/medsynth/test_long.jsonl
/data/project_runtime/datasets/primock57/test.jsonl
```

## Safe Launch Commands

Prepare long-context datasets:

```bash
cd /data/project
source scripts/setup_storage_env.sh
python scripts/prepare_long_context.py \
  --input /data/project_runtime/datasets/medsynth/test.jsonl \
  --output /data/project_runtime/datasets/medsynth/test_long.jsonl \
  --target-words 5000
```

Run an experiment:

```bash
cd /data/project
source scripts/setup_storage_env.sh
python scripts/preflight_storage.py --config configs/naive_baseline.yaml
python scripts/run_experiment.py --config configs/naive_baseline.yaml
```

Evaluate predictions:

```bash
cd /data/project
source scripts/setup_storage_env.sh
python scripts/evaluate_run.py \
  --predictions /data/project_runtime/outputs/naive_baseline/predictions.jsonl \
  --output /data/project_runtime/outputs/naive_baseline/metrics.json
```

## What Not To Do

- Do not run long jobs without sourcing `scripts/setup_storage_env.sh`.
- Do not point `--output` at a relative path unless you want it resolved under the managed `/data/project_runtime` directories.
- Do not store model caches in `~/.cache`, even temporarily.
- Do not run the legacy notebooks as the primary EC2 workflow. Use the CLI scripts instead.
- Do not create `outputs/`, `logs/`, `checkpoints/`, `wandb/`, `tmp/`, or `.cache/` directories in the repo checkout.

## Monitoring

Check free space on both mounts:

```bash
df -h / /data
```

Inspect the managed runtime tree:

```bash
du -sh /data/project_runtime/*
```

Watch temp and cache growth:

```bash
du -sh /data/project_runtime/tmp /data/project_runtime/cache /data/project_runtime/hf
```

Tail the active run log:

```bash
tail -f /data/project_runtime/logs/naive_baseline/run.log
```

## Recovery If `/` Starts Filling

1. Stop the job.
2. Check whether any cache or temp directory escaped to the root disk:

```bash
du -sh ~/.cache ~/.local /tmp 2>/dev/null
```

3. Re-source `scripts/setup_storage_env.sh`.
4. Re-run `python scripts/preflight_storage.py --config <config>` until it passes.
5. Remove stale root-disk caches only after confirming they are not needed.

## Long-Run Hygiene

- Logs are rotated with bounded size.
- Recovery checkpoints can be enabled with `output.save_frequency`.
- Recovery checkpoints are pruned to `output.max_checkpoints_to_keep`.
- Temporary run directories are created under `/data/project_runtime/tmp` and cleaned up on success by default.
- Raw intermediates are off by default and only saved when `output.save_raw_intermediates: true`.
