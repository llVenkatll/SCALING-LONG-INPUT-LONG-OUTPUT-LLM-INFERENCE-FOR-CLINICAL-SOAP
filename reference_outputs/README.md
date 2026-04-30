# Reference Outputs

This folder contains small example outputs so teammates can quickly verify that regenerated runs have the expected file shapes and key fields.

Important:

- These files are for reference only.
- The `smoke_naive` files come from the repository smoke path and use the fake generator used by `tests/test_runner_smoke.py`.
- The `systems_benchmark` example is a trimmed excerpt of a real local benchmark report and is included to show report structure, not to make a scientific claim.
- Do not cite these files as final benchmark results.

## Included Examples

- `smoke_naive/predictions.jsonl`
- `smoke_naive/metrics.json`
- `smoke_naive/runtime_metrics.json`
- `smoke_naive/sample_runtime_rows.json`
- `smoke_naive/generation_run_rows.json`
- `systems_benchmark/systems_benchmark_report_excerpt.json`

## How To Regenerate

Smoke-style example outputs:

```bash
PYTHONPATH=src python3 -m unittest tests.test_runner_smoke
```

Public dataset replication path:

```bash
source scripts/setup_storage_env.sh
python3 scripts/fetch_reference_datasets.py --dataset medsynth
python3 scripts/run_experiment.py --config configs/smoke_naive.yaml
python3 scripts/run_experiment.py --config configs/naive_baseline.yaml
python3 scripts/run_systems_benchmark.py --config configs/systems_benchmark.yaml
```
