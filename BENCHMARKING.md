# Benchmarking

## Scope

This repository now separates infrastructure smoke tests from paper-usable runtime benchmarking.

Smoke-test configs:

- `configs/smoke_naive.yaml`
- `configs/smoke_chunked.yaml`

Paper-facing benchmark config example:

- `configs/naive_baseline_benchmark.yaml`
- `configs/systems_benchmark.yaml`

## How TTFT Is Measured

TTFT is no longer set equal to full generation latency.

The current measurement path in `src/clinical_speech/models/note_generator.py` uses:

1. host-side tokenization timing before device transfer,
2. explicit device-transfer timing,
3. a Hugging Face `LogitsProcessor` callback to record the first observable post-prefill logits step,
4. a Hugging Face `BaseStreamer` callback to record when the first generated token is emitted,
5. CUDA synchronization around timing boundaries when enabled.

Reported fields:

- `final_tokenization_sec`
- `final_device_transfer_sec`
- `final_prefill_latency_sec`
- `ttft_sec`
- `final_decode_latency_sec`
- `final_generation_latency_sec`
- `end_to_end_latency_sec`

Honesty note:

- `final_prefill_latency_sec` is the closest observable prefill boundary available in the current Hugging Face code path.
- It is an approximation of prefill completion, not a lower-level kernel trace.

## Benchmark Config Controls

The `benchmark:` config section supports:

- `enabled`
- `warmup_runs`
- `repeat_runs`
- `synchronize_cuda`
- `benchmark_only`
- `save_per_run_metrics`
- `save_csv`

## Benchmark Output Artifacts

All benchmark artifacts go under:

```text
/data/project_runtime/benchmarks/<experiment>/
```

Current outputs:

- `runtime_metrics.json`
- `sample_runtime_rows.json`
- `sample_runtime_rows.csv`
- `generation_run_rows.json`
- `generation_run_rows.csv`

Systems benchmark outputs:

- `/data/project_runtime/benchmarks/systems_benchmark/systems/systems_benchmark_report.json`
- `/data/project_runtime/benchmarks/systems_benchmark/systems/systems_summary.csv`
- `/data/project_runtime/benchmarks/systems_benchmark/systems/systems_batch_rows.csv`
- `/data/project_runtime/benchmarks/systems_benchmark/systems/systems_request_rows.csv`
- `results/tables/systems_summary.csv`
- `results/figures/systems_throughput_vs_batch.png`
- `results/figures/systems_requests_vs_batch.png`

## Runtime Breakdown

Per-sample runtime rows distinguish:

- preprocessing / ASR
- chunking
- summarization wall time
- summarization tokenization / device transfer / prefill / TTFT / decode / generation totals
- final note tokenization / device transfer / prefill / TTFT / decode / generation totals
- total end-to-end latency
- GPU memory

## Recommended Benchmark Workflow

1. Source `scripts/setup_storage_env.sh`.
2. Run `python3 scripts/preflight_storage.py --config <config>`.
3. Use a repeated benchmark config such as `configs/naive_baseline_benchmark.yaml`.
4. Inspect both `runtime_metrics.json` and the CSV tables.
5. Do not report smoke-config timings as research results.

## Systems Benchmark Workflow

Use the systems benchmark when you want to compare the naive HF baseline against the custom runtime layer.

```bash
python3 scripts/run_systems_benchmark.py --config configs/systems_benchmark.yaml
```

This benchmark currently compares:

- `hf_sequential`
- `manual_static_batch`

If Triton runtime activation is available on the host, the benchmark can also evaluate the Triton-backed packing path. On the current EC2 box, the Triton kernel is implemented but falls back to the PyTorch reference path because the Python development headers required for Triton JIT activation are missing.

## Serving Stack Comparison Workflow

The repo also supports a compact serving-stack comparison that keeps the workload fixed while varying the local or hosted serving path:

- `HF Sequential (Mistral, local)`
- `Paged Static Batch (Mistral, local)`
- `Paged Static Batch + Triton (Mistral, ours)`
- `HF Sequential (Llama, local)`
- `Together Hosted (Llama)`

Important notes:

- The serving comparison is about backend/runtime behavior, not a claim that one foundation model family is universally better than another.
- The Together model ID is configurable because Together serverless availability can change over time.
- Hosted backends do not expose local KV-cache or GPU-memory internals, so those fields remain `null` / `N/A`.
- Hosted token counts use provider-reported usage when available. If the provider omits completion usage, the benchmark falls back to a local tokenizer estimate and marks that source explicitly in request rows.

Environment variables for the hosted Together baseline:

```bash
export TOGETHER_API_KEY=...
# optional override; defaults to https://api.together.xyz/v1
export TOGETHER_BASE_URL=https://api.together.xyz/v1
```

Example decode-heavy serving comparison:

```bash
source /data/project/.venv/bin/activate
PYTHONPATH=/data/project/src python /data/project/scripts/run_systems_benchmark.py \
  --config /data/project/configs/systems_benchmark_serving_stack_comparison.yaml
```

Example long-context serving comparison:

```bash
source /data/project/.venv/bin/activate
PYTHONPATH=/data/project/src python /data/project/scripts/run_systems_benchmark.py \
  --config /data/project/configs/systems_benchmark_serving_stack_long_context.yaml
```
