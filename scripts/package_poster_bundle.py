from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path("/data/project")
RUNTIME_BENCHMARKS = Path("/data/project_runtime/benchmarks")
OUTPUT_ROOT = PROJECT_ROOT / "poster_download_bundle"
STAGING = OUTPUT_ROOT / "poster_package"
ZIP_PATH = OUTPUT_ROOT / "poster_package.zip"

POSTER_FINAL = PROJECT_ROOT / "poster_final_bundle"
DEEP_STUDY = PROJECT_ROOT / "poster_assets" / "deep_systems_study"
PREFILL_OPT = PROJECT_ROOT / "poster_assets" / "prefill_optimization"

MAX_SUMMARY_FILE_BYTES = 8 * 1024 * 1024


POSTER_FINAL_FILES = [
    "all_runs_grouped.json",
    "all_rows_flat.json",
    "key_results.json",
    "methodology.md",
    "system_design.md",
    "model_architectures.md",
    "qualitative_comparison.md",
    "figures.md",
    "executive_summary.md",
    "manifest.json",
]

FIGURE_FILES = [
    "hero_backend_figure.png",
    "hero_backend_figure.pdf",
    "kv_efficiency_support.png",
    "kv_efficiency_support.pdf",
    "context_scaling_panels.png",
    "context_scaling_panels.pdf",
    "prefill_vs_context.png",
    "prefill_vs_context.pdf",
    "decode_vs_output_length.png",
    "decode_vs_output_length.pdf",
    "throughput_vs_output_length.png",
    "throughput_vs_output_length.pdf",
    "latency_quantiles_batch4.png",
    "latency_quantiles_batch4.pdf",
    "qualitative_review_default_prompt.md",
    "prompt_control_review.md",
    "prompt_control_review.json",
]

PREFILL_OPT_FILES = [
    "prompt_tokens_standard_vs_compact.png",
    "prompt_tokens_standard_vs_compact.pdf",
    "prefill_standard_vs_compact.png",
    "prefill_standard_vs_compact.pdf",
    "prefill_prompt_compression_comparison.csv",
    "two_stage_prompt_token_reduction.png",
    "two_stage_prompt_token_reduction.pdf",
    "two_stage_ttft_prefill_comparison.png",
    "two_stage_ttft_prefill_comparison.pdf",
    "prefill_two_stage_summary.csv",
]

REPORT_FILES = [
    "SYSTEMS_BENCHMARK_REPORT.md",
    "SYSTEMS_POSTER_STUDY.md",
    "BENCHMARKING.md",
    "PREFILL_STUDY_REPORT.md",
    "NEXT_EXPERIMENTS_PLAN.md",
    "MEMORY_CONCURRENCY_REPORT.md",
    "DEMO_PLAN.md",
    "DEMO_USAGE.md",
    "FINAL_POSTER_DIRECTION.md",
]

SCRIPT_FILES = [
    "scripts/build_poster_final_bundle.py",
    "scripts/run_poster_systems_study.py",
    "scripts/run_prompt_control_review.py",
    "scripts/generate_poster_assets.py",
    "scripts/analyze_prefill_prompt_compression.py",
    "scripts/run_prefill_two_stage_study.py",
    "scripts/generate_next_experiment_package.py",
    "scripts/run_live_demo.py",
    "scripts/run_demo.sh",
]

SUMMARY_FILE_NAMES = {
    "systems_summary.csv",
    "manifest.json",
    "systems_benchmark_report.json",
}


def _copy_file(src: Path, dst: Path, copied: list[str], skipped: list[str]) -> None:
    if not src.exists():
        skipped.append(f"missing: {src}")
        return
    if src.is_dir():
        skipped.append(f"directory skipped: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(f"{src} -> {dst}")


def _copy_summary_file(src: Path, copied: list[str], skipped: list[str]) -> None:
    if src.stat().st_size > MAX_SUMMARY_FILE_BYTES:
        skipped.append(f"too large for summary bundle: {src} ({src.stat().st_size} bytes)")
        return
    rel = src.relative_to(RUNTIME_BENCHMARKS)
    _copy_file(src, STAGING / "benchmark_summaries" / rel, copied, skipped)


def _write_readme() -> None:
    text = """# Poster Package

This folder is a compact, poster-focused export of the LLM systems project results. It is not a full project backup and intentionally excludes model weights, caches, checkpoints, `.venv`, and large runtime artifacts.

## Folder Contents

- `poster_final_bundle/`: aggregated JSON data and poster-ready markdown sections. Start with `executive_summary.md`, then `key_results.json`.
- `figures/`: poster-ready PNG/PDF figures. The hero figure is `hero_backend_figure.png`.
- `figures/prefill_optimization/`: prompt-compression and two-stage transcript-to-facts prefill study assets.
- `reports/`: main project benchmark/report markdown files.
- `scripts/`: reproducibility scripts used to rebuild the final bundle, study assets, prompt review, and poster figures.
- `benchmark_summaries/`: lightweight benchmark summaries only, mostly `systems_summary.csv`, `systems_benchmark_report.json`, and `manifest.json`.

## Read First

1. `poster_final_bundle/executive_summary.md`
2. `poster_final_bundle/system_design.md`
3. `poster_final_bundle/figures.md`
4. `poster_final_bundle/qualitative_comparison.md`

## Hero Figure

Use `figures/hero_backend_figure.png` for the main poster result. It summarizes the decode-heavy high-batch regime where the custom paged-KV + Triton backend has the strongest throughput result.

For the prefill limitation story, use `figures/prefill_optimization/two_stage_prompt_token_reduction.png` together with `figures/prefill_optimization/two_stage_ttft_prefill_comparison.png`.

## Rebuild Commands

From `/data/project` on the EC2 instance:

```bash
source /data/project/.venv/bin/activate
python /data/project/scripts/build_poster_final_bundle.py
python /data/project/scripts/package_poster_bundle.py
```

No experiments are rerun by the package script. It only copies existing artifacts.
"""
    (STAGING / "README.md").write_text(text, encoding="utf-8")


def _zip_staging() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(STAGING.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUTPUT_ROOT))


def _tree_lines(root: Path, max_lines: int = 220) -> list[str]:
    lines = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        depth = len(rel.parts) - 1
        prefix = "  " * depth
        suffix = "/" if path.is_dir() else ""
        lines.append(f"{prefix}{rel.name}{suffix}")
        if len(lines) >= max_lines:
            lines.append("... truncated ...")
            break
    return lines


def main() -> None:
    copied: list[str] = []
    skipped: list[str] = []

    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True, exist_ok=True)

    for name in POSTER_FINAL_FILES:
        _copy_file(POSTER_FINAL / name, STAGING / "poster_final_bundle" / name, copied, skipped)

    for name in FIGURE_FILES:
        _copy_file(DEEP_STUDY / name, STAGING / "figures" / name, copied, skipped)
    for name in PREFILL_OPT_FILES:
        _copy_file(PREFILL_OPT / name, STAGING / "figures" / "prefill_optimization" / name, copied, skipped)
    _copy_file(PROJECT_ROOT / "SYSTEMS_POSTER_STUDY.md", STAGING / "figures" / "SYSTEMS_POSTER_STUDY.md", copied, skipped)

    for name in REPORT_FILES:
        _copy_file(PROJECT_ROOT / name, STAGING / "reports" / name, copied, skipped)

    for name in SCRIPT_FILES:
        _copy_file(PROJECT_ROOT / name, STAGING / "scripts" / Path(name).name, copied, skipped)

    if RUNTIME_BENCHMARKS.exists():
        for src in sorted(RUNTIME_BENCHMARKS.rglob("*")):
            if src.is_file() and (src.name in SUMMARY_FILE_NAMES or "summary" in src.name.lower()):
                _copy_summary_file(src, copied, skipped)
    else:
        skipped.append(f"missing benchmark root: {RUNTIME_BENCHMARKS}")

    _write_readme()
    copied.append(f"generated README -> {STAGING / 'README.md'}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "staging_dir": str(STAGING),
        "zip_path": str(ZIP_PATH),
        "files_copied": len(copied),
        "files_skipped": len(skipped),
        "copied": copied,
        "skipped": skipped,
    }
    (STAGING / "package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _zip_staging()
    zip_size = ZIP_PATH.stat().st_size

    print(json.dumps({**manifest, "zip_size_bytes": zip_size}, indent=2))
    print("\nFolder tree:")
    print("\n".join(_tree_lines(STAGING)))
    print(f"\nFinal zip: {ZIP_PATH}")
    print(f"Zip size: {zip_size / (1024 * 1024):.2f} MB")
    print("\nSCP template:")
    print("scp -i /path/to/key.pem ubuntu@<EC2_PUBLIC_IP>:/data/project/poster_download_bundle/poster_package.zip ~/Downloads/")


if __name__ == "__main__":
    main()
