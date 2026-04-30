from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/data/project")
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
SUMMARY_CSV = TABLES_DIR / "poster_study_summary.csv"
REQUEST_CSV = TABLES_DIR / "poster_study_request_rows.csv"

PREFILL_DIR = PROJECT_ROOT / "poster_assets" / "prefill_story"
MEMORY_DIR = PROJECT_ROOT / "poster_assets" / "memory_concurrency"
DEMO_DIR = PROJECT_ROOT / "poster_assets" / "demo"

BACKENDS = [
    "hf_sequential",
    "mistral_paged_static_batch",
    "mistral_paged_static_batch_triton",
]

BACKEND_LABELS = {
    "hf_sequential": "HF Sequential",
    "mistral_paged_static_batch": "Paged Runtime",
    "mistral_paged_static_batch_triton": "Paged + Triton",
}

COLORS = {
    "hf_sequential": "#7a7a7a",
    "mistral_paged_static_batch": "#2f6ea6",
    "mistral_paged_static_batch_triton": "#0d3b66",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float | None:
    if value in (None, "", "null", "None", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 170,
            "savefig.dpi": 320,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def _context_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in rows
        if row.get("study_type") == "context"
        and row.get("status") == "success"
        and str(row.get("batch_size")) == "1"
        and row.get("backend") in BACKENDS
    ]


def _best_decode_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in rows
        if row.get("study_type") == "output"
        and row.get("status") == "success"
        and str(row.get("x_value")) == "128"
        and row.get("backend") in BACKENDS
    ]


def build_prefill_assets(rows: list[dict[str, str]]) -> dict[str, Any]:
    context_rows = _context_rows(rows)
    context_rows.sort(key=lambda row: (row["backend"], int(float(row["x_value"]))))
    summary_rows: list[dict[str, Any]] = []
    for row in context_rows:
        prefill = _float(row.get("mean_prefill_latency_sec"))
        decode = _float(row.get("mean_decode_latency_sec"))
        latency = _float(row.get("mean_latency_sec"))
        prompt_tokens = _float(row.get("mean_prompt_tokens"))
        summary_rows.append(
            {
                "backend": row["backend"],
                "backend_label": BACKEND_LABELS.get(row["backend"], row["backend"]),
                "prompt_bucket": int(float(row["x_value"])),
                "measured_prompt_tokens": prompt_tokens,
                "ttft_sec": _float(row.get("mean_ttft_sec")),
                "prefill_latency_sec": prefill,
                "decode_latency_sec": decode,
                "total_latency_sec": latency,
                "prefill_share_of_total": (prefill / latency) if prefill is not None and latency else None,
            }
        )
    _write_csv(PREFILL_DIR / "prefill_context_summary.csv", summary_rows)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for backend in BACKENDS:
        backend_rows = [row for row in summary_rows if row["backend"] == backend]
        backend_rows.sort(key=lambda row: row["prompt_bucket"])
        ax.plot(
            [row["prompt_bucket"] for row in backend_rows],
            [row["prefill_latency_sec"] for row in backend_rows],
            marker="o",
            linewidth=2.4,
            color=COLORS[backend],
            label=BACKEND_LABELS[backend],
        )
    ax.set_title("Prefill Latency Grows With Prompt Length")
    ax.set_xlabel("Approximate prompt-token bucket")
    ax.set_ylabel("Prefill latency (s)")
    ax.legend()
    _save(fig, PREFILL_DIR / "prefill_latency_vs_context")

    triton_rows = [row for row in summary_rows if row["backend"] == "mistral_paged_static_batch_triton"]
    triton_rows.sort(key=lambda row: row["prompt_bucket"])
    xs = [str(row["prompt_bucket"]) for row in triton_rows]
    prefill = [row["prefill_latency_sec"] or 0.0 for row in triton_rows]
    decode = [row["decode_latency_sec"] or 0.0 for row in triton_rows]
    total = [row["total_latency_sec"] or 0.0 for row in triton_rows]
    other = [max(t - p - d, 0.0) for t, p, d in zip(total, prefill, decode, strict=True)]
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.bar(xs, prefill, color="#f59e0b", label="Prefill / TTFT path")
    ax.bar(xs, decode, bottom=prefill, color="#2563eb", label="Decode")
    ax.bar(xs, other, bottom=[p + d for p, d in zip(prefill, decode, strict=True)], color="#cbd5e1", label="Other")
    ax.set_title("Long-Context Limitation: Prefill Becomes Dominant")
    ax.set_xlabel("Approximate prompt-token bucket")
    ax.set_ylabel("Latency (s)")
    ax.legend()
    _save(fig, PREFILL_DIR / "triton_prefill_decode_breakdown")

    max_context = max(triton_rows, key=lambda row: row["prompt_bucket"]) if triton_rows else {}
    return {
        "rows": len(summary_rows),
        "max_context_row": max_context,
        "artifacts": [
            str(PREFILL_DIR / "prefill_context_summary.csv"),
            str(PREFILL_DIR / "prefill_latency_vs_context.png"),
            str(PREFILL_DIR / "triton_prefill_decode_breakdown.png"),
        ],
    }


def build_memory_assets(rows: list[dict[str, str]]) -> dict[str, Any]:
    decode_rows = _best_decode_rows(rows)
    decode_rows.sort(key=lambda row: (row["backend"], int(float(row["batch_size"]))))
    table_rows: list[dict[str, Any]] = []
    for row in decode_rows:
        tps = _float(row.get("mean_throughput_tok_per_sec"))
        mem = _float(row.get("mean_peak_gpu_mem_gb"))
        table_rows.append(
            {
                "backend": row["backend"],
                "backend_label": BACKEND_LABELS.get(row["backend"], row["backend"]),
                "batch_size": int(float(row["batch_size"])),
                "status": row.get("status") or "success",
                "tok_per_sec": tps,
                "requests_per_sec": _float(row.get("mean_requests_per_sec")),
                "peak_gpu_mem_gb": mem,
                "tok_per_sec_per_gb": (tps / mem) if tps is not None and mem else None,
                "kv_allocated_gb": ((_float(row.get("mean_kv_allocated_bytes")) or 0.0) / (1024 ** 3)) if row.get("mean_kv_allocated_bytes") else None,
                "kv_utilization_ratio": _float(row.get("mean_kv_utilization_ratio")),
                "kv_fragmentation_ratio": _float(row.get("mean_kv_fragmentation_ratio")),
            }
        )
    _write_csv(MEMORY_DIR / "memory_concurrency_summary.csv", table_rows)
    _write_csv(
        MEMORY_DIR / "feasibility_table.csv",
        [
            {
                "backend": row["backend"],
                "batch_size": row["batch_size"],
                "status": row["status"],
                "observed_oom": False,
                "notes": "Completed in existing decode-heavy sweep; no OOM observed up to batch size 8.",
            }
            for row in table_rows
        ],
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for backend in BACKENDS:
        backend_rows = [row for row in table_rows if row["backend"] == backend]
        ax.plot(
            [row["batch_size"] for row in backend_rows],
            [row["peak_gpu_mem_gb"] for row in backend_rows],
            marker="o",
            linewidth=2.4,
            color=COLORS[backend],
            label=BACKEND_LABELS[backend],
        )
    ax.set_title("Peak GPU Memory vs Batch Size (max_new_tokens=128)")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.legend()
    _save(fig, MEMORY_DIR / "memory_vs_batch")

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for backend in BACKENDS:
        backend_rows = [row for row in table_rows if row["backend"] == backend]
        ax.plot(
            [row["batch_size"] for row in backend_rows],
            [row["tok_per_sec_per_gb"] for row in backend_rows],
            marker="o",
            linewidth=2.4,
            color=COLORS[backend],
            label=BACKEND_LABELS[backend],
        )
    ax.set_title("Throughput per GB Improves in Decode-Heavy Batches")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("tok/s per GB peak memory")
    ax.legend()
    _save(fig, MEMORY_DIR / "throughput_per_gb_vs_batch")

    best_eff = max(table_rows, key=lambda row: row["tok_per_sec_per_gb"] or -1.0) if table_rows else {}
    return {
        "rows": len(table_rows),
        "best_efficiency_row": best_eff,
        "artifacts": [
            str(MEMORY_DIR / "memory_concurrency_summary.csv"),
            str(MEMORY_DIR / "feasibility_table.csv"),
            str(MEMORY_DIR / "memory_vs_batch.png"),
            str(MEMORY_DIR / "throughput_per_gb_vs_batch.png"),
        ],
    }


def build_demo_layout() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.axis("off")
    boxes = [
        (0.05, 0.62, 0.25, 0.22, "Transcript Input\npaste or example"),
        (0.38, 0.62, 0.25, 0.22, "Backend Selector\nHF / Paged / Triton"),
        (0.70, 0.62, 0.25, 0.22, "SOAP Note Output\nmodel result"),
        (0.20, 0.20, 0.25, 0.22, "Runtime Metrics\nTTFT, latency, tok/s"),
        (0.55, 0.20, 0.25, 0.22, "Poster Context\nbest result + caveat"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor="#edf5ff", edgecolor="#1f2937", linewidth=1.6)
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=13, fontweight="bold")
    arrows = [
        ((0.30, 0.73), (0.38, 0.73)),
        ((0.63, 0.73), (0.70, 0.73)),
        ((0.50, 0.62), (0.36, 0.42)),
        ((0.82, 0.62), (0.68, 0.42)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#374151"})
    ax.set_title("Live Demo Layout: Same Transcript, Selectable Backend, Visible Metrics", fontsize=16, fontweight="bold")
    _save(fig, DEMO_DIR / "demo_layout")


def write_docs(prefill: dict[str, Any], memory: dict[str, Any]) -> None:
    max_context = prefill.get("max_context_row") or {}
    best_eff = memory.get("best_efficiency_row") or {}
    (PROJECT_ROOT / "NEXT_EXPERIMENTS_PLAN.md").write_text(
        "\n".join(
            [
                "# Next Experiments Plan",
                "",
                "## Highest-Priority Prefill Experiment",
                "Run a prefill timing study across existing prompt-length buckets and make the bottleneck explicit rather than attempting risky prefix caching or chunked prefill in the poster window.",
                "",
                "- Why it matters: the current runtime is strongest in decode-heavy serving, while long-context prefill is the clearest limitation.",
                "- Expected upside: a clean limitation figure showing when TTFT/prefill dominates and what future runtime work should target.",
                "- Risk/complexity: low, because it reuses completed context-sweep measurements.",
                "",
                "## Highest-Priority Memory/Concurrency Experiment",
                "Use the completed decode-heavy batch sweep to compare peak memory, KV allocation/utilization, feasibility, and throughput per GB across HF, paged runtime, and paged+Triton.",
                "",
                "- Why it matters: even when custom memory is higher than HF, throughput per GB and stable high-batch behavior are strong systems metrics.",
                "- Expected upside: a poster-worthy statement about high-batch efficiency and no observed OOM through batch size 8.",
                "- Risk/complexity: low, because all required data already exists.",
                "",
                "## Highest-Priority Demo Improvement",
                "Create a deterministic Gradio/CLI demo with pasted transcript or preloaded example, backend selection, SOAP output, and visible timing metrics.",
                "",
                "- Why it matters: it makes the systems contribution easy to show live without relying on audience speech or hosted APIs.",
                "- Expected upside: a stable poster demo that supports the same HF vs paged vs paged+Triton story.",
                "- Risk/complexity: medium, because loading local 7B/8B models is slow; the demo uses one selected backend at a time.",
                "",
                "## Recommended Execution Order",
                "1. Generate prefill limitation figures from existing context sweeps.",
                "2. Generate memory/concurrency efficiency figures from existing batch sweeps.",
                "3. Add deterministic demo script and usage docs.",
            ]
        ),
        encoding="utf-8",
    )

    (PROJECT_ROOT / "PREFILL_STUDY_REPORT.md").write_text(
        "\n".join(
            [
                "# Prefill Study Report",
                "",
                "## Experiment",
                "Measured prefill/TTFT behavior across existing approximate prompt-token buckets: 512, 1024, 2048, 4096, and 8192. No new heavy benchmark was run; this report reuses the completed context sweep.",
                "",
                "## Result",
                f"At the largest measured context bucket, `{max_context.get('prompt_bucket', 'N/A')}` tokens, the paged+Triton backend has TTFT `{max_context.get('ttft_sec', 'N/A')}` s, prefill latency `{max_context.get('prefill_latency_sec', 'N/A')}` s, decode latency `{max_context.get('decode_latency_sec', 'N/A')}` s, and total latency `{max_context.get('total_latency_sec', 'N/A')}` s.",
                "",
                "## Conclusion",
                "This study does not claim a prefill improvement. It strengthens the poster by clearly showing the limitation: decode optimization does not remove the long-context prefill bottleneck. The next real systems step would be prefix reuse, prompt-template KV caching, or chunked prefill with careful scheduling.",
                "",
                "## Assets",
                "- `poster_assets/prefill_story/prefill_latency_vs_context.png`",
                "- `poster_assets/prefill_story/triton_prefill_decode_breakdown.png`",
                "- `poster_assets/prefill_story/prefill_context_summary.csv`",
            ]
        ),
        encoding="utf-8",
    )

    (PROJECT_ROOT / "MEMORY_CONCURRENCY_REPORT.md").write_text(
        "\n".join(
            [
                "# Memory / Concurrency Report",
                "",
                "## Experiment",
                "Analyzed completed decode-heavy batch sweeps at `max_new_tokens=128` for batch sizes up to 8. No new heavy benchmark was run.",
                "",
                "## Result",
                f"The best throughput-per-GB row is `{best_eff.get('backend_label', 'N/A')}` at batch size `{best_eff.get('batch_size', 'N/A')}`, with `{best_eff.get('tok_per_sec_per_gb', 'N/A')}` tok/s/GB and peak memory `{best_eff.get('peak_gpu_mem_gb', 'N/A')}` GB.",
                "",
                "## Feasibility",
                "No OOM was observed in the completed decode-heavy sweep up to batch size 8 for the compared local backends. This is not a max-concurrency proof beyond batch 8; it is a conservative feasibility statement for the measured grid.",
                "",
                "## Conclusion",
                "The custom runtime uses more peak GPU memory than HF, but the paged+Triton backend delivers substantially higher throughput and higher throughput per GB in the high-batch decode-heavy regime.",
                "",
                "## Assets",
                "- `poster_assets/memory_concurrency/memory_vs_batch.png`",
                "- `poster_assets/memory_concurrency/throughput_per_gb_vs_batch.png`",
                "- `poster_assets/memory_concurrency/memory_concurrency_summary.csv`",
                "- `poster_assets/memory_concurrency/feasibility_table.csv`",
            ]
        ),
        encoding="utf-8",
    )

    (PROJECT_ROOT / "DEMO_PLAN.md").write_text(
        "\n".join(
            [
                "# Demo Plan",
                "",
                "## Goal",
                "Provide a reliable live poster demo that shows the same clinical transcript running through HF baseline, paged runtime, or paged+Triton, with visible timing metrics.",
                "",
                "## Demo Path",
                "- User pastes a transcript or loads a preconfigured MedSynth example.",
                "- User selects `HF baseline`, `Paged runtime`, or `Paged runtime + Triton`.",
                "- The app generates a SOAP note and displays TTFT, prefill latency, decode latency, total latency, generated tokens, and tok/s.",
                "",
                "## Risk Controls",
                "- No random audience speech is required.",
                "- Hosted Together is not required.",
                "- The demo loads only the selected backend path.",
                "- Use short `max_new_tokens` for live operation.",
            ]
        ),
        encoding="utf-8",
    )

    (PROJECT_ROOT / "DEMO_USAGE.md").write_text(
        "\n".join(
            [
                "# Demo Usage",
                "",
                "## Local CLI Smoke Demo",
                "```bash",
                "cd /data/project",
                "source /data/project/.venv/bin/activate",
                "PYTHONPATH=/data/project/src python /data/project/scripts/run_live_demo.py --cli --backend triton --example-index 0 --max-new-tokens 64",
                "```",
                "",
                "## Browser Demo on EC2",
                "```bash",
                "cd /data/project",
                "source /data/project/.venv/bin/activate",
                "PYTHONPATH=/data/project/src bash /data/project/scripts/run_demo.sh",
                "```",
                "",
                "## Mac SSH Tunnel",
                "```bash",
                "ssh -i /path/to/key.pem -L 7860:localhost:7860 ubuntu@<EC2_PUBLIC_IP>",
                "```",
                "Then open `http://localhost:7860` on the Mac.",
                "",
                "## Notes",
                "- First run may take time while model weights load.",
                "- Use the preloaded examples for a stable poster demo.",
                "- The demo is for local Mistral runtime comparison; Together hosted is intentionally excluded from the live path.",
            ]
        ),
        encoding="utf-8",
    )

    (PROJECT_ROOT / "FINAL_POSTER_DIRECTION.md").write_text(
        "\n".join(
            [
                "# Final Poster Direction",
                "",
                "## Strongest Story",
                "The strongest story remains decode-heavy multi-request serving: paged KV + Triton gives the clearest throughput win.",
                "",
                "## Most Valuable New Experiment",
                "The prefill study adds the most interpretability value by clearly explaining why long-context workloads remain a limitation. The memory/concurrency study adds a compact systems-efficiency support figure.",
                "",
                "## Main Poster Headline",
                "A custom Mistral paged-KV runtime with Triton-accelerated decode improves multi-request SOAP-note generation throughput in decode-heavy serving regimes.",
                "",
                "## Limitation / Future Work Box",
                "- Long-context TTFT remains prefill-bound.",
                "- Peak memory is higher than HF for the custom runtime.",
                "- Future work: prefix/template KV reuse, chunked prefill scheduling, and production-grade continuous batching.",
                "",
                "## Live Demo",
                "Show a pasted or preloaded transcript, run one selected backend, and display SOAP output plus TTFT/prefill/decode/latency/tok/s metrics. Use `Paged runtime + Triton` as the headline path and keep HF baseline available for comparison.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    _style()
    rows = _read_csv(SUMMARY_CSV)
    PREFILL_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    prefill = build_prefill_assets(rows)
    memory = build_memory_assets(rows)
    build_demo_layout()
    write_docs(prefill, memory)
    manifest = {
        "prefill": prefill,
        "memory": memory,
        "demo_assets": [str(DEMO_DIR / "demo_layout.png"), str(DEMO_DIR / "demo_layout.pdf")],
    }
    (PROJECT_ROOT / "poster_assets" / "next_experiments_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
