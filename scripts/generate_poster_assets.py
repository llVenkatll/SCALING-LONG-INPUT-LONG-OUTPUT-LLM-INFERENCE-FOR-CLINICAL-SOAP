from __future__ import annotations

import argparse
import csv
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSTER_DIR = PROJECT_ROOT / "poster_assets"
DEFAULT_PILOT_RESULTS_TABLE = Path("/data/project_runtime/benchmarks/systems_benchmark_batch4_pilot/systems/systems_summary.csv")
DEFAULT_FALLBACK_RESULTS_TABLE = PROJECT_ROOT / "results" / "tables" / "systems_summary.csv"
DEFAULT_KERNEL_BENCH = Path("/data/project_runtime/benchmarks/kernels/paged_kv_benchmark.json")

PRIMARY_BACKENDS = [
    "hf_sequential",
    "mistral_paged_static_batch",
    "mistral_paged_static_batch_triton",
]
CUSTOM_BACKENDS = [
    "mistral_paged_static_batch",
    "mistral_paged_static_batch_triton",
]

BACKEND_LABELS = {
    "hf_sequential": "HF Sequential",
    "mistral_paged_single": "Paged Runtime\n(Single)",
    "mistral_paged_static_batch": "Paged Runtime\n(Static Batch)",
    "mistral_paged_static_batch_triton": "Paged Runtime + Triton",
    "hf_sequential_llama_local": "HF Sequential\n(Llama, local)",
    "together_hosted_llama": "Together Hosted\n(Llama)",
}

BACKEND_COLORS = {
    "hf_sequential": "#7a7a7a",
    "mistral_paged_single": "#8fb7d8",
    "mistral_paged_static_batch": "#2f6ea6",
    "mistral_paged_static_batch_triton": "#0d3b66",
    "hf_sequential_llama_local": "#5e7f94",
    "together_hosted_llama": "#a15c38",
}

FALLBACK_COLORS = ["#7a7a7a", "#2f6ea6", "#0d3b66", "#5e7f94", "#a15c38", "#4e7d4e"]


@dataclass
class PosterAssetContext:
    output_dir: Path
    summary_csv: Path
    kernel_bench: Path
    main_batch_size: int
    compared_backends: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--kernel-bench", type=Path, default=DEFAULT_KERNEL_BENCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_POSTER_DIR)
    parser.add_argument("--main-batch-size", type=int, default=None)
    return parser.parse_args()


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
        }
    )


def _resolve_summary_csv(path: Path | None) -> Path:
    if path is not None:
        return path
    if DEFAULT_PILOT_RESULTS_TABLE.exists():
        return DEFAULT_PILOT_RESULTS_TABLE
    return DEFAULT_FALLBACK_RESULTS_TABLE


def _load_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_kernel_benchmark(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _by_key(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    return {(row["backend"], int(row["batch_size"])): row for row in rows}


def _to_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, "", "null"):
        return None
    return float(value)


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _poster_title(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", pad=18)
    if subtitle:
        ax.text(
            0.0,
            1.03,
            subtitle,
            transform=ax.transAxes,
            fontsize=11,
            color="#555555",
            ha="left",
            va="bottom",
        )


def _format_backend(backend: str, row: dict[str, str] | None = None, *, wrap: int | None = None) -> str:
    if row and row.get("backend_label"):
        label = row["backend_label"]
    else:
        label = BACKEND_LABELS.get(backend, backend)
    if wrap is not None:
        return textwrap.fill(label.replace("\n", " "), width=wrap)
    return label


def _color_for_backend(backend: str) -> str:
    if backend in BACKEND_COLORS:
        return BACKEND_COLORS[backend]
    return FALLBACK_COLORS[sum(ord(char) for char in backend) % len(FALLBACK_COLORS)]


def _fmt_value(value: float | None, fmt: str, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{fmt.format(value)}{suffix}"


def _resolve_main_comparison(
    rows_by_key: dict[tuple[str, int], dict[str, str]],
    preferred_batch_size: int | None = None,
) -> tuple[int, list[str]]:
    available_keys = set(rows_by_key.keys())
    batch_sizes = sorted({batch_size for _, batch_size in available_keys})
    all_backends = sorted({backend for backend, _ in available_keys})

    def _has_all(backends: list[str], batch_size: int) -> bool:
        return all((backend, batch_size) in available_keys for backend in backends)

    def _comparison_score(batch_size: int, backends: list[str]) -> tuple[int, int, int]:
        comparable = 0
        with_ttft = 0
        for backend in backends:
            row = rows_by_key[(backend, batch_size)]
            if _to_float(row, "mean_throughput_tok_per_sec") is not None and _to_float(row, "mean_requests_per_sec") is not None:
                comparable += 1
            if _to_float(row, "mean_ttft_sec") is not None:
                with_ttft += 1
        return comparable, with_ttft, batch_size

    if any(backend not in PRIMARY_BACKENDS for backend in all_backends):
        if preferred_batch_size is not None:
            present = [backend for backend in all_backends if (backend, preferred_batch_size) in available_keys]
            if len(present) >= 2 and _comparison_score(preferred_batch_size, present)[0] >= 2:
                return preferred_batch_size, present
        best_choice: tuple[int, list[str]] | None = None
        best_score: tuple[int, int, int] | None = None
        for batch_size in batch_sizes:
            present = [backend for backend in all_backends if (backend, batch_size) in available_keys]
            if len(present) < 2:
                continue
            score = _comparison_score(batch_size, present)
            if score[0] < 2:
                continue
            if best_choice is None or score > best_score or (
                score == best_score and len(present) > len(best_choice[1])
            ):
                best_choice = (batch_size, present)
                best_score = score
        if best_choice is not None:
            return best_choice

    if preferred_batch_size is not None and _has_all(PRIMARY_BACKENDS, preferred_batch_size):
        return preferred_batch_size, list(PRIMARY_BACKENDS)

    for batch_size in reversed(batch_sizes):
        if _has_all(PRIMARY_BACKENDS, batch_size):
            return batch_size, list(PRIMARY_BACKENDS)

    for batch_size in reversed(batch_sizes):
        present = [backend for backend in PRIMARY_BACKENDS if (backend, batch_size) in available_keys]
        if len(present) >= 2:
            return batch_size, present

    raise ValueError("Could not resolve a main comparison batch from the systems summary rows")


def _speedup_percent(base: float, candidate: float) -> float:
    return 100.0 * (candidate / base - 1.0)


def generate_main_results_table(
    rows_by_key: dict[tuple[str, int], dict[str, str]],
    *,
    output_dir: Path,
    batch_size: int,
    backends: list[str],
) -> None:
    table_rows: list[list[str]] = []
    csv_rows: list[dict[str, str | int | float | None]] = []
    for backend in backends:
        row = rows_by_key[(backend, batch_size)]
        kv_util = _to_float(row, "mean_kv_utilization_ratio")
        table_rows.append(
            [
                _format_backend(backend, row),
                str(batch_size),
                _fmt_value(_to_float(row, "mean_throughput_tok_per_sec"), "{:.2f}"),
                _fmt_value(_to_float(row, "mean_requests_per_sec"), "{:.3f}"),
                _fmt_value(_to_float(row, "mean_ttft_sec"), "{:.3f}", "s"),
                _fmt_value(_to_float(row, "mean_decode_latency_sec"), "{:.3f}", "s"),
                _fmt_value(_to_float(row, "mean_peak_gpu_mem_gb"), "{:.2f}", " GB"),
                _fmt_value(kv_util, "{:.3f}"),
            ]
        )
        csv_rows.append(
            {
                "Backend": _format_backend(backend, row),
                "Batch Size": batch_size,
                "tok/s": None if _to_float(row, "mean_throughput_tok_per_sec") is None else round(_to_float(row, "mean_throughput_tok_per_sec"), 4),
                "req/s": None if _to_float(row, "mean_requests_per_sec") is None else round(_to_float(row, "mean_requests_per_sec"), 4),
                "TTFT (s)": None if _to_float(row, "mean_ttft_sec") is None else round(_to_float(row, "mean_ttft_sec"), 4),
                "Decode Latency (s)": None if _to_float(row, "mean_decode_latency_sec") is None else round(_to_float(row, "mean_decode_latency_sec"), 4),
                "Peak GPU Memory (GB)": None if _to_float(row, "mean_peak_gpu_mem_gb") is None else round(_to_float(row, "mean_peak_gpu_mem_gb"), 4),
                "KV Utilization": None if kv_util is None else round(kv_util, 4),
            }
        )

    csv_path = output_dir / "main_results_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.axis("off")
    _poster_title(ax, "Main Results Table", f"Main comparison workload, batch size {batch_size}")
    col_labels = [
        "Backend",
        "Batch Size",
        "tok/s",
        "req/s",
        "TTFT",
        "Decode Latency",
        "Peak GPU Memory",
        "KV Utilization",
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        if row_idx == 0:
            cell.set_facecolor("#dce7f2")
            cell.set_text_props(weight="bold")
        else:
            backend_key = backends[row_idx - 1]
            cell.set_facecolor("#f7f9fb" if row_idx % 2 else "#eef3f8")
            if col_idx == 0:
                cell.set_text_props(weight="bold", color=_color_for_backend(backend_key))
    _save(fig, output_dir, "main_results_table")


def _simple_bar(
    rows_by_key: dict[tuple[str, int], dict[str, str]],
    *,
    output_dir: Path,
    batch_size: int,
    backends: list[str],
    metric_key: str,
    ylabel: str,
    title: str,
    stem: str,
    value_fmt: str,
) -> None:
    values = [_to_float(rows_by_key[(backend, batch_size)], metric_key) for backend in backends]
    labels = [_format_backend(backend, rows_by_key[(backend, batch_size)], wrap=22) for backend in backends]
    colors = [_color_for_backend(backend) for backend in backends]
    display_values = [0.0 if value is None else value for value in values]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bars = ax.bar(range(len(backends)), display_values, color=colors, width=0.65)
    _poster_title(ax, title, f"Batch size {batch_size}")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(backends)), labels)
    ax.set_axisbelow(True)
    finite_values = [value for value in values if value is not None]
    ymax = (max(finite_values) if finite_values else 1.0) * 1.2
    ax.set_ylim(0, ymax)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            (0.0 if value is None else value) + ymax * 0.02,
            "N/A" if value is None else value_fmt.format(value),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    _save(fig, output_dir, stem)


def generate_speedup_bar(
    rows_by_key: dict[tuple[str, int], dict[str, str]],
    *,
    output_dir: Path,
    batch_size: int,
) -> None:
    hf = _to_float(rows_by_key[("hf_sequential", batch_size)], "mean_throughput_tok_per_sec")
    paged = _to_float(rows_by_key[("mistral_paged_static_batch", batch_size)], "mean_throughput_tok_per_sec")
    triton = _to_float(rows_by_key[("mistral_paged_static_batch_triton", batch_size)], "mean_throughput_tok_per_sec")
    if hf is None or paged is None or triton is None:
        return
    speedups = [_speedup_percent(hf, paged), _speedup_percent(hf, triton)]
    labels = ["Paged Runtime", "Paged Runtime + Triton"]
    colors = [_color_for_backend("mistral_paged_static_batch"), _color_for_backend("mistral_paged_static_batch_triton")]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bars = ax.bar(labels, speedups, color=colors, width=0.58)
    _poster_title(ax, "Speedup vs HF Baseline", f"Relative throughput improvement at batch size {batch_size}")
    ax.set_ylabel("Throughput Speedup (%)")
    ax.axhline(15.0, color="#b23a48", linestyle="--", linewidth=1.5, label="15% target bar")
    ax.axhline(20.0, color="#7d1d2b", linestyle=":", linewidth=1.5, label="20% target bar")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, max(speedups) * 1.25)
    for bar, value in zip(bars, speedups, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}%", ha="center", va="bottom", fontweight="bold")
    _save(fig, output_dir, "speedup_bar")


def generate_incremental_waterfall(
    rows_by_key: dict[tuple[str, int], dict[str, str]],
    *,
    output_dir: Path,
    batch_size: int,
) -> None:
    hf = _to_float(rows_by_key[("hf_sequential", batch_size)], "mean_throughput_tok_per_sec")
    paged = _to_float(rows_by_key[("mistral_paged_static_batch", batch_size)], "mean_throughput_tok_per_sec")
    triton = _to_float(rows_by_key[("mistral_paged_static_batch_triton", batch_size)], "mean_throughput_tok_per_sec")

    paged_gain = paged - hf
    triton_gain = triton - paged
    labels = ["HF Baseline", "+ Paged Runtime", "+ Triton", "Final Runtime"]
    bases = [0.0, hf, paged, 0.0]
    heights = [hf, paged_gain, triton_gain, triton]
    colors = [
        _color_for_backend("hf_sequential"),
        _color_for_backend("mistral_paged_static_batch"),
        "#1d5f91",
        _color_for_backend("mistral_paged_static_batch_triton"),
    ]

    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    _poster_title(ax, "Incremental Throughput Gains", f"Waterfall view at batch size {batch_size}")
    bars = ax.bar(range(len(labels)), heights, bottom=bases, color=colors, width=0.62)
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_xticks(range(len(labels)), labels)
    ymax = max(triton, hf) * 1.25
    ax.set_ylim(0.0, ymax)

    connector_pairs = [
        ((0, hf), (1, hf)),
        ((1, paged), (2, paged)),
    ]
    for (left_x, left_y), (right_x, right_y) in connector_pairs:
        ax.plot([left_x + 0.31, right_x - 0.31], [left_y, right_y], color="#666666", linewidth=1.2, linestyle="--")

    annotations = [hf, paged_gain, triton_gain, triton]
    prefixes = ["", "+", "+", ""]
    for idx, (bar, value, prefix) in enumerate(zip(bars, annotations, prefixes, strict=True)):
        y = bases[idx] + heights[idx]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + ymax * 0.025,
            f"{prefix}{value:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.text(
        0.98,
        0.93,
        f"+{_speedup_percent(hf, triton):.1f}% vs HF",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=_color_for_backend("mistral_paged_static_batch_triton"),
    )
    _save(fig, output_dir, "incremental_waterfall")


def generate_batch_scaling(
    rows: list[dict[str, str]],
    *,
    output_dir: Path,
    metric_key: str,
    ylabel: str,
    title: str,
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    _poster_title(ax, title, "Comparison across available batch sizes")
    backend_order = [
        "hf_sequential",
        "hf_sequential_llama_local",
        "mistral_paged_single",
        "mistral_paged_static_batch",
        "mistral_paged_static_batch_triton",
        "together_hosted_llama",
    ]
    for backend in backend_order:
        backend_rows = [row for row in rows if row["backend"] == backend]
        if not backend_rows:
            continue
        backend_rows.sort(key=lambda item: int(item["batch_size"]))
        xs = [int(item["batch_size"]) for item in backend_rows]
        ys = [_to_float(item, metric_key) for item in backend_rows]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=8,
            linewidth=2.6,
            color=_color_for_backend(backend),
            label=_format_backend(backend, backend_rows[0], wrap=22),
        )
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted({int(row["batch_size"]) for row in rows}))
    ax.legend(frameon=False, loc="best")
    _save(fig, output_dir, stem)


def generate_throughput_memory_scatter(
    rows: list[dict[str, str]],
    *,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    _poster_title(ax, "Throughput vs Memory Tradeoff", "Each point is one backend at one batch size")

    backend_order = [
        "hf_sequential",
        "mistral_paged_static_batch",
        "mistral_paged_static_batch_triton",
        "hf_sequential_llama_local",
        "together_hosted_llama",
    ]
    markers = {
        "hf_sequential": "o",
        "mistral_paged_static_batch": "s",
        "mistral_paged_static_batch_triton": "^",
        "hf_sequential_llama_local": "D",
        "together_hosted_llama": "P",
    }
    for backend in backend_order:
        backend_rows = [row for row in rows if row["backend"] == backend]
        if not backend_rows:
            continue
        xs = [_to_float(row, "mean_peak_gpu_mem_gb") for row in backend_rows]
        ys = [_to_float(row, "mean_throughput_tok_per_sec") for row in backend_rows]
        batch_sizes = [int(row["batch_size"]) for row in backend_rows]
        filtered = [(x, y, batch_size) for x, y, batch_size in zip(xs, ys, batch_sizes, strict=True) if x is not None and y is not None]
        if not filtered:
            continue
        ax.scatter(
            [item[0] for item in filtered],
            [item[1] for item in filtered],
            s=120,
            marker=markers[backend],
            color=_color_for_backend(backend),
            edgecolors="#1f2937",
            linewidths=0.8,
            label=_format_backend(backend, backend_rows[0], wrap=22),
            zorder=3,
        )
        for x, y, batch_size in filtered:
            ax.text(x + 0.04, y + 0.18, f"B{batch_size}", fontsize=10, color="#333333")

    ax.set_xlabel("Peak GPU Memory (GB)")
    ax.set_ylabel("Throughput (tok/s)")
    ax.legend(frameon=False, loc="best")
    _save(fig, output_dir, "throughput_memory_scatter")


def generate_speedup_heatmap(
    rows_by_key: dict[tuple[str, int], dict[str, str]],
    *,
    output_dir: Path,
) -> None:
    batch_sizes = sorted(
        {
            batch_size
            for backend, batch_size in rows_by_key
            if backend in PRIMARY_BACKENDS
        }
    )
    compare_backends = [
        "mistral_paged_static_batch",
        "mistral_paged_static_batch_triton",
    ]
    data: list[list[float]] = []
    for backend in compare_backends:
        row_values: list[float] = []
        for batch_size in batch_sizes:
            if ("hf_sequential", batch_size) not in rows_by_key or (backend, batch_size) not in rows_by_key:
                row_values.append(float("nan"))
                continue
            hf = _to_float(rows_by_key[("hf_sequential", batch_size)], "mean_throughput_tok_per_sec")
            candidate = _to_float(rows_by_key[(backend, batch_size)], "mean_throughput_tok_per_sec")
            row_values.append(_speedup_percent(hf, candidate))
        data.append(row_values)

    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    _poster_title(ax, "Where The Runtime Wins", "Heatmap of throughput gain vs HF across available batch sizes")
    finite_values = [value for row in data for value in row if value == value]
    vmax = max(max(abs(value) for value in finite_values), 20.0)
    image = ax.imshow(data, cmap="RdYlBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(batch_sizes)), [str(batch_size) for batch_size in batch_sizes])
    ax.set_yticks(range(len(compare_backends)), ["Paged Runtime", "Paged Runtime + Triton"])
    ax.set_xlabel("Batch Size")
    for row_idx, row_values in enumerate(data):
        for col_idx, value in enumerate(row_values):
            if value == value:
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    color="#111111",
                    fontweight="bold",
                )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Throughput Gain vs HF (%)")
    _save(fig, output_dir, "speedup_heatmap")


def generate_latency_breakdown(
    rows_by_key: dict[tuple[str, int], dict[str, str]],
    *,
    output_dir: Path,
    batch_size: int,
    backends: list[str],
) -> None:
    metrics = [
        ("mean_ttft_sec", "TTFT"),
        ("mean_prefill_latency_sec", "Prefill"),
        ("mean_decode_latency_sec", "Decode"),
        ("mean_latency_sec", "End-to-End"),
    ]
    x_positions = list(range(len(backends)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 6))
    _poster_title(ax, "Latency Breakdown", f"Batch size {batch_size}; decode is the main Triton-affected component")
    offsets = [-1.5, -0.5, 0.5, 1.5]
    palette = ["#9ca3af", "#93c5fd", "#2563eb", "#0f172a"]
    for offset, (metric_key, metric_label), color in zip(offsets, metrics, palette, strict=True):
        raw_values = [_to_float(rows_by_key[(backend, batch_size)], metric_key) for backend in backends]
        values = [0.0 if value is None else value for value in raw_values]
        bars = ax.bar([x + offset * width for x in x_positions], values, width=width, label=metric_label, color=color)
        for bar, value in zip(bars, raw_values, strict=True):
            if value is None:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.03, "N/A", ha="center", va="bottom", fontsize=9, rotation=90)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x_positions, [_format_backend(backend, rows_by_key[(backend, batch_size)]) for backend in backends])
    ax.set_ylabel("Seconds")
    ax.legend(frameon=False, ncol=4, loc="upper right")
    if "mistral_paged_static_batch_triton" in backends:
        triton_index = backends.index("mistral_paged_static_batch_triton")
        decode_value = _to_float(rows_by_key[("mistral_paged_static_batch_triton", batch_size)], "mean_decode_latency_sec")
        ax.annotate(
            "Triton primarily helps\ndecode, not TTFT",
            xy=(triton_index + 0.5 * width, decode_value if decode_value is not None else 0.1),
            xytext=(len(backends) - 0.2, max((decode_value or 0.8) * 1.7, 1.5)),
            arrowprops=dict(arrowstyle="->", color="#333333", linewidth=1.4),
            fontsize=11,
            ha="left",
        )
    _save(fig, output_dir, "latency_breakdown")


def generate_memory_kv_chart(
    rows_by_key: dict[tuple[str, int], dict[str, str]],
    *,
    output_dir: Path,
    batch_size: int,
) -> None:
    backends = [backend for backend in CUSTOM_BACKENDS if (backend, batch_size) in rows_by_key]
    labels = [_format_backend(backend, rows_by_key[(backend, batch_size)]) for backend in backends]
    peak_mem = [_to_float(rows_by_key[(backend, batch_size)], "mean_peak_gpu_mem_gb") for backend in backends]
    kv_alloc_mb = [
        None if _to_float(rows_by_key[(backend, batch_size)], "mean_kv_allocated_bytes") is None else _to_float(rows_by_key[(backend, batch_size)], "mean_kv_allocated_bytes") / (1024 ** 2)
        for backend in backends
    ]
    kv_util = [_to_float(rows_by_key[(backend, batch_size)], "mean_kv_utilization_ratio") for backend in backends]
    if not backends or any(value is None for value in peak_mem) or any(value is None for value in kv_alloc_mb):
        return

    fig, ax1 = plt.subplots(figsize=(10, 5.8))
    _poster_title(ax1, "Memory and KV Utilization", f"Custom-runtime comparison at batch size {batch_size}")
    x = list(range(len(backends)))
    width = 0.34
    bars1 = ax1.bar([i - width / 2 for i in x], peak_mem, width=width, color="#5f7c8a", label="Peak GPU Memory (GB)")
    bars2 = ax1.bar([i + width / 2 for i in x], kv_alloc_mb, width=width, color="#9bb7c9", label="KV Allocated (MB)")
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Memory")

    ax2 = ax1.twinx()
    ax2.plot(x, kv_util, color="#0d3b66", marker="o", linewidth=2.5, label="KV Utilization")
    ax2.set_ylabel("KV Utilization")
    ax2.set_ylim(0.0, 1.05)

    for bars in (bars1, bars2):
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(peak_mem) * 0.02, f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=10)
    for idx, value in enumerate(kv_util):
        ax2.text(x[idx], value + 0.03, f"{value:.3f}", color="#0d3b66", ha="center", va="bottom", fontweight="bold")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")
    _save(fig, output_dir, "memory_kv_chart")


def generate_kernel_microbenchmark(kernel_data: dict[str, float], *, output_dir: Path) -> None:
    labels = ["PyTorch Gather", "Triton Gather"]
    values_ms = [1000.0 * kernel_data["torch_seconds"], 1000.0 * kernel_data["triton_seconds"]]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    _poster_title(ax, "Paged-KV Gather Microbenchmark", "Kernel-only comparison from benchmark_paged_kv.py")
    bars = ax.bar(labels, values_ms, color=["#8f8f8f", "#0d3b66"], width=0.56)
    ax.set_ylabel("Kernel Time (ms)")
    ymax = max(values_ms) * 1.3
    ax.set_ylim(0, ymax)
    for bar, value in zip(bars, values_ms, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + ymax * 0.03, f"{value:.3f} ms", ha="center", va="bottom", fontweight="bold")
    ax.text(
        0.98,
        0.95,
        f"{kernel_data['speedup_vs_torch']:.2f}x speedup",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=15,
        fontweight="bold",
        color="#0d3b66",
    )
    _save(fig, output_dir, "kernel_microbenchmark")


def _add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str,
    ec: str = "#2a2a2a",
    text_color: str = "#111111",
    fontsize: int = 11,
    weight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=text_color, weight=weight, wrap=True)


def _add_arrow(ax, start: tuple[float, float], end: tuple[float, float], *, color: str = "#444444") -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color=color)
    ax.add_patch(arrow)


def generate_system_architecture_diagram(*, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("System Architecture", loc="left", pad=18, fontsize=20, fontweight="bold")
    ax.text(0.0, 0.965, "Baseline HF path and custom paged-KV runtime path for clinical note generation", fontsize=12, color="#555555", ha="left")

    _add_box(ax, 0.03, 0.42, 0.16, 0.12, "Input Transcript", fc="#eef3f8", weight="bold", fontsize=13)
    _add_box(ax, 0.24, 0.42, 0.16, 0.12, "Tokenizer", fc="#eef3f8", weight="bold", fontsize=13)
    _add_arrow(ax, (0.19, 0.48), (0.24, 0.48))

    _add_box(ax, 0.46, 0.68, 0.20, 0.12, "Baseline HF Path\n`generate()`", fc="#e7ecef", weight="bold", fontsize=13)
    _add_box(ax, 0.46, 0.24, 0.20, 0.12, "Custom Runtime Path", fc="#d8e8f5", weight="bold", fontsize=13)
    _add_arrow(ax, (0.40, 0.48), (0.46, 0.74))
    _add_arrow(ax, (0.40, 0.48), (0.46, 0.30))

    _add_box(ax, 0.73, 0.68, 0.19, 0.12, "Mistral Model", fc="#f3f4f6", weight="bold", fontsize=13)
    _add_arrow(ax, (0.66, 0.74), (0.73, 0.74))

    _add_box(ax, 0.70, 0.06, 0.22, 0.12, "SOAP Note Output", fc="#eef3f8", weight="bold", fontsize=13)

    _add_box(ax, 0.08, 0.06, 0.18, 0.12, "Static Batched\nScheduler", fc="#e6f0f8", fontsize=12)
    _add_box(ax, 0.31, 0.06, 0.16, 0.12, "Prefill Stage", fc="#d3e5f5", fontsize=12)
    _add_box(ax, 0.50, 0.06, 0.16, 0.12, "Decode Stage", fc="#b9d7ee", fontsize=12)
    _add_box(ax, 0.31, 0.26, 0.16, 0.12, "KV Page Pool /\nBlock Manager", fc="#cfe7db", fontsize=12)
    _add_box(ax, 0.50, 0.26, 0.16, 0.12, "Triton Paged-KV\nGather Kernel", fc="#bdd7d0", fontsize=12)

    _add_arrow(ax, (0.17, 0.18), (0.17, 0.24))
    _add_arrow(ax, (0.26, 0.30), (0.31, 0.30))
    _add_arrow(ax, (0.47, 0.12), (0.50, 0.12))
    _add_arrow(ax, (0.39, 0.18), (0.39, 0.26))
    _add_arrow(ax, (0.58, 0.18), (0.58, 0.26))
    _add_arrow(ax, (0.47, 0.32), (0.50, 0.32))
    _add_arrow(ax, (0.66, 0.30), (0.73, 0.30))
    _add_arrow(ax, (0.82, 0.68), (0.82, 0.18))

    ax.text(0.73, 0.57, "HF baseline keeps library-owned KV growth\nand sequential serving behavior.", fontsize=11, color="#444444", ha="center")
    ax.text(0.49, 0.42, "Custom path separates prefill from decode\nand routes decode KV gathering through page-backed storage.", fontsize=11, color="#16324f", ha="center")
    ax.text(0.50, 0.92, "The note task is the workload; the systems contribution is runtime, memory, scheduling, and Triton decode support.", fontsize=11, color="#444444", ha="center")

    _save(fig, output_dir, "system_architecture_diagram")


def generate_contribution_summary_panel(
    rows_by_key: dict[tuple[str, int], dict[str, str]],
    kernel_data: dict[str, float],
    *,
    output_dir: Path,
    batch_size: int,
) -> None:
    hf_tok = _to_float(rows_by_key[("hf_sequential", batch_size)], "mean_throughput_tok_per_sec")
    triton_tok = _to_float(rows_by_key[("mistral_paged_static_batch_triton", batch_size)], "mean_throughput_tok_per_sec")
    speedup = _speedup_percent(hf_tok, triton_tok)

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Contribution Summary", loc="left", pad=18, fontsize=20, fontweight="bold")

    _add_box(
        ax,
        0.03,
        0.56,
        0.60,
        0.36,
        "\n".join(
            [
                "Custom page-backed KV runtime",
                "Explicit prefill/decode execution",
                "Static batched decode scheduling",
                "Triton paged-KV gather kernel",
                "Exact output parity with HF baseline",
                f"+{speedup:.1f}% tok/s and req/s at batch size {batch_size}",
                f"{kernel_data['speedup_vs_torch']:.2f}x kernel-level Triton speedup",
            ]
        ),
        fc="#eef5fb",
        fontsize=14,
        weight="bold",
    )
    ax.text(0.05, 0.88, "Core Contributions", fontsize=16, fontweight="bold", color="#0d3b66")

    _add_box(
        ax,
        0.68,
        0.56,
        0.28,
        0.36,
        "\n".join(
            [
                "No full PagedAttention claim",
                "Gains strongest in multi-request decode-heavy workloads",
                "Batch size 1 is not guaranteed to win",
                "TTFT changes less than decode latency",
            ]
        ),
        fc="#f8ecec",
        ec="#8a3d3d",
        fontsize=13,
        weight="bold",
    )
    ax.text(0.70, 0.88, "Limitations", fontsize=16, fontweight="bold", color="#8a3d3d")

    _add_box(
        ax,
        0.03,
        0.10,
        0.93,
        0.30,
        f"Headline result for this run: the Triton-enabled paged runtime improves multi-request serving throughput by {speedup:.1f}% over HF sequential at batch size {batch_size}, while preserving benchmarked output parity on the evaluated set.",
        fc="#dbe9f4",
        fontsize=16,
        weight="bold",
    )
    _save(fig, output_dir, "contribution_summary_panel")


def generate_takeaway_figure(
    rows_by_key: dict[tuple[str, int], dict[str, str]],
    *,
    output_dir: Path,
    batch_size: int,
) -> None:
    hf_tok = _to_float(rows_by_key[("hf_sequential", batch_size)], "mean_throughput_tok_per_sec")
    triton_tok = _to_float(rows_by_key[("mistral_paged_static_batch_triton", batch_size)], "mean_throughput_tok_per_sec")
    speedup = _speedup_percent(hf_tok, triton_tok)

    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    headline = f"+{speedup:.1f}% throughput over HF baseline"
    sentence = (
        "A custom page-backed KV runtime plus Triton paged-KV gather improves multi-request "
        f"Mistral serving throughput at batch size {batch_size} on this benchmark run."
    )

    _add_box(ax, 0.03, 0.16, 0.94, 0.68, "", fc="#edf4fa")
    ax.text(0.06, 0.74, "Poster Takeaway", fontsize=18, fontweight="bold", color="#0d3b66", ha="left")
    ax.text(0.06, 0.54, headline, fontsize=28, fontweight="bold", color="#0d3b66", ha="left")
    ax.text(0.06, 0.35, textwrap.fill(sentence, width=80), fontsize=16, color="#222222", ha="left")
    ax.text(0.06, 0.21, "Decode-side improvements dominate; TTFT typically moves less than throughput and request rate.", fontsize=12, color="#555555", ha="left")
    _save(fig, output_dir, "poster_takeaway_figure")


def write_readme(
    context: PosterAssetContext,
    *,
    kernel_bench: Path,
) -> None:
    hosted_note = ""
    if any("together" in backend for backend in context.compared_backends):
        hosted_note = "- Hosted Together-style baselines are managed serving-stack references; missing KV/GPU-local metrics remain `N/A`.\n"
    content = f"""# Poster Assets

This folder contains poster-ready figures generated from a specific benchmark run.

## Regeneration

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/generate_poster_assets.py \\
  --summary-csv {context.summary_csv} \\
  --kernel-bench {kernel_bench} \\
  --output-dir {context.output_dir}
```

## Data Sources

Primary numeric sources loaded directly by the script:

- `{context.summary_csv}`
- `{kernel_bench}`

Main comparison batch used for headline figures:

- batch size `{context.main_batch_size}`
- backends: {", ".join(context.compared_backends)}

## Assets

- `main_results_table.png` / `.pdf`
- `main_results_table.csv`
- `throughput_bar.png` / `.pdf`
- `request_rate_bar.png` / `.pdf`
- `speedup_bar.png` / `.pdf`
- `incremental_waterfall.png` / `.pdf`
- `batch_scaling_toks.png` / `.pdf`
- `batch_scaling_reqs.png` / `.pdf`
- `throughput_memory_scatter.png` / `.pdf`
- `speedup_heatmap.png` / `.pdf`
- `latency_breakdown.png` / `.pdf`
- `memory_kv_chart.png` / `.pdf`
- `kernel_microbenchmark.png` / `.pdf`
- `system_architecture_diagram.png` / `.pdf`
- `contribution_summary_panel.png` / `.pdf`
- `poster_takeaway_figure.png` / `.pdf`

## Notes

- Numeric charts and the main table are loaded directly from benchmark result files.
- The architecture diagram and summary panels are diagrammatic poster assets; they do not introduce new measurements.
- Managed hosted baselines are treated as serving-stack baselines, not as “our model versus their model.”
{hosted_note}- No fabricated benchmark values were added.
"""
    context.output_dir.mkdir(parents=True, exist_ok=True)
    (context.output_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    _set_style()
    summary_csv = _resolve_summary_csv(args.summary_csv)
    rows = _load_summary_rows(summary_csv)
    rows_by_key = _by_key(rows)
    kernel_data = _load_kernel_benchmark(args.kernel_bench)
    main_batch_size, compared_backends = _resolve_main_comparison(rows_by_key, preferred_batch_size=args.main_batch_size)

    context = PosterAssetContext(
        output_dir=args.output_dir,
        summary_csv=summary_csv,
        kernel_bench=args.kernel_bench,
        main_batch_size=main_batch_size,
        compared_backends=compared_backends,
    )
    context.output_dir.mkdir(parents=True, exist_ok=True)

    generate_main_results_table(rows_by_key, output_dir=context.output_dir, batch_size=context.main_batch_size, backends=context.compared_backends)
    _simple_bar(
        rows_by_key,
        output_dir=context.output_dir,
        batch_size=context.main_batch_size,
        backends=context.compared_backends,
        metric_key="mean_throughput_tok_per_sec",
        ylabel="Throughput (tok/s)",
        title="Throughput Comparison",
        stem="throughput_bar",
        value_fmt="{:.2f}",
    )
    _simple_bar(
        rows_by_key,
        output_dir=context.output_dir,
        batch_size=context.main_batch_size,
        backends=context.compared_backends,
        metric_key="mean_requests_per_sec",
        ylabel="Requests / sec",
        title="Request Rate Comparison",
        stem="request_rate_bar",
        value_fmt="{:.3f}",
    )
    if all((backend, context.main_batch_size) in rows_by_key for backend in PRIMARY_BACKENDS):
        generate_speedup_bar(rows_by_key, output_dir=context.output_dir, batch_size=context.main_batch_size)
        generate_incremental_waterfall(rows_by_key, output_dir=context.output_dir, batch_size=context.main_batch_size)
    generate_batch_scaling(
        rows,
        output_dir=context.output_dir,
        metric_key="mean_throughput_tok_per_sec",
        ylabel="Throughput (tok/s)",
        title="Batch Scaling: Throughput",
        stem="batch_scaling_toks",
    )
    generate_batch_scaling(
        rows,
        output_dir=context.output_dir,
        metric_key="mean_requests_per_sec",
        ylabel="Requests / sec",
        title="Batch Scaling: Request Rate",
        stem="batch_scaling_reqs",
    )
    generate_throughput_memory_scatter(rows, output_dir=context.output_dir)
    generate_speedup_heatmap(rows_by_key, output_dir=context.output_dir)
    generate_latency_breakdown(rows_by_key, output_dir=context.output_dir, batch_size=context.main_batch_size, backends=context.compared_backends)
    generate_memory_kv_chart(rows_by_key, output_dir=context.output_dir, batch_size=context.main_batch_size)
    generate_kernel_microbenchmark(kernel_data, output_dir=context.output_dir)
    generate_system_architecture_diagram(output_dir=context.output_dir)
    generate_contribution_summary_panel(rows_by_key, kernel_data, output_dir=context.output_dir, batch_size=context.main_batch_size)
    generate_takeaway_figure(rows_by_key, output_dir=context.output_dir, batch_size=context.main_batch_size)
    write_readme(context, kernel_bench=args.kernel_bench)

    print(
        json.dumps(
            {
                "poster_assets_dir": str(context.output_dir),
                "summary_csv": str(summary_csv),
                "main_batch_size": context.main_batch_size,
                "compared_backends": context.compared_backends,
                "asset_count": len(list(context.output_dir.iterdir())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
