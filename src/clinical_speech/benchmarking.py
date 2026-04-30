from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any


def is_numeric_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def numeric_summary(values: list[float | int]) -> dict[str, float | int]:
    if not values:
        return {}
    normalized = [float(value) for value in values]
    summary: dict[str, float | int] = {
        "count": len(normalized),
        "mean": statistics.mean(normalized),
        "min": min(normalized),
        "max": max(normalized),
    }
    summary["std"] = statistics.stdev(normalized) if len(normalized) > 1 else 0.0
    summary["median"] = statistics.median(normalized)
    return summary


def aggregate_numeric_records(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    numeric_keys: set[str] = set()
    for record in records:
        for key, value in record.items():
            if is_numeric_value(value):
                numeric_keys.add(key)

    aggregated: dict[str, dict[str, float | int]] = {}
    for key in sorted(numeric_keys):
        values = [record[key] for record in records if is_numeric_value(record.get(key))]
        if values:
            aggregated[key] = numeric_summary(values)
    return aggregated


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(target, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_rate(numerator: int | float, denominator: int | float | None) -> float | None:
    if denominator is None:
        return None
    if math.isclose(float(denominator), 0.0):
        return None
    return float(numerator) / float(denominator)
