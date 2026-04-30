from pathlib import Path

from clinical_speech.data.schema import ClinicalSample
from clinical_speech.utils.io import read_jsonl


def load_dataset(path: str | Path, max_samples: int | None = None) -> list[ClinicalSample]:
    rows = read_jsonl(path)
    if max_samples is not None:
        rows = rows[:max_samples]
    return [ClinicalSample.model_validate(row) for row in rows]
