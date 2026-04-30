from pathlib import Path

import evaluate
from bert_score import score as bertscore_score

from clinical_speech.benchmarking import aggregate_numeric_records
from clinical_speech.storage import get_runtime_paths
from clinical_speech.utils.io import read_jsonl


def aggregate_runtime_metrics(predictions: list[dict]) -> dict:
    runtimes = [row["runtime"] for row in predictions]
    return aggregate_numeric_records(runtimes)


def evaluation_warnings(predictions: list[dict]) -> list[str]:
    warnings: list[str] = []
    if len(predictions) < 10:
        warnings.append(
            f"Evaluation is based on only {len(predictions)} samples. Treat these results as smoke-test or pilot numbers, not scientific evidence."
        )
    smoke_flags = [
        bool((row.get("metadata") or {}).get("smoke_test"))
        for row in predictions
    ]
    if any(smoke_flags):
        warnings.append("Smoke-test dataset detected in prediction metadata. Do not report these metrics as scientific results.")
    return warnings


def maybe_score_references(predictions: list[dict]) -> dict:
    warnings = evaluation_warnings(predictions)
    quality_report = {
        "metrics": {},
        "supported_metrics": [],
        "unsupported_metrics": [],
        "warnings": warnings,
    }

    references = [row.get("reference_note") for row in predictions]
    candidates = [row.get("predicted_note") for row in predictions]
    if not references or any(ref is None for ref in references):
        quality_report["unsupported_metrics"].append(
            {
                "name": "reference_note_metrics",
                "reason": "reference_note is missing for one or more samples.",
            }
        )
        return quality_report

    runtime_paths = get_runtime_paths()
    rouge = evaluate.load("rouge", cache_dir=str(runtime_paths.cache / "evaluate"))
    rouge_scores = {
        key: float(value)
        for key, value in rouge.compute(predictions=candidates, references=references).items()
    }
    quality_report["metrics"]["rouge"] = rouge_scores
    quality_report["supported_metrics"].append("rouge")

    try:
        _p, _r, f1 = bertscore_score(candidates, references, lang="en", verbose=False)
    except Exception as exc:
        quality_report["unsupported_metrics"].append(
            {
                "name": "bertscore_f1_mean",
                "reason": f"BERTScore failed at runtime: {exc}",
            }
        )
    else:
        quality_report["metrics"]["bertscore_f1_mean"] = float(f1.mean().item())
        quality_report["supported_metrics"].append("bertscore_f1_mean")

    reference_transcripts = [row.get("reference_transcript") for row in predictions]
    generated_transcripts = [row.get("generated_transcript") for row in predictions]
    if reference_transcripts and generated_transcripts and all(reference_transcripts) and all(generated_transcripts):
        quality_report["metrics"]["wer"] = compute_wer(reference_transcripts, generated_transcripts)
        quality_report["supported_metrics"].append("wer")
    else:
        quality_report["unsupported_metrics"].append(
            {
                "name": "wer",
                "reason": "WER requires both reference_transcript and generated_transcript for every prediction.",
            }
        )

    return quality_report


def evaluate_predictions_file(path: str | Path) -> dict:
    predictions = read_jsonl(path)
    return {
        "runtime": aggregate_runtime_metrics(predictions),
        "quality": maybe_score_references(predictions),
    }


def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    from jiwer import wer

    return float(wer(references, hypotheses))
