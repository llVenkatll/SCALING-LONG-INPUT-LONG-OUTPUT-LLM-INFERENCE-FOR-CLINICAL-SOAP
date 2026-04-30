from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clinical_speech.config import BackendSpec, BenchmarkConfig, GenerationConfig, ModelConfig, RuntimeConfig
from clinical_speech.models.hosted_together import TogetherHostedModel
from clinical_speech.models.note_generator import NoteGenerator, _release_cuda_memory
from clinical_speech.pipeline.prompts import SOAP_SYSTEM_PROMPT, build_note_prompt
from clinical_speech.runtime.engine import ManualBatchEngine
from clinical_speech.utils.io import read_jsonl, write_json


OUTPUT_DIR = PROJECT_ROOT / "poster_assets" / "deep_systems_study"


@dataclass(frozen=True)
class ReviewBackend:
    name: str
    kind: str
    model_config: ModelConfig | None = None


def build_strict_note_prompt(transcript: str) -> str:
    strict_system = (
        f"{SOAP_SYSTEM_PROMPT}\n\n"
        "Use only facts explicitly present in the transcript.\n"
        "If vitals, labs, medications, or exam findings are missing, write unknown.\n"
        "Do not invent values, diagnoses, or treatments."
    )
    return f"{strict_system}\n\nClinical conversation:\n{transcript}\n\nSOAP note:"


def _contains(patterns: list[str], text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in patterns)


def _flags(transcript: str, output: str) -> dict[str, bool]:
    transcript_lower = transcript.lower()
    output_lower = output.lower()
    transcript_has_vitals = _contains(["blood pressure", "bp ", "temperature", "pulse", "respiratory rate", "oxygen saturation"], transcript_lower)
    transcript_has_labs = _contains(["cbc", "wbc", "blood culture", "lab", "labs", "glucose"], transcript_lower)
    transcript_has_meds = _contains(["medication", "acetaminophen", "ibuprofen", "antibiotic", "insulin"], transcript_lower)
    transcript_has_demographics = _contains(
        ["year-old", "male", "female", "asian", "black", "white", "hispanic", "latino"],
        transcript_lower,
    )
    return {
        "possible_invented_vitals": _contains(["blood pressure", "bp ", "temperature", "pulse", "respiratory rate", "oxygen saturation"], output_lower) and not transcript_has_vitals,
        "possible_invented_labs": _contains(["cbc", "wbc", "blood culture", "lab", "labs", "glucose"], output_lower) and not transcript_has_labs,
        "possible_invented_meds": _contains(["acetaminophen", "ibuprofen", "antibiotic", "insulin", "medication"], output_lower) and not transcript_has_meds,
        "possible_invented_demographics": _contains(
            ["year-old", "male", "female", "asian", "black", "white", "hispanic", "latino"],
            output_lower,
        ) and not transcript_has_demographics,
    }


def _build_backends(mistral_cfg: ModelConfig, llama_cfg: ModelConfig) -> list[ReviewBackend]:
    backends = [
        ReviewBackend(name="hf_mistral_local", kind="note", model_config=mistral_cfg),
        ReviewBackend(name="paged_triton_mistral", kind="paged", model_config=mistral_cfg),
        ReviewBackend(name="hf_llama_local", kind="note", model_config=llama_cfg),
    ]
    if os.environ.get("TOGETHER_API_KEY"):
        backends.append(ReviewBackend(name="together_llama_hosted", kind="hosted", model_config=llama_cfg))
    return backends


def _build_generator(
    backend: ReviewBackend,
    *,
    gen_cfg: GenerationConfig,
    runtime_cfg: RuntimeConfig,
) -> object:
    if backend.kind == "note":
        return NoteGenerator(backend.model_config, gen_cfg)
    if backend.kind == "paged":
        return ManualBatchEngine(backend.model_config, gen_cfg, runtime_cfg)
    if backend.kind == "hosted":
        spec = BackendSpec(
            provider="together_hosted",
            api_model_id="venkat011003_84d6/togethercomputer/meta-llama-3.1-8B-Instruct-AWQ-INT4-69f5dd64",
            llm_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            api_key_env="TOGETHER_API_KEY",
            base_url_env="TOGETHER_BASE_URL",
            stream=False,
        )
        return TogetherHostedModel(spec, generation_config=gen_cfg)
    raise ValueError(f"Unknown review backend kind: {backend.kind}")


def _run_backend(
    backend: ReviewBackend,
    *,
    dataset: list[dict[str, object]],
    prompt_modes: dict[str, object],
    benchmark_cfg: BenchmarkConfig,
    gen_cfg: GenerationConfig,
    runtime_cfg: RuntimeConfig,
) -> list[dict[str, object]]:
    generator = _build_generator(backend, gen_cfg=gen_cfg, runtime_cfg=runtime_cfg)
    rows: list[dict[str, object]] = []
    try:
        for sample_index, sample in enumerate(dataset):
            transcript = str(sample.get("transcript", ""))
            for prompt_mode, prompt_builder in prompt_modes.items():
                prompt = prompt_builder(transcript)
                if isinstance(generator, ManualBatchEngine):
                    result = generator.run_batch(
                        request_ids=[f"{backend.name}_{prompt_mode}_{sample_index}"],
                        prompts=[prompt],
                    ).requests[0].text
                elif isinstance(generator, TogetherHostedModel):
                    result = generator.generate(prompt)
                else:
                    result = generator.generate(prompt, benchmark_cfg).text
                rows.append(
                    {
                        "sample_index": sample_index,
                        "backend": backend.name,
                        "prompt_mode": prompt_mode,
                        "transcript_excerpt": transcript[:300].replace("\n", " "),
                        "output": result,
                        **_flags(transcript, result),
                    }
                )
    finally:
        del generator
        _release_cuda_memory()
    return rows


def main() -> None:
    dataset = read_jsonl("/data/project_runtime/datasets/medsynth/test.jsonl")[:5]
    benchmark_cfg = BenchmarkConfig(enabled=False)
    gen_cfg = GenerationConfig(max_new_tokens=64, temperature=0.0, top_p=1.0, do_sample=False)

    mistral_cfg = ModelConfig(
        asr_model_id="openai/whisper-small",
        llm_model_id="mistralai/Mistral-7B-Instruct-v0.3",
        device="cuda",
        dtype="float16",
        load_in_8bit=False,
        attn_implementation="sdpa",
    )
    llama_cfg = ModelConfig(
        asr_model_id="openai/whisper-small",
        llm_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        device="cuda",
        dtype="float16",
        load_in_8bit=False,
        attn_implementation="sdpa",
    )
    runtime_cfg = RuntimeConfig(
        backend="mistral_paged",
        scheduler_mode="none",
        max_batch_size=1,
        max_concurrent_requests=1,
        max_cache_budget_gb=4.0,
        triton_enabled=True,
        triton_paged_kv_enabled=True,
    )

    backends = _build_backends(mistral_cfg, llama_cfg)

    prompt_modes = {
        "default": build_note_prompt,
        "strict": build_strict_note_prompt,
    }

    rows: list[dict[str, object]] = []
    for backend in backends:
        rows.extend(
            _run_backend(
                backend,
                dataset=dataset,
                prompt_modes=prompt_modes,
                benchmark_cfg=benchmark_cfg,
                gen_cfg=gen_cfg,
                runtime_cfg=runtime_cfg,
            )
        )

    summary: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        key = (row["backend"], row["prompt_mode"])
        bucket = summary.setdefault(
            key,
            {
                "possible_invented_vitals": 0,
                "possible_invented_labs": 0,
                "possible_invented_meds": 0,
                "possible_invented_demographics": 0,
                "count": 0,
            },
        )
        bucket["count"] += 1
        for field in (
            "possible_invented_vitals",
            "possible_invented_labs",
            "possible_invented_meds",
            "possible_invented_demographics",
        ):
            bucket[field] += int(bool(row[field]))

    serializable_summary = [
        {
            "backend": backend,
            "prompt_mode": prompt_mode,
            **values,
        }
        for (backend, prompt_mode), values in sorted(summary.items())
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "prompt_control_review.json", {"rows": rows, "summary": serializable_summary})
    lines = [
        "# Prompt Control Review",
        "",
        "Heuristic hallucination flags are only for manual inspection.",
        "",
        "| Backend | Prompt Mode | Count | Invented Vitals | Invented Labs | Invented Meds | Invented Demographics |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in serializable_summary:
        lines.append(
            f"| {row['backend']} | {row['prompt_mode']} | {row['count']} | {row['possible_invented_vitals']} | {row['possible_invented_labs']} | {row['possible_invented_meds']} | {row['possible_invented_demographics']} |"
        )
    lines.append("")
    for row in rows:
        lines.append(f"## Sample {row['sample_index']} | {row['backend']} | {row['prompt_mode']}")
        lines.append(f"Transcript excerpt: {row['transcript_excerpt']}")
        lines.append(f"Output: {row['output'][:1200]}")
        lines.append("")
    (OUTPUT_DIR / "prompt_control_review.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": serializable_summary}, indent=2))


if __name__ == "__main__":
    main()
