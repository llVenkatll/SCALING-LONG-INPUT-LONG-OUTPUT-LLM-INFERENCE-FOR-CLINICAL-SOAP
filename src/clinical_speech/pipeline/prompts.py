SOAP_SYSTEM_PROMPT = """You are a clinical documentation assistant.
Generate a concise, medically faithful SOAP note from the provided clinical conversation.
Use the following structure:

Subjective:
Objective:
Assessment:
Plan:

Do not invent facts. If information is missing, say so briefly."""


def build_note_prompt(transcript: str) -> str:
    return f"{SOAP_SYSTEM_PROMPT}\n\nClinical conversation:\n{transcript}\n\nSOAP note:"


COMPACT_SOAP_PROMPT = """Write a concise SOAP note from the transcript.
Use only stated facts; write unknown for missing vitals, labs, meds, or exam."""


def build_compact_note_prompt(transcript: str) -> str:
    return f"{COMPACT_SOAP_PROMPT}\n\nTranscript:\n{transcript}\n\nSOAP:"


def build_prompt_for_mode(transcript: str, prompt_mode: str = "standard") -> str:
    if prompt_mode in {"standard", "soap", "default"}:
        return build_note_prompt(transcript)
    if prompt_mode in {"compact", "compact_soap"}:
        return build_compact_note_prompt(transcript)
    raise ValueError(f"Unknown prompt mode: {prompt_mode}")


FACT_EXTRACTION_PROMPT = """Extract compact clinical facts from the transcript.
Use terse bullets under:
- Chief concern
- Symptoms and duration
- Relevant negatives
- Vitals/labs/medications/exam
- Assessment clues
- Plan preferences
Only include stated facts. Write unknown if missing."""


def build_clinical_facts_prompt(transcript: str) -> str:
    return f"{FACT_EXTRACTION_PROMPT}\n\nTranscript:\n{transcript}\n\nClinical facts:"


def build_note_from_facts_prompt(facts: str) -> str:
    return (
        "Write a concise SOAP note from these extracted clinical facts.\n"
        "Use only the facts provided. If a field is unknown, write unknown.\n\n"
        f"Clinical facts:\n{facts}\n\nSOAP note:"
    )


def build_chunk_summary_prompt(chunk_text: str, instruction: str) -> str:
    return f"{instruction}\n\nChunk:\n{chunk_text}\n\nSummary:"


def build_final_note_from_summaries_prompt(summaries: list[str]) -> str:
    joined = "\n\n".join(f"Chunk {idx + 1} summary:\n{text}" for idx, text in enumerate(summaries))
    return (
        f"{SOAP_SYSTEM_PROMPT}\n\nBelow are chunk-level summaries of a long clinical conversation.\n"
        f"Combine them into one coherent SOAP note.\n\n{joined}\n\nSOAP note:"
    )
