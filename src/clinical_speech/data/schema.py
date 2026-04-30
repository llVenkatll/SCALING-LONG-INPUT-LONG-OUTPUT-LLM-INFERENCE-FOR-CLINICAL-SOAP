from pydantic import BaseModel


class ClinicalSample(BaseModel):
    id: str
    transcript: str | None = None
    reference_note: str
    audio_path: str | None = None
    metadata: dict | None = None
