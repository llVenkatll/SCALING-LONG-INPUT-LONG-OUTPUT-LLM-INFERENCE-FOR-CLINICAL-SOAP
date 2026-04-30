from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from array import array
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clinical_speech.storage import PROJECT_ROOT as PACKAGE_PROJECT_ROOT
from clinical_speech.storage import bootstrap_storage_env, resolve_existing_input_path
from clinical_speech.utils.io import write_jsonl


INTERVAL_START_RE = re.compile(r"^\s*intervals \[\d+\]:\s*$")
TEXT_RE = re.compile(r'^\s*text = "(.*)"\s*$')
XMIN_RE = re.compile(r"^\s*xmin = ([0-9.]+)\s*$")
XMAX_RE = re.compile(r"^\s*xmax = ([0-9.]+)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a local PriMock57 checkout into the clinical_speech JSONL schema."
    )
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("/data/project_runtime/datasets/primock57/test.jsonl"))
    parser.add_argument("--mixed-audio-dir", type=Path, default=Path("/data/project_runtime/datasets/primock57/audio_mixed"))
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--skip-audio", action="store_true")
    return parser.parse_args()


def _clean_text(text: str) -> str:
    cleaned = text.replace("<UNIN/>", "[unintelligible]")
    cleaned = cleaned.replace("<UNSURE>", "").replace("</UNSURE>", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _parse_textgrid(path: Path, speaker: str) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip("\n")
        if INTERVAL_START_RE.match(line):
            if current and current.get("text"):
                intervals.append(current)
            current = {"speaker": speaker}
            continue
        if current is None:
            continue
        xmin_match = XMIN_RE.match(line)
        if xmin_match:
            current["xmin"] = float(xmin_match.group(1))
            continue
        xmax_match = XMAX_RE.match(line)
        if xmax_match:
            current["xmax"] = float(xmax_match.group(1))
            continue
        text_match = TEXT_RE.match(line)
        if text_match:
            current["text"] = _clean_text(text_match.group(1))

    if current and current.get("text"):
        intervals.append(current)

    return [item for item in intervals if item.get("text")]


def _load_wav_mono(path: Path) -> tuple[array, int, int]:
    raw = path.read_bytes()
    if raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"{path} looks like a Git LFS pointer, not real audio. Run `git lfs pull` in the PriMock57 checkout first."
        )

    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
    except wave.Error as exc:
        raise RuntimeError(f"Failed to read WAV audio from {path}: {exc}") from exc

    if sample_width != 2:
        raise RuntimeError(f"Expected 16-bit WAV audio in {path}, found sample width {sample_width}")

    samples = array("h")
    samples.frombytes(frames)
    if channels > 1:
        mono = array("h")
        for index in range(0, len(samples), channels):
            channel_values = samples[index : index + channels]
            mono.append(int(sum(channel_values) / len(channel_values)))
        samples = mono
    return samples, sample_rate, sample_width


def _write_stereo_wav(left_path: Path, right_path: Path, output_path: Path) -> None:
    left_samples, left_rate, sample_width = _load_wav_mono(left_path)
    right_samples, right_rate, right_sample_width = _load_wav_mono(right_path)
    if left_rate != right_rate:
        raise RuntimeError(f"Sample rate mismatch between {left_path} and {right_path}")
    if sample_width != right_sample_width:
        raise RuntimeError(f"Sample width mismatch between {left_path} and {right_path}")

    length = max(len(left_samples), len(right_samples))
    if len(left_samples) < length:
        left_samples.extend([0] * (length - len(left_samples)))
    if len(right_samples) < length:
        right_samples.extend([0] * (length - len(right_samples)))

    stereo = array("h")
    for left_value, right_value in zip(left_samples, right_samples):
        stereo.append(left_value)
        stereo.append(right_value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(left_rate)
        wav_file.writeframes(stereo.tobytes())


def _combine_transcript(input_root: Path, base_name: str) -> str:
    doctor_intervals = _parse_textgrid(input_root / "transcripts" / f"{base_name}_doctor.TextGrid", "doctor")
    patient_intervals = _parse_textgrid(input_root / "transcripts" / f"{base_name}_patient.TextGrid", "patient")
    utterances = doctor_intervals + patient_intervals
    utterances.sort(key=lambda item: (item.get("xmin", 0.0), 0 if item["speaker"] == "doctor" else 1))
    return "\n".join(f"[{item['speaker']}]: {item['text']}" for item in utterances if item.get("text"))


def _resolve_output_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return base_dir / path


def main() -> None:
    args = parse_args()
    runtime_paths = bootstrap_storage_env(args.runtime_root, override=args.runtime_root is not None)
    input_root = resolve_existing_input_path(
        args.input_root,
        candidate_bases=[runtime_paths.datasets, PACKAGE_PROJECT_ROOT],
    )
    output_path = _resolve_output_path(args.output, runtime_paths.datasets)
    mixed_audio_dir = _resolve_output_path(args.mixed_audio_dir, runtime_paths.datasets)

    notes_dir = input_root / "notes"
    rows: list[dict[str, Any]] = []

    for note_path in sorted(notes_dir.glob("day*_consultation*.json")):
        payload = json.loads(note_path.read_text(encoding="utf-8"))
        base_name = note_path.stem
        transcript = _combine_transcript(input_root, base_name)

        audio_path: str | None = None
        if not args.skip_audio:
            doctor_audio = input_root / "audio" / f"{base_name}_doctor.wav"
            patient_audio = input_root / "audio" / f"{base_name}_patient.wav"
            mixed_audio_path = mixed_audio_dir / f"{base_name}.wav"
            _write_stereo_wav(doctor_audio, patient_audio, mixed_audio_path)
            audio_path = str(mixed_audio_path)

        rows.append(
            {
                "id": base_name,
                "transcript": transcript,
                "reference_note": payload["note"].strip(),
                "audio_path": audio_path,
                "metadata": {
                    "source": "primock57",
                    "day": payload.get("day"),
                    "consultation": payload.get("consultation"),
                    "presenting_complaint": payload.get("presenting_complaint"),
                    "highlights": payload.get("highlights", []),
                    "audio_mode": "stereo_from_doctor_patient_channels" if audio_path else "not_generated",
                },
            }
        )

    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    if not args.skip_audio:
        print(f"Wrote mixed audio files to {mixed_audio_dir}")


if __name__ == "__main__":
    main()
