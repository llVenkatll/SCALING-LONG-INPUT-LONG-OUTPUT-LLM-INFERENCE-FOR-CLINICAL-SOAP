from pathlib import Path

import numpy as np
import torch

from transformers import pipeline

from clinical_speech.config import ModelConfig
from clinical_speech.storage import get_runtime_paths


class ASRModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        runtime_paths = get_runtime_paths()
        model_kwargs = {"cache_dir": str(runtime_paths.hf)}
        if config.dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16
        elif config.dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif config.dtype == "float32":
            model_kwargs["torch_dtype"] = torch.float32
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=config.asr_model_id,
            device=self._resolve_device(config.device),
            model_kwargs=model_kwargs,
        )

    @staticmethod
    def _resolve_device(device: str):
        lowered = str(device).lower()
        if lowered.startswith("cuda"):
            return 0
        if lowered == "cpu":
            return -1
        return device

    def _transcribe_wav_path(self, audio_path: Path) -> str:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(audio_path))
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        audio_array = waveform.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        result = self.pipe(
            {
                "array": audio_array,
                "sampling_rate": int(sample_rate),
            }
        )
        if isinstance(result, dict):
            return result.get("text", "").strip()
        return str(result).strip()

    def transcribe(self, audio_path: str | Path | tuple[int, np.ndarray] | dict) -> str:
        if isinstance(audio_path, tuple) and len(audio_path) == 2:
            sample_rate, audio_array = audio_path
            result = self.pipe(
                {
                    "array": np.asarray(audio_array, dtype=np.float32),
                    "sampling_rate": int(sample_rate),
                }
            )
        elif isinstance(audio_path, dict):
            result = self.pipe(audio_path)
        else:
            resolved = Path(audio_path)
            if resolved.suffix.lower() == ".wav":
                return self._transcribe_wav_path(resolved)
            result = self.pipe(str(resolved))
        if isinstance(result, dict):
            return result.get("text", "").strip()
        return str(result).strip()
