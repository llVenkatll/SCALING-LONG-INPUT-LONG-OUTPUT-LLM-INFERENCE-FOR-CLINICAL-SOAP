from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from clinical_speech.config import ModelConfig
from clinical_speech.storage import get_runtime_paths


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(name, torch.float16)


@dataclass
class CausalLMBundle:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM


def load_causal_lm_bundle(model_config: ModelConfig, *, padding_side: str = "right") -> CausalLMBundle:
    runtime_paths = get_runtime_paths()
    cache_dir = str(runtime_paths.hf)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.llm_model_id,
        cache_dir=cache_dir,
        padding_side=padding_side,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if model_config.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_kwargs = {
        "dtype": dtype_from_name(model_config.dtype),
        "device_map": "auto",
        "cache_dir": cache_dir,
    }
    if model_config.attn_implementation:
        model_kwargs["attn_implementation"] = model_config.attn_implementation
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(
        model_config.llm_model_id,
        **model_kwargs,
    )
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None
    return CausalLMBundle(tokenizer=tokenizer, model=model)
