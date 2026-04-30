import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from clinical_speech.storage import PROJECT_ROOT, RuntimePaths, bootstrap_storage_env


class ExperimentConfig(BaseModel):
    name: str
    output_dir: Path | None = None
    log_dir: Path | None = None
    benchmark_dir: Path | None = None
    profiler_dir: Path | None = None
    checkpoint_dir: Path | None = None
    seed: int = 42


class DatasetConfig(BaseModel):
    path: Path
    max_samples: int | None = None
    task: str = "transcript_to_note"


class ModelConfig(BaseModel):
    asr_model_id: str
    llm_model_id: str
    device: str = "cuda"
    dtype: str = "float16"
    load_in_8bit: bool = False
    attn_implementation: str | None = None


class GenerationConfig(BaseModel):
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


class ChunkingConfig(BaseModel):
    enabled: bool = False
    chunk_words: int = 1200
    overlap_words: int = 150
    summary_prompt: str


class OutputConfig(BaseModel):
    note_style: str = "soap"
    save_predictions: bool = True
    save_metrics: bool = True
    save_raw_intermediates: bool = False
    cleanup_temp_on_success: bool = True
    max_checkpoints_to_keep: int = 2
    save_frequency: int = 0
    keep_best_checkpoint: bool = True
    keep_latest_checkpoint: bool = True
    max_log_bytes: int = 5 * 1024 * 1024
    log_backup_count: int = 3
    stale_tmp_hours: int = 24


class BenchmarkConfig(BaseModel):
    enabled: bool = True
    warmup_runs: int = Field(default=0, ge=0)
    repeat_runs: int = Field(default=1, ge=1)
    synchronize_cuda: bool = True
    benchmark_only: bool = False
    save_per_run_metrics: bool = True
    save_csv: bool = True


class RuntimeConfig(BaseModel):
    backend: str = "hf_generate"
    scheduler_mode: str = "none"
    block_size_tokens: int = Field(default=128, ge=1)
    max_batch_size: int = Field(default=4, ge=1)
    max_concurrent_requests: int = Field(default=4, ge=1)
    max_cache_budget_gb: float = Field(default=2.0, gt=0.0)
    enable_prefix_cache: bool = False
    prefix_cache_kind: str = "note_prompt"
    triton_enabled: bool = True
    triton_paged_kv_enabled: bool = False
    pad_to_multiple_of: int = Field(default=8, ge=1)


class SystemsBackendSpecConfig(BaseModel):
    label: str | None = None
    provider: str = "local_hf"
    api_model_id: str | None = None
    tokenizer_model_id: str | None = None
    llm_model_id: str | None = None
    asr_model_id: str | None = None
    device: str | None = None
    dtype: str | None = None
    load_in_8bit: bool | None = None
    attn_implementation: str | None = None
    model_family: str | None = None
    deployment: str | None = None
    serving_stack: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    stream: bool = True
    stream_include_usage: bool = True
    timeout_sec: float = Field(default=180.0, gt=0.0)


# Backward-compatible shorthand for simple hosted-backend snippets.
BackendSpec = SystemsBackendSpecConfig


class SystemsBenchmarkConfig(BaseModel):
    enabled: bool = False
    num_prompts: int = Field(default=8, ge=1)
    batch_sizes: list[int] = Field(default_factory=lambda: [1, 2, 4])
    backends: list[str] = Field(
        default_factory=lambda: [
            "hf_sequential",
            "mistral_paged_single",
            "mistral_paged_static_batch",
        ]
    )
    warmup_batches: int = Field(default=1, ge=0)
    repeat_batches: int = Field(default=3, ge=1)
    max_new_tokens: int | None = Field(default=None, ge=1)
    prompt_mode: str = "standard"
    stage1_max_new_tokens: int = Field(default=256, ge=1)
    stage2_max_new_tokens: int | None = Field(default=None, ge=1)
    save_figures: bool = True
    backend_specs: dict[str, SystemsBackendSpecConfig] = Field(default_factory=dict)


class StorageConfig(BaseModel):
    runtime_root: Path | None = None
    datasets_dir: Path | None = None
    checkpoints_dir: Path | None = None
    logs_dir: Path | None = None
    cache_dir: Path | None = None
    hf_dir: Path | None = None
    torch_dir: Path | None = None
    profiler_dir: Path | None = None
    outputs_dir: Path | None = None
    tmp_dir: Path | None = None
    benchmarks_dir: Path | None = None


class PreflightConfig(BaseModel):
    enabled: bool = True
    min_free_data_gb: float = 20.0
    min_free_root_gb: float = 5.0


class AppConfig(BaseModel):
    experiment: ExperimentConfig
    dataset: DatasetConfig
    model: ModelConfig
    generation: GenerationConfig
    chunking: ChunkingConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    systems_benchmark: SystemsBenchmarkConfig = Field(default_factory=SystemsBenchmarkConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    preflight: PreflightConfig = Field(default_factory=PreflightConfig)

    def managed_paths(self) -> list[tuple[str, Path]]:
        return [
            ("runtime_root", self.storage.runtime_root),
            ("datasets_dir", self.storage.datasets_dir),
            ("checkpoints_dir", self.storage.checkpoints_dir),
            ("logs_dir", self.storage.logs_dir),
            ("cache_dir", self.storage.cache_dir),
            ("hf_dir", self.storage.hf_dir),
            ("torch_dir", self.storage.torch_dir),
            ("profiler_dir", self.storage.profiler_dir),
            ("outputs_dir", self.storage.outputs_dir),
            ("tmp_dir", self.storage.tmp_dir),
            ("benchmarks_dir", self.storage.benchmarks_dir),
            ("dataset_path", self.dataset.path),
            ("output_dir", self.experiment.output_dir),
            ("log_dir", self.experiment.log_dir),
            ("benchmark_dir", self.experiment.benchmark_dir),
            ("profiler_dir", self.experiment.profiler_dir),
            ("checkpoint_dir", self.experiment.checkpoint_dir),
        ]


def _expand_env_values(payload):
    if isinstance(payload, dict):
        return {key: _expand_env_values(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_expand_env_values(item) for item in payload]
    if isinstance(payload, str):
        return os.path.expanduser(os.path.expandvars(payload))
    return payload


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(config_path: Path, seen: set[Path] | None = None) -> dict:
    seen = seen or set()
    resolved_path = config_path.resolve()
    if resolved_path in seen:
        raise ValueError(f"Config inheritance cycle detected at {resolved_path}")
    seen.add(resolved_path)

    with open(resolved_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    parent_ref = raw.pop("extends", None)
    if not parent_ref:
        return raw

    parent_path = Path(parent_ref)
    if not parent_path.is_absolute():
        local_parent = (resolved_path.parent / parent_path).resolve()
        parent_path = local_parent if local_parent.exists() else (PROJECT_ROOT / parent_path).resolve()
    parent_raw = _load_raw_config(parent_path, seen=seen)
    return _deep_merge_dicts(parent_raw, raw)


def _resolve_directory(path: Path | None, *, default: Path, base_dir: Path) -> Path:
    if path is None:
        resolved = default
    else:
        resolved = path if path.is_absolute() else (base_dir / path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    return path if path.is_absolute() else (base_dir / path).resolve()


def _apply_runtime_defaults(cfg: AppConfig, runtime_paths: RuntimePaths) -> AppConfig:
    cfg.storage.runtime_root = _resolve_directory(
        cfg.storage.runtime_root,
        default=runtime_paths.root,
        base_dir=PROJECT_ROOT,
    )
    cfg.storage.datasets_dir = _resolve_directory(
        cfg.storage.datasets_dir,
        default=runtime_paths.datasets,
        base_dir=cfg.storage.runtime_root,
    )
    cfg.storage.checkpoints_dir = _resolve_directory(
        cfg.storage.checkpoints_dir,
        default=runtime_paths.checkpoints,
        base_dir=cfg.storage.runtime_root,
    )
    cfg.storage.logs_dir = _resolve_directory(
        cfg.storage.logs_dir,
        default=runtime_paths.logs,
        base_dir=cfg.storage.runtime_root,
    )
    cfg.storage.cache_dir = _resolve_directory(
        cfg.storage.cache_dir,
        default=runtime_paths.cache,
        base_dir=cfg.storage.runtime_root,
    )
    cfg.storage.hf_dir = _resolve_directory(
        cfg.storage.hf_dir,
        default=runtime_paths.hf,
        base_dir=cfg.storage.runtime_root,
    )
    cfg.storage.torch_dir = _resolve_directory(
        cfg.storage.torch_dir,
        default=runtime_paths.torch,
        base_dir=cfg.storage.runtime_root,
    )
    cfg.storage.profiler_dir = _resolve_directory(
        cfg.storage.profiler_dir,
        default=runtime_paths.profiler,
        base_dir=cfg.storage.runtime_root,
    )
    cfg.storage.outputs_dir = _resolve_directory(
        cfg.storage.outputs_dir,
        default=runtime_paths.outputs,
        base_dir=cfg.storage.runtime_root,
    )
    cfg.storage.tmp_dir = _resolve_directory(
        cfg.storage.tmp_dir,
        default=runtime_paths.tmp,
        base_dir=cfg.storage.runtime_root,
    )
    cfg.storage.benchmarks_dir = _resolve_directory(
        cfg.storage.benchmarks_dir,
        default=runtime_paths.benchmarks,
        base_dir=cfg.storage.runtime_root,
    )

    cfg.dataset.path = _resolve_path(cfg.dataset.path, base_dir=PROJECT_ROOT)
    cfg.experiment.output_dir = _resolve_directory(
        cfg.experiment.output_dir,
        default=cfg.storage.outputs_dir / cfg.experiment.name,
        base_dir=cfg.storage.outputs_dir,
    )
    cfg.experiment.log_dir = _resolve_directory(
        cfg.experiment.log_dir,
        default=cfg.storage.logs_dir / cfg.experiment.name,
        base_dir=cfg.storage.logs_dir,
    )
    cfg.experiment.benchmark_dir = _resolve_directory(
        cfg.experiment.benchmark_dir,
        default=cfg.storage.benchmarks_dir / cfg.experiment.name,
        base_dir=cfg.storage.benchmarks_dir,
    )
    cfg.experiment.profiler_dir = _resolve_directory(
        cfg.experiment.profiler_dir,
        default=cfg.storage.profiler_dir / cfg.experiment.name,
        base_dir=cfg.storage.profiler_dir,
    )
    cfg.experiment.checkpoint_dir = _resolve_directory(
        cfg.experiment.checkpoint_dir,
        default=cfg.storage.checkpoints_dir / cfg.experiment.name,
        base_dir=cfg.storage.checkpoints_dir,
    )
    return cfg


def load_config(path: str | Path, runtime_root: str | Path | None = None) -> AppConfig:
    runtime_paths = bootstrap_storage_env(runtime_root, override=runtime_root is not None)
    config_path = Path(path)
    if not config_path.is_absolute():
        repo_relative = (PROJECT_ROOT / config_path).resolve()
        config_path = repo_relative if repo_relative.exists() else config_path.resolve()
    raw = _load_raw_config(config_path)
    systems_benchmark = raw.get("systems_benchmark")
    if isinstance(systems_benchmark, dict) and "enabled_backends" in systems_benchmark:
        systems_benchmark["backends"] = systems_benchmark["enabled_backends"]
        systems_benchmark.pop("enabled_backends", None)
    cfg = AppConfig.model_validate(_expand_env_values(raw or {}))
    return _apply_runtime_defaults(cfg, runtime_paths)
