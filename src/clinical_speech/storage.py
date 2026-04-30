from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_ROOT = Path("/data/project_runtime")
DATA_MOUNT_ROOT = Path("/data")
UNSAFE_PREFIXES = (
    Path("/"),
    Path("/home/ubuntu"),
    Path("/tmp"),
)
ENV_TO_SUBDIR = {
    "CLINICAL_SPEECH_DATASETS_DIR": "datasets",
    "CLINICAL_SPEECH_CHECKPOINTS_DIR": "checkpoints",
    "CLINICAL_SPEECH_LOGS_DIR": "logs",
    "CLINICAL_SPEECH_CACHE_DIR": "cache",
    "CLINICAL_SPEECH_HF_DIR": "hf",
    "CLINICAL_SPEECH_TORCH_DIR": "torch",
    "CLINICAL_SPEECH_PROFILER_DIR": "profiler",
    "CLINICAL_SPEECH_OUTPUTS_DIR": "outputs",
    "CLINICAL_SPEECH_TMP_DIR": "tmp",
    "CLINICAL_SPEECH_BENCHMARKS_DIR": "benchmarks",
}


def _expand_path(value: str | Path) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(value)))).resolve()


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    datasets: Path
    checkpoints: Path
    logs: Path
    cache: Path
    hf: Path
    torch: Path
    profiler: Path
    outputs: Path
    tmp: Path
    benchmarks: Path

    def directories(self) -> tuple[Path, ...]:
        return (
            self.root,
            self.datasets,
            self.checkpoints,
            self.logs,
            self.cache,
            self.hf,
            self.torch,
            self.profiler,
            self.outputs,
            self.tmp,
            self.benchmarks,
        )

    def ensure_directories(self) -> None:
        for directory in self.directories():
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DiskStatus:
    mount: Path
    total_gb: float
    used_gb: float
    free_gb: float
    used_pct: float


class StorageValidationError(RuntimeError):
    pass


_ACTIVE_RUNTIME_PATHS: RuntimePaths | None = None


def build_runtime_paths(
    runtime_root: str | Path | None = None,
    *,
    ignore_env_overrides: bool = False,
) -> RuntimePaths:
    root_value = runtime_root or os.environ.get("CLINICAL_SPEECH_RUNTIME_ROOT") or DEFAULT_RUNTIME_ROOT
    root = _expand_path(root_value)
    subdirs: dict[str, Path] = {}
    for env_name, subdir in ENV_TO_SUBDIR.items():
        value = None if ignore_env_overrides else os.environ.get(env_name)
        subdirs[subdir] = _expand_path(value) if value else (root / subdir).resolve()

    return RuntimePaths(
        root=root,
        datasets=subdirs["datasets"],
        checkpoints=subdirs["checkpoints"],
        logs=subdirs["logs"],
        cache=subdirs["cache"],
        hf=subdirs["hf"],
        torch=subdirs["torch"],
        profiler=subdirs["profiler"],
        outputs=subdirs["outputs"],
        tmp=subdirs["tmp"],
        benchmarks=subdirs["benchmarks"],
    )


def _apply_env_defaults(paths: RuntimePaths, *, override: bool) -> None:
    managed_env = {
        "CLINICAL_SPEECH_RUNTIME_ROOT": paths.root,
        "CLINICAL_SPEECH_DATASETS_DIR": paths.datasets,
        "CLINICAL_SPEECH_CHECKPOINTS_DIR": paths.checkpoints,
        "CLINICAL_SPEECH_LOGS_DIR": paths.logs,
        "CLINICAL_SPEECH_CACHE_DIR": paths.cache,
        "CLINICAL_SPEECH_HF_DIR": paths.hf,
        "CLINICAL_SPEECH_TORCH_DIR": paths.torch,
        "CLINICAL_SPEECH_PROFILER_DIR": paths.profiler,
        "CLINICAL_SPEECH_OUTPUTS_DIR": paths.outputs,
        "CLINICAL_SPEECH_TMP_DIR": paths.tmp,
        "CLINICAL_SPEECH_BENCHMARKS_DIR": paths.benchmarks,
        "HF_HOME": paths.hf,
        "HUGGINGFACE_HUB_CACHE": paths.hf / "hub",
        "TRANSFORMERS_CACHE": paths.hf / "transformers",
        "HF_DATASETS_CACHE": paths.hf / "datasets",
        "TORCH_HOME": paths.torch,
        "XDG_CACHE_HOME": paths.cache / "xdg",
        "TMPDIR": paths.tmp,
        "TMP": paths.tmp,
        "TEMP": paths.tmp,
        "WANDB_DIR": paths.outputs / "wandb",
        "WANDB_CACHE_DIR": paths.cache / "wandb",
        "WANDB_CONFIG_DIR": paths.cache / "wandb-config",
        "PIP_CACHE_DIR": paths.cache / "pip",
        "MPLCONFIGDIR": paths.cache / "matplotlib",
    }
    for env_name, value in managed_env.items():
        if override or env_name not in os.environ:
            os.environ[env_name] = str(value)
        _expand_path(os.environ[env_name]).mkdir(parents=True, exist_ok=True)

    tempfile.tempdir = os.environ["TMPDIR"]


def bootstrap_storage_env(
    runtime_root: str | Path | None = None,
    *,
    create_dirs: bool = True,
    override: bool = False,
) -> RuntimePaths:
    global _ACTIVE_RUNTIME_PATHS

    paths = build_runtime_paths(
        runtime_root,
        ignore_env_overrides=override and runtime_root is not None,
    )
    if create_dirs:
        paths.ensure_directories()
    _apply_env_defaults(paths, override=override)
    _ACTIVE_RUNTIME_PATHS = paths
    return paths


def get_runtime_paths(runtime_root: str | Path | None = None) -> RuntimePaths:
    global _ACTIVE_RUNTIME_PATHS
    if runtime_root is not None:
        return bootstrap_storage_env(runtime_root, override=True)
    if _ACTIVE_RUNTIME_PATHS is None:
        _ACTIVE_RUNTIME_PATHS = bootstrap_storage_env()
    return _ACTIVE_RUNTIME_PATHS


def resolve_managed_path(
    path: str | Path,
    *,
    base_dir: Path,
    strip_prefixes: Sequence[str] = (),
) -> Path:
    raw = Path(os.path.expanduser(os.path.expandvars(str(path))))
    if raw.is_absolute():
        return raw.resolve()
    parts = raw.parts
    if parts and parts[0] in strip_prefixes:
        raw = Path(*parts[1:]) if len(parts) > 1 else Path()
    return (base_dir / raw).resolve()


def resolve_existing_input_path(
    path: str | Path,
    *,
    candidate_bases: Sequence[Path],
    strip_prefixes: Sequence[str] = (),
) -> Path:
    raw = Path(os.path.expanduser(os.path.expandvars(str(path))))
    if raw.is_absolute():
        return raw.resolve()
    parts = raw.parts
    if parts and parts[0] in strip_prefixes:
        raw = Path(*parts[1:]) if len(parts) > 1 else Path()
    candidate_paths = [(base / raw).resolve() for base in candidate_bases]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return candidate_paths[0]


def disk_status(mount: str | Path) -> DiskStatus:
    total, used, free = shutil.disk_usage(str(mount))
    total_gb = total / (1024 ** 3)
    used_gb = used / (1024 ** 3)
    free_gb = free / (1024 ** 3)
    used_pct = (used / total * 100.0) if total else 0.0
    return DiskStatus(
        mount=Path(mount),
        total_gb=total_gb,
        used_gb=used_gb,
        free_gb=free_gb,
        used_pct=used_pct,
    )


def format_disk_status(status: DiskStatus) -> str:
    return (
        f"{status.mount}: free={status.free_gb:.1f} GiB, "
        f"used={status.used_gb:.1f}/{status.total_gb:.1f} GiB ({status.used_pct:.1f}%)"
    )


def repo_storage_risks(repo_root: Path = PROJECT_ROOT) -> list[str]:
    warnings = []
    suspicious_paths = [
        repo_root / "outputs",
        repo_root / "logs",
        repo_root / "checkpoints",
        repo_root / "cache",
        repo_root / "tmp",
        repo_root / "profiler",
        repo_root / "benchmarks",
        repo_root / "wandb",
        repo_root / ".cache",
        repo_root / "033026_colab" / "outputs",
    ]
    for candidate in suspicious_paths:
        if candidate.exists():
            warnings.append(
                f"Repository path {candidate} exists. Keep heavy runtime artifacts under /data/project_runtime instead of the checkout."
            )

    home_cache_paths = [
        Path.home() / ".cache" / "huggingface",
        Path.home() / ".cache" / "torch",
        Path.home() / ".cache" / "wandb",
    ]
    for candidate in home_cache_paths:
        if candidate.exists():
            warnings.append(
                f"Legacy cache {candidate} exists. Remove or migrate it if root disk pressure returns."
            )
    return warnings


def validate_storage_layout(
    managed_paths: Sequence[tuple[str, Path]],
    *,
    runtime_root: Path,
    repo_root: Path = PROJECT_ROOT,
    min_data_free_gb: float = 20.0,
    min_root_free_gb: float = 5.0,
) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []
    input_like_labels = {
        "dataset_path",
        "predictions_path",
        "long_context_input",
        "dataset_input",
    }
    data_status = disk_status(DATA_MOUNT_ROOT)
    root_status = disk_status(Path("/"))

    if data_status.free_gb < min_data_free_gb:
        errors.append(
            f"{format_disk_status(data_status)}. Free space on /data is below the required {min_data_free_gb:.1f} GiB."
        )
    if root_status.free_gb < min_root_free_gb:
        errors.append(
            f"{format_disk_status(root_status)}. Free space on / is below the required {min_root_free_gb:.1f} GiB."
        )

    for label, managed_path in managed_paths:
        resolved = managed_path.resolve()
        if not _is_relative_to(resolved, DATA_MOUNT_ROOT):
            errors.append(
                f"{label} resolves to {resolved}, which is outside /data. Move it under {runtime_root} or another /data/... directory."
            )
        if _is_relative_to(resolved, Path("/home/ubuntu")):
            errors.append(
                f"{label} resolves to {resolved} under /home/ubuntu. Heavy artifacts must not be written to the home directory."
            )
        if _is_relative_to(resolved, Path("/tmp")):
            errors.append(
                f"{label} resolves to {resolved} under /tmp. Set it to {runtime_root} so temp files stay on the data disk."
            )
        if _is_relative_to(resolved, repo_root) and not _is_relative_to(resolved, runtime_root):
            message = (
                f"{label} resolves to {resolved} inside the repository checkout."
                f" Use {runtime_root} for heavy outputs instead of the repo root."
            )
            if label in input_like_labels:
                warnings.append(message)
            else:
                errors.append(message)

    env_checks = [
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
        "TMPDIR",
        "WANDB_DIR",
        "WANDB_CACHE_DIR",
        "WANDB_CONFIG_DIR",
        "PIP_CACHE_DIR",
    ]
    for env_name in env_checks:
        value = os.environ.get(env_name)
        if not value:
            warnings.append(f"{env_name} is not set. The Python bootstrap normally sets it automatically.")
            continue
        resolved = _expand_path(value)
        if not _is_relative_to(resolved, DATA_MOUNT_ROOT):
            errors.append(
                f"{env_name} points to {resolved}, which is outside /data. Re-source scripts/setup_storage_env.sh before launching jobs."
            )

    warnings.extend(repo_storage_risks(repo_root))

    if errors:
        raise StorageValidationError("\n".join(errors))
    return warnings


def configure_rotating_log(
    log_dir: Path,
    *,
    logger_name: str = "clinical_speech",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return log_path


def prune_old_files(directory: Path, *, pattern: str, keep: int) -> list[Path]:
    if keep < 0:
        return []
    files = sorted(directory.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    removed: list[Path] = []
    for stale in files[keep:]:
        stale.unlink(missing_ok=True)
        removed.append(stale)
    return removed


def cleanup_stale_temp_dirs(base_tmp_dir: Path, *, older_than_hours: int = 24) -> list[Path]:
    if older_than_hours <= 0:
        return []
    cutoff = time.time() - older_than_hours * 3600
    removed: list[Path] = []
    if not base_tmp_dir.exists():
        return removed
    for candidate in base_tmp_dir.iterdir():
        try:
            mtime = candidate.stat().st_mtime
        except FileNotFoundError:
            continue
        if mtime >= cutoff:
            continue
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)
        else:
            candidate.unlink(missing_ok=True)
        removed.append(candidate)
    return removed


@contextmanager
def managed_temp_dir(
    base_tmp_dir: Path,
    *,
    prefix: str,
    cleanup: bool = True,
):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    temp_dir = (base_tmp_dir / f"{prefix}{timestamp}").resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield temp_dir
    finally:
        if cleanup:
            shutil.rmtree(temp_dir, ignore_errors=True)


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
