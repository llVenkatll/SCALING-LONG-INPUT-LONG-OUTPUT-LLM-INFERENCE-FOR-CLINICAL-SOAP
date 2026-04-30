import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clinical_speech.storage import (
    DATA_MOUNT_ROOT,
    bootstrap_storage_env,
    disk_status,
    format_disk_status,
    validate_storage_layout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate that runtime storage stays on /data.")
    parser.add_argument("--config", type=Path, default=None, help="Experiment config to validate.")
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--min-free-data-gb", type=float, default=None)
    parser.add_argument("--min-free-root-gb", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_paths = bootstrap_storage_env(args.runtime_root, override=args.runtime_root is not None)

    if args.config is not None:
        try:
            from clinical_speech.config import load_config
        except ModuleNotFoundError as exc:
            missing = exc.name or "project dependencies"
            raise SystemExit(
                f"Config-aware preflight requires {missing}. Install the project dependencies, then rerun this command."
            ) from exc

        cfg = load_config(args.config, runtime_root=args.runtime_root)
        min_free_data_gb = args.min_free_data_gb or cfg.preflight.min_free_data_gb
        min_free_root_gb = args.min_free_root_gb or cfg.preflight.min_free_root_gb
        managed_paths = cfg.managed_paths()
    else:
        min_free_data_gb = args.min_free_data_gb or 20.0
        min_free_root_gb = args.min_free_root_gb or 5.0
        managed_paths = [
            ("runtime_root", runtime_paths.root),
            ("datasets_dir", runtime_paths.datasets),
            ("outputs_dir", runtime_paths.outputs),
            ("logs_dir", runtime_paths.logs),
            ("checkpoints_dir", runtime_paths.checkpoints),
            ("cache_dir", runtime_paths.cache),
            ("tmp_dir", runtime_paths.tmp),
            ("benchmarks_dir", runtime_paths.benchmarks),
            ("profiler_dir", runtime_paths.profiler),
        ]

    warnings = validate_storage_layout(
        managed_paths,
        runtime_root=runtime_paths.root,
        min_data_free_gb=min_free_data_gb,
        min_root_free_gb=min_free_root_gb,
    )

    print("Storage preflight passed.")
    print(format_disk_status(disk_status(DATA_MOUNT_ROOT)))
    print(format_disk_status(disk_status(Path("/"))))
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
