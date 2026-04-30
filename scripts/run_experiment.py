import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> None:
    try:
        from clinical_speech.pipeline.runner import main as runner_main
    except ModuleNotFoundError as exc:
        missing = exc.name or "project dependencies"
        raise SystemExit(
            f"Experiment execution requires {missing}. Install the project dependencies, then rerun this command."
        ) from exc

    runner_main()


if __name__ == "__main__":
    main()
