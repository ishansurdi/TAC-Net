"""Single entry point for TAC-Net verification and retraining."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tacnet"))


def main() -> None:
    commands = ("verify", "benchmark", "train", "paper-run")
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        choices = ", ".join(commands)
        raise SystemExit(f"Usage: python run.py <command> [options]\nCommands: {choices}")
    command = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]

    if command == "verify":
        import report
        report.main()
        return

    if command in ("benchmark", "train"):
        import model
        model.main()
        return

    import experiment
    experiment.main()


if __name__ == "__main__":
    main()
