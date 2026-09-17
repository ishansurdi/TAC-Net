"""Single entry point for TAC-Net verification and retraining."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tacnet"))


def main() -> None:
    parser = argparse.ArgumentParser(description="TAC-Net reproducibility entry point")
    parser.add_argument("command", choices=("verify", "train"))
    args, forwarded = parser.parse_known_args()
    sys.argv = [sys.argv[0], *forwarded]

    if args.command == "verify":
        import report
        report.main()
        return

    import experiment
    experiment.main()


if __name__ == "__main__":
    main()
