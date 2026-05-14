#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation import run_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate locally generated Task 1 simulator outputs.")
    parser.add_argument("--config", default="config/default_config.yaml", help="Path to default YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_validation(args.config)
    print("Validation complete. Outputs written to outputs/validation/ by default.")


if __name__ == "__main__":
    main()
