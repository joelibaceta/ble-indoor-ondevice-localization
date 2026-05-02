#!/usr/bin/env python3
"""Write synthetic trajectory training CSV (path loss) to `omnet.training_trace_csv`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from ble_indoor import BaselineStudy, ProjectLayout


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing CSV (default: skip if file exists).",
    )
    args = p.parse_args()

    layout = ProjectLayout(ROOT)
    study = BaselineStudy(layout)
    out = study.train_dataset_csv_path()
    if out.is_file() and not args.force:
        raise SystemExit(f"Refusing to overwrite {out} (use --force).")
    df = study.generate_base_dataset("trajectory", save=True)
    print(out.resolve())
    print("rows", len(df))


if __name__ == "__main__":
    main()
