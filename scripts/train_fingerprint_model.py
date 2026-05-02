#!/usr/bin/env python3
"""CLI: entrena kNN o RandomForest (ver ``ble_indoor.train.fingerprint``)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from ble_indoor.train.fingerprint import fingerprint_artifact_paths, train_fingerprint_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        choices=("knn", "rf"),
        default="knn",
        help="Fingerprint backend: kNN (lazy) or RandomForest.",
    )
    ap.add_argument("--no-sweep", action="store_true", help="(kNN only) Skip validation vs k plot.")
    ap.add_argument("--k-sweep-max", type=int, default=30, metavar="K", help="(kNN only) Max k for sweep.")
    args = ap.parse_args()

    result = train_fingerprint_model(
        ROOT,
        model=args.model,
        no_sweep=args.no_sweep,
        k_sweep_max=args.k_sweep_max,
    )

    print(json.dumps(result.metrics, indent=2))
    for p in fingerprint_artifact_paths(result.results_dir):
        print(p.resolve())
    print(result.model_path.resolve())
    print(result.metrics_path.resolve())


if __name__ == "__main__":
    main()
