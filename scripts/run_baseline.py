#!/usr/bin/env python3
"""Baseline: prioriza `omnet.training_trace_csv` del YAML, luego env/rutas por defecto."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from ble_indoor import BaselineStudy, ChannelPerturbation, ProjectLayout


def main() -> None:
    layout = ProjectLayout(ROOT)
    study = BaselineStudy(layout)

    df_cfg = study.load_training_trace_from_config()
    if df_cfg is not None:
        print("Dataset:", study.config.omnet.training_trace_csv)
    else:
        trace = Path(
            os.environ.get("OMNET_TRAINING_TRACE", layout.data_simulated_dir() / "omnet_training_trace.csv")
        )
        if trace.is_file():
            study.load_training_trace_from_omnet(trace, save_copy=True)
            print("Dataset:", trace)
        elif os.environ.get("ALLOW_LEGACY_PATHLOSS", "").lower() in ("1", "true", "yes"):
            print("ALLOW_LEGACY_PATHLOSS: synthetic trajectory (Python path loss).")
            study.generate_base_dataset("trajectory", save=True)
        else:
            example = ROOT / "simulations/omnet/examples/minimal_valid_example.csv"
            raise SystemExit(
                f"No training CSV at {trace}\n"
                "simulations/omnet/EXPORT_FORMAT.txt\n"
                f"{example}\n"
                "ALLOW_LEGACY_PATHLOSS=1 enables synthetic trajectory."
            )

    study.prepare_holdout()

    study.train("zone")
    print("Zones nx×ny:", study.config.spatial_zones.nx, study.config.spatial_zones.ny)
    print(study.validate("zone"))
    print(
        "Interference (zones):",
        study.evaluate_interference(ChannelPerturbation(noise_sigma_multiplier=1.6, reception_prob_multiplier=0.85), task="zone"),
    )

    study.train("position")
    print("Position validation (m):", study.validate("position")["metrics"])
    print(
        "Interference (position):",
        study.evaluate_interference(ChannelPerturbation(noise_sigma_multiplier=1.8, reception_prob_multiplier=0.9), task="position")[
            "metrics"
        ],
    )


if __name__ == "__main__":
    main()
