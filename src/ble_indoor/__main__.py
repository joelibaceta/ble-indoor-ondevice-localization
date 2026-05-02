"""CLI: ``python -m ble_indoor`` (requiere ``PYTHONPATH=src`` desde la raíz del repo o paquete instalado)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    # .../src/ble_indoor/__main__.py -> parents[2] == raíz del repo
    return Path(__file__).resolve().parents[2]


def _bootstrap_paths() -> Path:
    root = _repo_root()
    src = root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))
    return root


def _cmd_train(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")

    from ble_indoor.train.fingerprint import fingerprint_artifact_paths, train_fingerprint_model

    root = _bootstrap_paths()
    result = train_fingerprint_model(
        root,
        model=args.model,
        no_sweep=args.no_sweep,
        k_sweep_max=args.k_sweep_max,
    )
    print(json.dumps(result.metrics, indent=2))
    for p in fingerprint_artifact_paths(result.results_dir):
        print(p.resolve())
    print(result.model_path.resolve())
    print(result.metrics_path.resolve())


def _cmd_generate_csv(args: argparse.Namespace) -> None:
    _bootstrap_paths()
    from ble_indoor import BaselineStudy, ProjectLayout

    root = _repo_root()
    layout = ProjectLayout(root)
    study = BaselineStudy(layout)
    out = study.train_dataset_csv_path()
    if out.is_file() and not args.force:
        raise SystemExit(f"Refusing to overwrite {out} (use --force).")
    df = study.generate_base_dataset("trajectory", save=True)
    print(out.resolve())
    print("rows", len(df))


def _cmd_baseline(args: argparse.Namespace) -> None:
    _bootstrap_paths()
    from ble_indoor import BaselineStudy, ChannelPerturbation, ProjectLayout

    root = _repo_root()
    layout = ProjectLayout(root)
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
            example = root / "simulations/omnet/examples/minimal_valid_example.csv"
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
        study.evaluate_interference(
            ChannelPerturbation(noise_sigma_multiplier=1.6, reception_prob_multiplier=0.85), task="zone"
        ),
    )

    study.train("position")
    print("Position validation (m):", study.validate("position")["metrics"])
    print(
        "Interference (position):",
        study.evaluate_interference(
            ChannelPerturbation(noise_sigma_multiplier=1.8, reception_prob_multiplier=0.9), task="position"
        )["metrics"],
    )


def main() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(prog="python -m ble_indoor", description="CLI BLE indoor (desde la raíz del repo con PYTHONPATH=src).")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Entrenar fingerprint kNN o RandomForest")
    t.add_argument("--model", choices=("knn", "rf"), default="knn")
    t.add_argument("--no-sweep", action="store_true", help="(kNN) Sin gráfico validation_vs_k")
    t.add_argument("--k-sweep-max", type=int, default=30, metavar="K")
    t.set_defaults(func=_cmd_train)

    g = sub.add_parser("generate-csv", help="CSV sintético path-loss (trayectoria)")
    g.add_argument("--force", action="store_true")
    g.set_defaults(func=_cmd_generate_csv)

    b = sub.add_parser("baseline", help="Estudio baseline (holdout, kNN zona/posición, interferencia)")
    b.set_defaults(func=_cmd_baseline)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
