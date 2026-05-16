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
    config_path = Path(args.config).resolve() if args.config else None
    result = train_fingerprint_model(
        root,
        config_path=config_path,
        model=args.model,
        no_sweep=args.no_sweep,
        k_sweep_max=args.k_sweep_max,
    )
    print(json.dumps(result.metrics, indent=2))
    for p in fingerprint_artifact_paths(result.results_dir):
        print(p.resolve())
    print(result.model_path.resolve())
    print(result.metrics_path.resolve())

    if args.model == "mlp" and args.export_tflite:
        from ble_indoor.models.fingerprint_mlp import FingerprintMlpEstimator

        est = FingerprintMlpEstimator.load(result.model_path)
        quantize = not args.no_quantize
        tflite_path = result.results_dir / f"model_{'int8' if quantize else 'f32'}.tflite"
        est.export_tflite(tflite_path, quantize=quantize)
        print(tflite_path.resolve())


def _cmd_generate_csv(args: argparse.Namespace) -> None:
    _bootstrap_paths()
    from ble_indoor import BaselineStudy, ProjectLayout
    from ble_indoor.simulation.ports import RssiObservationSource

    root = _repo_root()
    layout = ProjectLayout(root)
    config_path = Path(args.config).resolve() if args.config else None
    study = BaselineStudy(layout, config_path=config_path)
    out = study.train_dataset_csv_path()
    if out.is_file() and not args.force:
        raise SystemExit(f"Refusing to overwrite {out} (use --force).")

    src: RssiObservationSource | None = None
    if args.simulator == "sionna":
        try:
            from ble_indoor.simulation.sionna_rt_simulator import SionnaRTSimulator
        except ImportError as exc:
            raise SystemExit(
                "Sionna RT no está instalado. Ejecutar:\n"
                "  pip install -r requirements-sionna.txt\n"
                f"Error original: {exc}"
            ) from exc
        print(f"[generate-csv] Usando SionnaRTSimulator (cache: {study.config.sionna_rt.cache_file})")
        src = SionnaRTSimulator(study.environment, study.config.sionna_rt, layout_root=root)
    else:
        print("[generate-csv] Usando PathLossSimulator (analítico)")

    df = study.generate_base_dataset("trajectory", save=True, rssi_source=src)
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
        print("Dataset:", study.config.training_data.training_trace_csv)
    else:
        trace = Path(
            os.environ.get("TRAINING_TRACE_CSV", layout.data_simulated_dir() / "training_trace.csv")
        )
        if trace.is_file():
            study.load_training_trace_from_omnet(trace, save_copy=True)
            print("Dataset:", trace)
        elif os.environ.get("ALLOW_LEGACY_PATHLOSS", "").lower() in ("1", "true", "yes"):
            print("ALLOW_LEGACY_PATHLOSS: synthetic trajectory (Python path loss).")
            study.generate_base_dataset("trajectory", save=True)
        else:
            example = root / "simulations/examples/minimal_valid_example.csv"
            raise SystemExit(
                f"No training CSV at {trace}\n"
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

    t = sub.add_parser("train", help="Entrenar fingerprint kNN, RandomForest o MLP")
    t.add_argument("--model", choices=("knn", "rf", "mlp"), default="knn")
    t.add_argument("--no-sweep", action="store_true", help="(kNN) Sin gráfico validation_vs_k")
    t.add_argument("--k-sweep-max", type=int, default=30, metavar="K")
    t.add_argument("--config", default=None, metavar="YAML", help="Ruta al YAML de configuración (default: config/baseline_room.yaml)")
    t.add_argument("--export-tflite", action="store_true", help="(mlp) Exportar modelo a TFLite tras el entrenamiento")
    t.add_argument("--no-quantize", action="store_true", help="(mlp) Exportar en float32 en lugar de INT8")
    t.set_defaults(func=_cmd_train)

    g = sub.add_parser("generate-csv", help="CSV de entrenamiento (path loss o Sionna RT)")
    g.add_argument("--force", action="store_true")
    g.add_argument(
        "--simulator",
        choices=("pathloss", "sionna"),
        default="pathloss",
        help="Fuente RSSI: 'pathloss' (analítico, default) o 'sionna' (Sionna RT ray tracing, requiere requirements-sionna.txt)",
    )
    g.add_argument("--config", default=None, metavar="YAML", help="Ruta al YAML de configuración (default: config/baseline_room.yaml)")
    g.set_defaults(func=_cmd_generate_csv)

    b = sub.add_parser("baseline", help="Estudio baseline (holdout, kNN zona/posición, interferencia)")
    b.set_defaults(func=_cmd_baseline)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
