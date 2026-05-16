"""Entrenamiento de localizadores fingerprint (kNN o RandomForest) y escritura de artefactos."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.model_selection import train_test_split

from ble_indoor import FingerprintKnnEstimator, FingerprintMlpEstimator, FingerprintRfEstimator, ProjectConfig, ProjectLayout
from ble_indoor.evaluation.knn_sweep import sweep_k_neighbors
from ble_indoor.models.knn_zone import ZONE_ID_COLUMN
from ble_indoor.simulation.trace_loader import load_training_trace

ModelId = Literal["knn", "rf", "mlp"]


@dataclass(frozen=True)
class FingerprintTrainResult:
    """Rutas y métricas tras ``train_fingerprint_model``."""

    metrics: dict[str, Any]
    results_dir: Path
    model_path: Path
    metrics_path: Path


def train_fingerprint_model(
    repo_root: Path,
    *,
    config_path: Path | None = None,
    model: ModelId = "knn",
    no_sweep: bool = False,
    k_sweep_max: int = 30,
    random_state: int = 123,
    test_size: float = 0.20,
) -> FingerprintTrainResult:
    """
    Carga el CSV de entrenamiento, hace split train/val, ajusta el modelo y guarda
    artefactos bajo ``data/results/<exp>/<model>/`` (exp = nombre del archivo de config).

    Importa módulos que usan Matplotlib solo aquí dentro, para permitir ``matplotlib.use('Agg')``
    en el entrypoint antes de llamar a esta función.
    """
    from ble_indoor.evaluation.plots import plot_knn_validation_vs_k
    from ble_indoor.evaluation.training_figures import write_fingerprint_training_figures

    layout = ProjectLayout(repo_root)
    cfg_path = config_path or layout.default_config_path()
    cfg = ProjectConfig.load(cfg_path)
    csv_path = layout.resolve_repo_path(cfg.training_data.training_trace_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"No training CSV at {csv_path}")
    df = load_training_trace(csv_path, cfg.environment, cfg.spatial_zones)

    y_zone = df[ZONE_ID_COLUMN].to_numpy(dtype=np.int64)
    counts = np.bincount(y_zone, minlength=cfg.spatial_zones.n_zones)
    strat = y_zone if int(np.min(counts)) >= 5 else None
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True, stratify=strat
    )

    exp_name = Path(cfg_path).stem
    if exp_name == "baseline_room":
        results_dir = layout.model_results_dir(model)
    else:
        results_dir = layout.data_results_dir() / exp_name / model
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {
        "model_id": model,
        "n_train": len(train_df),
        "n_validation": len(val_df),
    }

    if model == "knn":
        est = FingerprintKnnEstimator.from_config(cfg)
        est.fit(train_df, fit_zone=True)
        metrics["model"] = {
            "type": "knn",
            "position_k_neighbors": est.position_k_neighbors,
            "zone_k_neighbors": est.zone_k_neighbors,
            "weights": est.position_weights,
            "standardize_rssi": est.standardize_rssi,
        }
        if not no_sweep:
            sweep = sweep_k_neighbors(train_df, val_df, cfg.environment, cfg, k_max=k_sweep_max)
            metrics["k_sweep"] = sweep
            k_mark = min(cfg.zone_knn.k_neighbors, len(train_df) - 1)
            plot_knn_validation_vs_k(
                sweep["k"],
                sweep["validation_zone_accuracy"],
                sweep["validation_rmse_xy_m"],
                results_dir / "validation_vs_k.png",
                mark_k=k_mark,
            )
    elif model == "rf":
        est = FingerprintRfEstimator.from_config(cfg)
        est.fit(train_df, fit_zone=True)
        metrics["model"] = {
            "type": "random_forest",
            "n_estimators": est.n_estimators,
            "max_depth": est.max_depth,
            "random_state": est.random_state,
            "standardize_rssi": est.standardize_rssi,
        }
    else:
        est = FingerprintMlpEstimator.from_config(cfg)
        est.fit(train_df, fit_zone=True)
        metrics["model"] = {
            "type": "mlp",
            "hidden_layer_sizes": list(est.hidden_layer_sizes),
            "activation": est.activation,
            "learning_rate_init": est.learning_rate_init,
            "max_iter": est.max_iter,
            "random_state": est.random_state,
            "standardize_rssi": est.standardize_rssi,
            "position_n_iter": getattr(est._position, "n_iter_", None),
            "zone_n_iter": getattr(est._zone, "n_iter_", None),
        }

    metrics["train"] = est.evaluate(train_df)
    metrics["validation"] = est.evaluate(val_df)

    model_path = results_dir / "model.joblib"
    metrics_path = results_dir / "metrics.json"
    est.save(model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    write_fingerprint_training_figures(
        cfg.environment,
        cfg,
        train_df,
        val_df,
        est,
        metrics,
        results_dir,
    )

    return FingerprintTrainResult(
        metrics=metrics,
        results_dir=results_dir,
        model_path=model_path,
        metrics_path=metrics_path,
    )


def fingerprint_artifact_paths(results_dir: Path) -> list[Path]:
    """Rutas típicas generadas (las que existan)."""
    names = (
        "metrics_table.png",
        "confusion_zone_train.png",
        "confusion_zone_validation.png",
        "position_validation.png",
        "position_error_validation.png",
        "validation_vs_k.png",
    )
    return [results_dir / n for n in names if (results_dir / n).is_file()]
