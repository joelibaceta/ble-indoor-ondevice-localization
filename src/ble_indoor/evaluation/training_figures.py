"""Figures for fingerprint kNN training reports (confusion, maps, metrics table)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from ble_indoor.domain.environment import Environment
from ble_indoor.evaluation.metrics import position_errors_m
from ble_indoor.evaluation.plots import (
    plot_confusion_matrix_counts,
    plot_error_heatmap,
    plot_metrics_comparison_table,
    plot_room_overview,
)
from ble_indoor.models.features import position_matrix, rssi_feature_matrix
from ble_indoor.models.fingerprint_knn import FingerprintKnnEstimator
from ble_indoor.models.fingerprint_rf import FingerprintRfEstimator
from ble_indoor.models.knn_zone import ZONE_ID_COLUMN
from ble_indoor.settings import ProjectConfig


def write_fingerprint_training_figures(
    env: Environment,
    cfg: ProjectConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    est: FingerprintKnnEstimator | FingerprintRfEstimator,
    metrics: dict,
    out_dir: str | Path,
) -> list[Path]:
    """Write PNGs under ``out_dir``; requires Matplotlib (non-interactive backend set by caller)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    paths.append(
        plot_metrics_comparison_table(
            metrics["train"],
            metrics["validation"],
            out_dir / "metrics_table.png",
            title="Train vs validation",
        )
    )

    zmap = cfg.spatial_zones
    labels = zmap.zone_labels()
    label_ids = list(range(zmap.n_zones))
    if est.fitted_zone:
        for split_name, split_df in (("train", train_df), ("validation", val_df)):
            if ZONE_ID_COLUMN not in split_df.columns:
                continue
            X = rssi_feature_matrix(split_df, env)
            y_t = split_df[ZONE_ID_COLUMN].to_numpy(dtype=np.int64)
            y_p = est.predict_zone_batch(X)
            cm = confusion_matrix(y_t, y_p, labels=label_ids)
            p = out_dir / f"confusion_zone_{split_name}.png"
            plot_confusion_matrix_counts(
                cm,
                list(labels),
                p,
                title=f"Zone confusion ({split_name})",
            )
            paths.append(p)

    Xv = rssi_feature_matrix(val_df, env)
    true_xy = position_matrix(val_df)
    pred_xy = est.predict_xy_batch(Xv)
    err = position_errors_m(true_xy, pred_xy)
    paths.append(
        plot_room_overview(
            env,
            train_df,
            true_xy,
            pred_xy,
            out_dir / "position_validation.png",
            title="Validation: ground truth vs estimate",
        )
    )
    paths.append(
        plot_error_heatmap(
            env,
            true_xy,
            err,
            out_dir / "position_error_validation.png",
            title="Validation: localization error (m)",
        )
    )
    return paths
