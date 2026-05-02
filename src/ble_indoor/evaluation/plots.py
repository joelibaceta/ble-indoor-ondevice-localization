from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ble_indoor.domain.environment import Environment


def plot_confusion_matrix_counts(
    cm: np.ndarray,
    labels: list[str],
    out_path: str | Path,
    title: str = "Confusion matrix (counts)",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(labels)
    fig_side = min(10.0, 4.2 + 0.45 * max(0, n - 4))
    fig, ax = plt.subplots(figsize=(fig_side, fig_side))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_ylabel("Truth")
    ax.set_xlabel("Predicted")
    thresh = cm.max() / 2.0 if cm.size else 0.0
    fs = 9 if cm.shape[0] <= 4 else 7
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=fs,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_room_overview(
    env: Environment,
    train_df: pd.DataFrame | None,
    test_true_xy_m: np.ndarray,
    test_est_xy_m: np.ndarray,
    out_path: str | Path,
    title: str = "Ground truth vs estimate",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_aspect("equal", adjustable="box")
    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            env.room.width_m,
            env.room.height_m,
            fill=False,
            edgecolor="black",
            linewidth=1.2,
        )
    )

    gw = env.gateway_positions_m()
    ax.scatter(gw[:, 0], gw[:, 1], c="tab:blue", s=120, zorder=3, label="Gateways")
    for g, (gx, gy) in zip(env.gateways, gw, strict=True):
        ax.annotate(g.id, (gx, gy), textcoords="offset points", xytext=(4, 4), fontsize=9)

    if train_df is not None and not train_df.empty:
        ax.scatter(
            train_df["x_m"],
            train_df["y_m"],
            c="lightgray",
            s=18,
            zorder=1,
            label="Train",
        )

    ax.scatter(test_true_xy_m[:, 0], test_true_xy_m[:, 1], c="tab:green", s=28, zorder=2, label="Truth")
    ax.scatter(
        test_est_xy_m[:, 0],
        test_est_xy_m[:, 1],
        c="tab:red",
        s=22,
        marker="x",
        zorder=2,
        label="Estimate",
    )
    for i in range(test_true_xy_m.shape[0]):
        ax.plot(
            [test_true_xy_m[i, 0], test_est_xy_m[i, 0]],
            [test_true_xy_m[i, 1], test_est_xy_m[i, 1]],
            c="tab:orange",
            alpha=0.45,
            linewidth=1.0,
        )

    ax.set_xlim(-0.5, env.room.width_m + 0.5)
    ax.set_ylim(-0.5, env.room.height_m + 0.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_error_heatmap(
    env: Environment,
    test_true_xy_m: np.ndarray,
    errors_m: np.ndarray,
    out_path: str | Path,
    title: str = "Localization error (m)",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter(
        test_true_xy_m[:, 0],
        test_true_xy_m[:, 1],
        c=errors_m,
        cmap="inferno",
        s=80,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, env.room.width_m)
    ax.set_ylim(0, env.room.height_m)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label="error (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_metrics_comparison_table(
    train_block: dict[str, Any],
    val_block: dict[str, Any],
    out_path: str | Path,
    *,
    title: str = "Train vs validation",
) -> Path:
    """Scalar metrics from `FingerprintKnnEstimator.evaluate` blocks (position + optional zone)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pt, pv = train_block.get("position", {}), val_block.get("position", {})
    zt, zv = train_block.get("zone", {}), val_block.get("zone", {})

    headers = ["Metric", "Train", "Validation"]
    rows: list[list[str]] = [
        ["RMSE xy (m)", f"{pt.get('rmse_xy_m', float('nan')):.4f}", f"{pv.get('rmse_xy_m', float('nan')):.4f}"],
        ["R²", f"{pt.get('r2', float('nan')):.4f}", f"{pv.get('r2', float('nan')):.4f}"],
        ["Mean error (m)", f"{pt.get('mean_m', float('nan')):.4f}", f"{pv.get('mean_m', float('nan')):.4f}"],
        ["Median error (m)", f"{pt.get('median_m', float('nan')):.4f}", f"{pv.get('median_m', float('nan')):.4f}"],
        ["P90 error (m)", f"{pt.get('p90_m', float('nan')):.4f}", f"{pv.get('p90_m', float('nan')):.4f}"],
    ]
    if zt or zv:
        rows.append(
            [
                "Zone accuracy",
                f"{zt.get('accuracy', float('nan')):.4f}" if zt else "—",
                f"{zv.get('accuracy', float('nan')):.4f}" if zv else "—",
            ]
        )

    fig, ax = plt.subplots(figsize=(9, 0.55 + 0.35 * len(rows)))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=12)
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colColours=["#e8e8e8"] * 3,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_knn_validation_vs_k(
    ks: list[int],
    zone_accuracy: list[float],
    rmse_xy_m: list[float],
    out_path: str | Path,
    *,
    mark_k: int | None = None,
    mark_legend: str | None = None,
    title: str = "Validation vs k (kNN has no epochs; k is the hyperparameter)",
) -> Path:
    """Two curves over neighbor count k: zone accuracy and position RMSE on a fixed validation split."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax0.plot(ks, zone_accuracy, "o-", color="tab:blue", markersize=4)
    ax0.set_ylabel("Zone accuracy")
    ax0.set_ylim(0.0, 1.02)
    ax0.grid(True, alpha=0.3)
    ax0.set_title(title, fontsize=10)

    ax1.plot(ks, rmse_xy_m, "o-", color="tab:red", markersize=4)
    ax1.set_xlabel("k (neighbors)")
    ax1.set_ylabel("RMSE xy (m)")
    ax1.grid(True, alpha=0.3)

    if mark_k is not None and mark_k in ks:
        for ax in (ax0, ax1):
            ax.axvline(mark_k, color="gray", linestyle="--", linewidth=1.0)
        leg = mark_legend if mark_legend is not None else f"YAML k={mark_k}"
        fig.text(0.5, 0.02, f"Dashed line: {leg}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
