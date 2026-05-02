"""Orchestrates dataset load/generation, holdout split, kNN train/validate, interference runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from ble_indoor.data.builders import (
    GridFingerprintDatasetBuilder,
    TrajectoryFingerprintDatasetBuilder,
    write_csv,
)
from ble_indoor.domain.environment import Environment
from ble_indoor.evaluation.interference import ChannelPerturbation
from ble_indoor.evaluation.metrics import error_summary, position_errors_m
from ble_indoor.models.features import rssi_feature_matrix
from ble_indoor.models.knn_position import KnnFingerprintPositionModel
from ble_indoor.models.knn_zone import ZONE_ID_COLUMN, KnnZoneClassifier
from ble_indoor.settings import ProjectConfig, ProjectLayout
from ble_indoor.simulation.omnet_trace_loader import load_omnet_training_trace
from ble_indoor.simulation.path_loss import PathLossSimulator
from ble_indoor.simulation.ports import RssiObservationSource

DatasetMode = Literal["trajectory", "grid", "omnet_trace"]
Task = Literal["position", "zone"]


@dataclass
class BaselineStudy:
    """Dataset load/build, holdout, kNN models, evaluation."""

    layout: ProjectLayout
    config_path: Path | None = None
    config: ProjectConfig = field(init=False)
    simulator: PathLossSimulator = field(init=False)
    environment: Environment = field(init=False)

    dataset_mode: DatasetMode | None = field(init=False, default=None)
    dataset_df: pd.DataFrame | None = field(init=False, default=None)
    dataset_path: Path | None = field(init=False, default=None)
    _test_positions_grid: pd.DataFrame | None = field(init=False, default=None)
    _omnet_point_cloud: pd.DataFrame | None = field(init=False, default=None)

    train_df: pd.DataFrame | None = field(init=False, default=None)
    test_df: pd.DataFrame | None = field(init=False, default=None)

    position_model: KnnFingerprintPositionModel | None = field(init=False, default=None)
    zone_model: KnnZoneClassifier | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        path = self.config_path or self.layout.default_config_path()
        self.config = ProjectConfig.load(path)
        self.environment = self.config.environment
        self.simulator = PathLossSimulator(self.environment)

    def train_dataset_csv_path(self) -> Path:
        """Path from `omnet.training_trace_csv` in config (resolved to repo root)."""
        rel = self.config.omnet.training_trace_csv
        if rel:
            return self.layout.resolve_repo_path(rel)
        return self.layout.data_simulated_dir() / "trajectory_fingerprints.csv"

    def load_training_trace_from_omnet(
        self,
        csv_path: str | Path,
        *,
        save_copy: bool = True,
    ) -> pd.DataFrame:
        """Load OMNeT-compatible training trace CSV."""
        path = Path(csv_path)
        df = load_omnet_training_trace(path, self.environment, self.config.spatial_zones)
        self.dataset_mode = "omnet_trace"
        self._test_positions_grid = None
        out = self.train_dataset_csv_path()
        if save_copy:
            write_csv(df, out)
        self.dataset_df = df
        self.dataset_path = out
        rssi_cols = ["x_m", "y_m"] + [f"rssi_{g}" for g in self.environment.gateway_ids]
        self._omnet_point_cloud = df[rssi_cols].copy()
        return df

    def load_training_trace_from_config(self, *, strict: bool = False) -> pd.DataFrame | None:
        """Load `omnet.training_trace_csv` if set and file exists; else None. strict raises if missing."""
        rel = self.config.omnet.training_trace_csv
        if not rel:
            return None
        path = self.layout.resolve_repo_path(rel)
        if not path.is_file():
            if strict:
                raise FileNotFoundError(f"omnet.training_trace_csv not found: {path}")
            return None
        return self.load_training_trace_from_omnet(path, save_copy=True)

    def generate_grid_from_omnet_point_cloud(
        self,
        omnet_trace_csv: str | Path,
        *,
        k_interp: int = 10,
        save: bool = True,
    ) -> pd.DataFrame:
        """Fingerprint grid RSSI from OMNeT point cloud via `OmnetTraceRssiSource` interpolation."""
        from ble_indoor.simulation.omnet_interpolated_source import OmnetTraceRssiSource
        from ble_indoor.simulation.omnet_trace_loader import load_omnet_trace_points_only

        pts = load_omnet_trace_points_only(omnet_trace_csv, self.environment)
        self._omnet_point_cloud = pts.copy()
        k = min(int(k_interp), max(3, len(pts)))
        src: RssiObservationSource = OmnetTraceRssiSource(pts, self.environment, k_neighbors=k)
        gb = GridFingerprintDatasetBuilder(self.environment, self.config.fingerprint_grid, src)
        df = gb.build_train_dataframe()
        zids, znames = self.config.spatial_zones.label_xy_batch(
            df["x_m"].to_numpy(dtype=np.float64),
            df["y_m"].to_numpy(dtype=np.float64),
        )
        df[ZONE_ID_COLUMN] = zids
        df["zone_name"] = znames.astype(str)
        self._test_positions_grid = gb.build_test_positions_dataframe()
        self.dataset_mode = "grid"
        out = self.layout.data_simulated_dir() / "fingerprints_train_omnet.csv"
        test_out = self.layout.data_simulated_dir() / "test_positions.csv"
        if save:
            write_csv(self._test_positions_grid, test_out)
            write_csv(df, out)
        self.dataset_df = df
        self.dataset_path = out
        return df

    def generate_trajectory_from_omnet_point_cloud(
        self,
        omnet_trace_csv: str | Path,
        *,
        k_interp: int = 12,
        save: bool = True,
    ) -> pd.DataFrame:
        """Synthetic trajectory with RSSI interpolated from an OMNeT point-cloud CSV."""
        from ble_indoor.simulation.omnet_interpolated_source import OmnetTraceRssiSource
        from ble_indoor.simulation.omnet_trace_loader import load_omnet_trace_points_only

        pts = load_omnet_trace_points_only(omnet_trace_csv, self.environment)
        self._omnet_point_cloud = pts.copy()
        src: RssiObservationSource = OmnetTraceRssiSource(pts, self.environment, k_neighbors=k_interp)
        builder = TrajectoryFingerprintDatasetBuilder(
            self.environment,
            self.config.trajectory,
            self.config.spatial_zones,
            src,
        )
        df = builder.build_dataframe()
        self.dataset_mode = "trajectory"
        self._test_positions_grid = None
        out = self.layout.data_simulated_dir() / "trajectory_interpolated_from_omnet.csv"
        if save:
            write_csv(df, out)
        self.dataset_df = df
        self.dataset_path = out
        return df

    def generate_base_dataset(
        self,
        mode: DatasetMode = "trajectory",
        *,
        save: bool = True,
        rssi_source: RssiObservationSource | None = None,
    ) -> pd.DataFrame:
        """Build trajectory or grid dataset (default RSSI source: path loss simulator)."""
        if mode == "omnet_trace":
            raise ValueError("Use load_training_trace_from_omnet(csv); mode 'omnet_trace' is not valid here.")
        self.dataset_mode = mode
        self._omnet_point_cloud = None
        src = rssi_source if rssi_source is not None else self.simulator
        if mode == "trajectory":
            builder = TrajectoryFingerprintDatasetBuilder(
                self.environment,
                self.config.trajectory,
                self.config.spatial_zones,
                src,
            )
            df = builder.build_dataframe()
            out = self.train_dataset_csv_path()
            self._test_positions_grid = None
        elif mode == "grid":
            gb = GridFingerprintDatasetBuilder(self.environment, self.config.fingerprint_grid, src)
            df = gb.build_train_dataframe()
            self._test_positions_grid = gb.build_test_positions_dataframe()
            out = self.layout.data_simulated_dir() / "fingerprints_train.csv"
            test_out = self.layout.data_simulated_dir() / "test_positions.csv"
            if save:
                write_csv(self._test_positions_grid, test_out)
        else:
            raise ValueError(mode)

        if save:
            write_csv(df, out)
        self.dataset_df = df
        self.dataset_path = out
        return df

    def prepare_holdout(self, test_size: float = 0.25, random_state: int = 123) -> None:
        """Train/test split: random split for trajectory/omnet_trace; grid uses train grid vs offset test positions."""
        if self.dataset_df is None or self.dataset_mode is None:
            raise RuntimeError("Call generate_base_dataset() or load_training_trace_from_omnet() first.")

        if self.dataset_mode in ("trajectory", "omnet_trace"):
            y = self.dataset_df[ZONE_ID_COLUMN].to_numpy(dtype=np.int64)
            counts = np.bincount(y, minlength=self.config.spatial_zones.n_zones)
            strat = y if int(np.min(counts)) >= 5 else None
            self.train_df, self.test_df = train_test_split(
                self.dataset_df,
                test_size=test_size,
                random_state=random_state,
                shuffle=True,
                stratify=strat,
            )
        else:
            self.train_df = self.dataset_df.copy()
            if self._test_positions_grid is None:
                raise RuntimeError("Grid mode requires test positions.")
            self.test_df = self._test_positions_grid.copy()

    def train(self, task: Task = "position") -> None:
        """Fit kNN for `position` or `zone` task."""
        if self.train_df is None:
            raise RuntimeError("Call prepare_holdout() first.")
        env = self.environment
        if task == "position":
            k = min(self.config.baseline_knn.k_neighbors, max(1, len(self.train_df) - 1))
            self.position_model = KnnFingerprintPositionModel(
                k_neighbors=k,
                weights=self.config.baseline_knn.weights,
                standardize_rssi=self.config.baseline_knn.standardize_rssi,
            )
            self.position_model.fit(self.train_df, env)
        elif task == "zone":
            if ZONE_ID_COLUMN not in self.train_df.columns:
                raise RuntimeError(f"Task 'zone' requires column '{ZONE_ID_COLUMN}'.")
            k = min(self.config.zone_knn.k_neighbors, max(1, len(self.train_df) - 1))
            self.zone_model = KnnZoneClassifier(
                k_neighbors=k,
                weights=self.config.zone_knn.weights,
                standardize_rssi=self.config.baseline_knn.standardize_rssi,
            )
            self.zone_model.fit(self.train_df, env)
        else:
            raise ValueError(task)

    def validate(self, task: Task = "position") -> dict[str, Any]:
        """Validation metrics on nominal channel (no extra perturbation unless passed)."""
        return self._evaluate(task, perturbation=None, plot_confusion_path=None)

    def evaluate_interference(
        self,
        perturbation: ChannelPerturbation,
        task: Task = "zone",
        *,
        confusion_png: Path | None = None,
    ) -> dict[str, Any]:
        """Evaluate on test positions with perturbed channel (noise / reception)."""
        results_dir = self.layout.data_results_dir()
        results_dir.mkdir(parents=True, exist_ok=True)
        out = confusion_png or (results_dir / "confusion_interference.png")
        return self._evaluate(task, perturbation=perturbation, plot_confusion_path=out)

    def _evaluate(
        self,
        task: Task,
        perturbation: ChannelPerturbation | None,
        plot_confusion_path: Path | None,
    ) -> dict[str, Any]:
        if self.test_df is None:
            raise RuntimeError("test_df missing; call prepare_holdout().")
        env = self.environment
        rng = np.random.default_rng(7_001)

        if task == "position":
            if self.position_model is None:
                raise RuntimeError("Call train('position') first.")
            true_xy = self.test_df[["x_m", "y_m"]].to_numpy(dtype=np.float64)
            if perturbation is None and self.dataset_mode in ("trajectory", "omnet_trace"):
                cols_ok = all(f"rssi_{g}" in self.test_df.columns for g in env.gateway_ids)
                X = rssi_feature_matrix(self.test_df, env) if cols_ok else self._simulate_rssi_observations(
                    true_xy, rng, None, use_grid_style=False
                )
            elif perturbation is None and self.dataset_mode == "grid":
                X = self._simulate_rssi_observations(true_xy, rng, None, use_grid_style=True)
            else:
                X = self._simulate_rssi_observations(
                    true_xy,
                    rng,
                    perturbation,
                    use_grid_style=(self.dataset_mode == "grid"),
                )
            est_xy = self.position_model.predict_xy_batch(X)
            err = position_errors_m(true_xy, est_xy)
            return {"metrics": error_summary(err), "true_xy_m": true_xy, "est_xy_m": est_xy}

        if self.zone_model is None:
            raise RuntimeError("Call train('zone') first.")
        if ZONE_ID_COLUMN not in self.test_df.columns:
            raise RuntimeError(f"Zone validation requires '{ZONE_ID_COLUMN}' in test_df.")

        y_true = self.test_df[ZONE_ID_COLUMN].to_numpy(dtype=np.int64)
        true_xy = self.test_df[["x_m", "y_m"]].to_numpy(dtype=np.float64)
        if perturbation is None and all(f"rssi_{g}" in self.test_df.columns for g in env.gateway_ids):
            X = rssi_feature_matrix(self.test_df, env)
        else:
            X = self._simulate_rssi_observations(true_xy, rng, perturbation, use_grid_style=False)
        y_pred = self.zone_model.predict_zone_batch(X)
        acc = float(accuracy_score(y_true, y_pred))
        zmap = self.config.spatial_zones
        labels = zmap.zone_labels()
        label_ids = list(range(zmap.n_zones))
        cm = confusion_matrix(y_true, y_pred, labels=label_ids)
        out: dict[str, Any] = {"accuracy": acc, "confusion_matrix": cm.tolist(), "labels": labels}
        if plot_confusion_path is not None:
            from ble_indoor.evaluation.plots import plot_confusion_matrix_counts

            out["confusion_png"] = str(
                plot_confusion_matrix_counts(
                    cm,
                    labels,
                    plot_confusion_path,
                    title="Spatial zones (validation)",
                )
            )
        return out

    def _simulate_rssi_observations(
        self,
        xy_m: np.ndarray,
        rng: np.random.Generator,
        perturbation: ChannelPerturbation | None,
        *,
        use_grid_style: bool,
    ) -> np.ndarray:
        """RSS matrix (N, G) for batch of positions."""
        from ble_indoor.simulation.omnet_interpolated_source import OmnetTraceRssiSource

        env = self.environment
        rows: list[np.ndarray] = []
        base_sigma = env.rssi_model.noise_sigma_db
        traj = self.config.trajectory
        use_omnet_cloud = self._omnet_point_cloud is not None and self._omnet_point_cloud.shape[0] >= 3
        k_cloud = min(24, len(self._omnet_point_cloud)) if use_omnet_cloud else 0
        omnet_src: OmnetTraceRssiSource | None = (
            OmnetTraceRssiSource(self._omnet_point_cloud, env, k_neighbors=k_cloud)
            if use_omnet_cloud and not use_grid_style
            else None
        )
        omnet_src_grid: OmnetTraceRssiSource | None = (
            OmnetTraceRssiSource(self._omnet_point_cloud, env, k_neighbors=k_cloud)
            if use_omnet_cloud and use_grid_style
            else None
        )

        for i in range(xy_m.shape[0]):
            pos = xy_m[i]
            if use_grid_style:
                sigma = base_sigma if perturbation is None else perturbation.effective_noise_sigma_db(base_sigma)
                if omnet_src_grid is not None:
                    rows.append(omnet_src_grid.sample_rssi_dbm(pos, rng, noise_sigma_db=sigma))
                else:
                    rows.append(self.simulator.sample_rssi_dbm(pos, rng, noise_sigma_db=sigma))
                continue

            p0 = traj.gateway_reception_prob
            p_eff = p0 if perturbation is None else perturbation.effective_reception_prob(p0)
            sigma_eff = base_sigma if perturbation is None else perturbation.effective_noise_sigma_db(base_sigma)

            if omnet_src is not None:
                for _ in range(40):
                    rssi, vis = omnet_src.sample_rssi_with_reception(
                        pos,
                        rng,
                        gateway_reception_prob=p_eff,
                        missing_rssi_dbm=traj.missing_rssi_dbm,
                        noise_sigma_db=sigma_eff,
                    )
                    if int(np.sum(vis)) >= traj.min_visible_gateways:
                        rows.append(rssi)
                        break
                else:
                    rows.append(rssi)
                continue

            for _ in range(40):
                rssi, vis = self.simulator.sample_rssi_with_reception(
                    pos,
                    rng,
                    gateway_reception_prob=p_eff,
                    missing_rssi_dbm=traj.missing_rssi_dbm,
                    noise_sigma_db=sigma_eff,
                )
                if int(np.sum(vis)) >= traj.min_visible_gateways:
                    rows.append(rssi)
                    break
            else:
                rows.append(rssi)
        return np.stack(rows, axis=0)
