"""Sionna RT ray-tracing RSSI simulator (lazy import). Implements RssiObservationSource.

Install dependencies before use:
    pip install -r requirements-sionna.txt   # CPU
    pip install -r requirements-sionna.txt tensorflow[and-cuda]  # GPU

Workflow:
  1. Precomputes RSSI values over a fine grid by running Sionna RT once.
  2. Caches the grid to a .npz file (skips recomputation on subsequent runs).
  3. Uses KD-tree interpolation for fast per-position queries at inference time.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import textwrap
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ble_indoor.domain.environment import Environment
from ble_indoor.settings import SionnaRTSettings


class SionnaRTSimulator:
    """Ray-tracing RSSI simulator backed by Sionna RT.

    Sionna and TensorFlow are imported lazily — non-Sionna users pay no import cost.
    """

    def __init__(
        self,
        environment: Environment,
        settings: SionnaRTSettings,
        *,
        layout_root: Path | None = None,
    ) -> None:
        self._env = environment
        self._cfg = settings
        self._root = layout_root or Path(".")
        self._grid_xy: np.ndarray | None = None    # (N, 2) receiver positions
        self._grid_rssi: np.ndarray | None = None  # (N, G) dBm values
        self._tree: cKDTree | None = None
        self._precompute()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _config_hash(self) -> str:
        env = self._env
        cfg = self._cfg
        meta = {
            "carrier_hz": cfg.carrier_frequency_hz,
            "max_depth": cfg.max_depth,
            "num_samples": cfg.num_samples,
            "grid_res": cfg.grid_resolution_m,
            "gw_height": cfg.gateway_height_m,
            "rx_height": cfg.rx_height_m,
            "wall_material": cfg.wall_material,
            "wall_conductivity": cfg.wall_conductivity,
            "wall_permittivity": cfg.wall_permittivity,
            "room_w": env.room.width_m,
            "room_h": env.room.height_m,
            "gateways": [(g.id, g.x_m, g.y_m, g.tx_power_dbm) for g in env.gateways],
        }
        return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:16]

    def _cache_path(self) -> Path | None:
        if not self._cfg.cache_file:
            return None
        p = Path(self._cfg.cache_file)
        return p if p.is_absolute() else (self._root / p)

    def _try_load_cache(self) -> bool:
        p = self._cache_path()
        if p is None or not p.is_file():
            return False
        try:
            data = np.load(p, allow_pickle=False)
            if data["config_hash"].item().decode() != self._config_hash():
                return False
            self._grid_xy = data["grid_xy"]
            self._grid_rssi = data["grid_rssi"]
            self._tree = cKDTree(self._grid_xy)
            print(f"[SionnaRTSimulator] Loaded precomputed grid from {p}")
            return True
        except Exception:
            return False

    def _save_cache(self) -> None:
        p = self._cache_path()
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            grid_xy=self._grid_xy,
            grid_rssi=self._grid_rssi,
            config_hash=np.bytes_(self._config_hash()),
        )
        print(f"[SionnaRTSimulator] Cached precomputed grid to {p}")

    # ------------------------------------------------------------------
    # Grid / scene helpers
    # ------------------------------------------------------------------

    def _build_grid_xy(self) -> np.ndarray:
        """Regular 2D grid of RX positions within the room boundaries."""
        res = self._cfg.grid_resolution_m
        xs = np.arange(res / 2, self._env.room.width_m, res, dtype=np.float64)
        ys = np.arange(res / 2, self._env.room.height_m, res, dtype=np.float64)
        grid = np.array(np.meshgrid(xs, ys, indexing="ij")).reshape(2, -1).T
        return grid

    def _mitsuba_material_id(self) -> str:
        return "mat_wall"

    def _mitsuba_material_block(self) -> str:
        """Sionna RT radio_material block for the scene XML."""
        mat = self._cfg.wall_material.lower()
        mat_id = self._mitsuba_material_id()
        if mat in ("concrete", "itu_concrete"):
            # ITU-R P.2040 concrete at 2.4 GHz
            eps, sigma = 5.24, 0.014
        else:
            eps = self._cfg.wall_permittivity if self._cfg.wall_permittivity is not None else 5.24
            sigma = self._cfg.wall_conductivity if self._cfg.wall_conductivity is not None else 0.014
        return textwrap.dedent(f"""
            <bsdf type="radio_material" id="{mat_id}">
              <float name="relative_permittivity" value="{eps}"/>
              <float name="conductivity" value="{sigma}"/>
            </bsdf>""")

    def _build_mitsuba_xml(self, tmp_dir: str) -> str:
        """Write a Mitsuba3 XML scene (closed rectangular room) and return its path.

        Sionna RT coordinate system: Y = vertical axis, room floor in XZ plane.
        Room (x_m, y_m) maps to scene (x, 0, y_m); gateways at height gateway_height_m.
        """
        env = self._env
        w = env.room.width_m
        d = env.room.height_m  # room depth along Z axis
        h = self._cfg.gateway_height_m + 1.0  # ceiling slightly above gateways
        mat_id = self._mitsuba_material_id()
        mat_block = self._mitsuba_material_block()

        # 6 rectangle shapes: floor, ceiling, wall -X, wall +X, wall -Z, wall +Z.
        # Each rectangle is a unit square in the XZ plane, scaled and translated.
        xml = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="utf-8"?>
        <scene version="3.0.0">
          {mat_block}
          <!-- Floor (y=0) -->
          <shape type="rectangle">
            <transform name="to_world">
              <scale x="{w / 2:.4f}" y="1" z="{d / 2:.4f}"/>
              <translate x="{w / 2:.4f}" y="0" z="{d / 2:.4f}"/>
              <rotate x="1" angle="-90"/>
            </transform>
            <ref id="{mat_id}"/>
          </shape>
          <!-- Ceiling (y=h) -->
          <shape type="rectangle">
            <transform name="to_world">
              <scale x="{w / 2:.4f}" y="1" z="{d / 2:.4f}"/>
              <translate x="{w / 2:.4f}" y="{h:.4f}" z="{d / 2:.4f}"/>
              <rotate x="1" angle="90"/>
            </transform>
            <ref id="{mat_id}"/>
          </shape>
          <!-- Wall x=0 -->
          <shape type="rectangle">
            <transform name="to_world">
              <scale x="{d / 2:.4f}" y="{h / 2:.4f}" z="1"/>
              <translate x="0" y="{h / 2:.4f}" z="{d / 2:.4f}"/>
              <rotate y="1" angle="90"/>
            </transform>
            <ref id="{mat_id}"/>
          </shape>
          <!-- Wall x=w -->
          <shape type="rectangle">
            <transform name="to_world">
              <scale x="{d / 2:.4f}" y="{h / 2:.4f}" z="1"/>
              <translate x="{w:.4f}" y="{h / 2:.4f}" z="{d / 2:.4f}"/>
              <rotate y="1" angle="-90"/>
            </transform>
            <ref id="{mat_id}"/>
          </shape>
          <!-- Wall z=0 -->
          <shape type="rectangle">
            <transform name="to_world">
              <scale x="{w / 2:.4f}" y="{h / 2:.4f}" z="1"/>
              <translate x="{w / 2:.4f}" y="{h / 2:.4f}" z="0"/>
              <rotate y="1" angle="180"/>
            </transform>
            <ref id="{mat_id}"/>
          </shape>
          <!-- Wall z=d -->
          <shape type="rectangle">
            <transform name="to_world">
              <scale x="{w / 2:.4f}" y="{h / 2:.4f}" z="1"/>
              <translate x="{w / 2:.4f}" y="{h / 2:.4f}" z="{d:.4f}"/>
            </transform>
            <ref id="{mat_id}"/>
          </shape>
        </scene>
        """)

        scene_path = str(Path(tmp_dir) / "room_scene.xml")
        Path(scene_path).write_text(xml, encoding="utf-8")
        return scene_path

    # ------------------------------------------------------------------
    # Sionna RT path computation (all Sionna/TF imports live here)
    # ------------------------------------------------------------------

    def _compute_path_gains(
        self,
        scene: object,
        rx_positions: np.ndarray,
    ) -> np.ndarray:
        """Run Sionna RT path solver; return linear path gains (N_rx, N_tx).

        All Sionna API surface is isolated in this method so version changes
        only require updating here.
        """
        import tensorflow as tf
        import sionna.rt as rt

        rx_pos_tf = tf.constant(rx_positions, dtype=tf.float32)

        solver = rt.PathSolver()
        paths = solver(
            scene,
            rx_pos_tf,
            max_depth=self._cfg.max_depth,
            num_samples=self._cfg.num_samples,
            los=True,
            reflection=True,
            diffraction=False,
            scattering=False,
        )
        # CIR: a shape varies by Sionna version; we reduce over all non-batch dims.
        a, _ = paths.cir()
        a_np = a.numpy() if hasattr(a, "numpy") else np.array(a)
        # Sum |a_i|^2 over all paths and antenna dims except (n_rx, n_tx)
        # Typical shape: [n_rx, n_tx, n_rx_ant, n_tx_ant, n_clusters, n_paths] (complex)
        sq = np.abs(a_np) ** 2
        # Collapse all dims after first two
        path_gains = sq.reshape(sq.shape[0], sq.shape[1], -1).sum(axis=-1)  # (n_rx, n_tx)
        return path_gains.astype(np.float64)

    def _run_sionna_precompute(self) -> tuple[np.ndarray, np.ndarray]:
        """Run full Sionna RT precomputation; return (grid_xy, grid_rssi_dbm)."""
        import sionna.rt as rt

        grid_xy = self._build_grid_xy()
        n_points = len(grid_xy)
        n_gateways = len(self._env.gateways)
        rx_h = self._cfg.rx_height_m

        # Build 3D receiver positions: (n_rx, 3) in scene coords (x, rx_height, z)
        rx_positions = np.column_stack([
            grid_xy[:, 0],
            np.full(n_points, rx_h, dtype=np.float64),
            grid_xy[:, 1],
        ]).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_path = self._build_mitsuba_xml(tmp_dir)
            scene = rt.load_scene(scene_path)
            scene.frequency = self._cfg.carrier_frequency_hz

            for gw in self._env.gateways:
                tx = rt.Transmitter(
                    name=gw.id,
                    position=[gw.x_m, self._cfg.gateway_height_m, gw.y_m],
                )
                scene.add(tx)

            print(f"[SionnaRTSimulator] Precomputing {n_points} RX positions × {n_gateways} gateways...")
            path_gains = self._compute_path_gains(scene, rx_positions)  # (n_rx, n_tx)

        # Convert linear path gain to dBm
        rssi_grid = np.full((n_points, n_gateways), -200.0, dtype=np.float64)
        for gw_idx, gw in enumerate(self._env.gateways):
            gains = path_gains[:, gw_idx]
            valid = gains > 1e-30
            rssi_grid[valid, gw_idx] = 10.0 * np.log10(gains[valid]) + gw.tx_power_dbm

        print(f"[SionnaRTSimulator] Precompute complete. RSSI range: "
              f"{rssi_grid[rssi_grid > -200].min():.1f} … {rssi_grid.max():.1f} dBm")
        return grid_xy, rssi_grid

    def _precompute(self) -> None:
        if self._try_load_cache():
            return
        self._grid_xy, self._grid_rssi = self._run_sionna_precompute()
        self._tree = cKDTree(self._grid_xy)
        self._save_cache()

    # ------------------------------------------------------------------
    # RssiObservationSource protocol
    # ------------------------------------------------------------------

    def _interp(self, position_m: np.ndarray, k: int = 4) -> np.ndarray:
        pos = np.asarray(position_m, dtype=np.float64).reshape(2,)
        k_eff = min(k, len(self._grid_xy))
        dists, idx = self._tree.query(pos, k=k_eff)
        if np.isscalar(idx):
            idx = np.array([int(idx)])
            dists = np.array([float(dists)])
        w = 1.0 / (dists + 1e-6)
        w /= w.sum()
        return (w @ self._grid_rssi[idx]).astype(np.float64)

    def mean_rssi_dbm(self, position_m: np.ndarray) -> np.ndarray:
        return self._interp(position_m)

    def sample_rssi_dbm(
        self,
        position_m: np.ndarray,
        rng: np.random.Generator,
        *,
        noise_sigma_db: float | None = None,
    ) -> np.ndarray:
        mean = self._interp(position_m)
        sigma = (
            self._env.rssi_model.noise_sigma_db
            if noise_sigma_db is None
            else float(noise_sigma_db)
        )
        return mean + rng.normal(loc=0.0, scale=sigma, size=mean.shape)

    def sample_rssi_with_reception(
        self,
        position_m: np.ndarray,
        rng: np.random.Generator,
        *,
        gateway_reception_prob: float,
        missing_rssi_dbm: float,
        noise_sigma_db: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        noisy = self.sample_rssi_dbm(position_m, rng, noise_sigma_db=noise_sigma_db)
        visible = rng.random(size=noisy.shape) < gateway_reception_prob
        out = np.where(visible, noisy, float(missing_rssi_dbm))
        return out, visible
