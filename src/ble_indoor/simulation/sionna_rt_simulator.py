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

    def _mitsuba_material_id(self) -> str:
        return "mat_wall"

    def _mitsuba_material_block(self) -> str:
        """Standard Mitsuba3 diffuse bsdf (radio properties set programmatically after load)."""
        mat_id = self._mitsuba_material_id()
        return textwrap.dedent(f"""
            <bsdf type="twosided" id="{mat_id}">
              <bsdf type="diffuse">
                <rgb name="reflectance" value="0.5, 0.5, 0.5"/>
              </bsdf>
            </bsdf>""")

    def _assign_radio_material(self, scene: object, rt: object) -> None:
        """Create a RadioMaterial and assign it to every object in the scene."""
        cfg = self._cfg
        if cfg.wall_material.lower() in ("concrete", "itu_concrete"):
            eps, sigma = 5.24, 0.014
        else:
            eps = cfg.wall_permittivity if cfg.wall_permittivity is not None else 5.24
            sigma = cfg.wall_conductivity if cfg.wall_conductivity is not None else 0.014
        mat = rt.RadioMaterial("wall_mat", relative_permittivity=eps, conductivity=sigma)
        scene.add(mat)  # type: ignore[attr-defined]
        for obj in scene.objects.values():  # type: ignore[attr-defined]
            obj.radio_material = "wall_mat"

    @staticmethod
    def _write_rect_ply(path: Path, v0, v1, v2, v3) -> None:
        """Write a double-sided rectangle as two triangles to a PLY file."""
        lines = [
            "ply", "format ascii 1.0",
            "element vertex 4",
            "property float x", "property float y", "property float z",
            "element face 4",
            "property list uchar int vertex_indices",
            "end_header",
            f"{v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}",
            f"{v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}",
            f"{v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}",
            f"{v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}",
            "3 0 1 2", "3 0 2 3",  # front side
            "3 0 2 1", "3 0 3 2",  # back side (for double-sided)
        ]
        path.write_text("\n".join(lines), encoding="utf-8")

    def _build_mitsuba_xml(self, tmp_dir: str) -> str:
        """Write a Mitsuba3 XML scene (closed rectangular room) and return its path.

        Sionna RT coordinate system: Y = vertical axis, room floor in XZ plane.
        Room (x_m, y_m) maps to scene (x, *, y_m); gateways at height gateway_height_m.
        Each surface is a PLY triangle mesh — Sionna RT only supports triangle meshes.
        """
        env = self._env
        w = env.room.width_m
        d = env.room.height_m   # room depth along Z axis in scene
        h = self._cfg.gateway_height_m + 1.0  # ceiling slightly above gateways
        mat_id = self._mitsuba_material_id()
        mat_block = self._mitsuba_material_block()
        td = Path(tmp_dir)

        # Build 6 surfaces as PLY files (vertices in scene XYZ: X=room_x, Y=height, Z=room_y)
        surfaces = {
            "floor":   ((0,0,0), (w,0,0), (w,0,d), (0,0,d)),
            "ceiling": ((0,h,0), (0,h,d), (w,h,d), (w,h,0)),
            "wall_x0": ((0,0,0), (0,0,d), (0,h,d), (0,h,0)),
            "wall_xw": ((w,0,0), (w,h,0), (w,h,d), (w,0,d)),
            "wall_z0": ((0,0,0), (0,h,0), (w,h,0), (w,0,0)),
            "wall_zd": ((0,0,d), (w,0,d), (w,h,d), (0,h,d)),
        }
        shape_blocks = []
        for name, (v0, v1, v2, v3) in surfaces.items():
            ply_path = td / f"{name}.ply"
            self._write_rect_ply(ply_path, v0, v1, v2, v3)
            shape_blocks.append(textwrap.dedent(f"""\
              <shape type="ply" id="{name}">
                <string name="filename" value="{ply_path}"/>
                <ref id="{mat_id}"/>
              </shape>"""))

        xml = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="utf-8"?>
        <scene version="3.0.0">
          {mat_block}
        """) + "\n".join(f"  {b}" for b in shape_blocks) + "\n</scene>\n"

        scene_path = str(td / "room_scene.xml")
        Path(scene_path).write_text(xml, encoding="utf-8")
        return scene_path

    # ------------------------------------------------------------------
    # Sionna RT precompute via coverage_map (all Sionna/TF imports live here)
    # ------------------------------------------------------------------

    def _run_sionna_precompute(self) -> tuple[np.ndarray, np.ndarray]:
        """Run Sionna RT coverage_map for all gateways; return (grid_xy, grid_rssi_dbm).

        Uses scene.coverage_map() which evaluates path gain over a regular 2D grid
        at a fixed receiver height — the natural API for this use case in Sionna 0.19.
        All Sionna/TF imports are confined here.
        """
        import sionna.rt as rt

        env = self._env
        cfg = self._cfg
        res = cfg.grid_resolution_m
        room_w = env.room.width_m
        room_d = env.room.height_m  # room depth → Sionna Z axis

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_path = self._build_mitsuba_xml(tmp_dir)
            scene = rt.load_scene(scene_path)
            scene.frequency = cfg.carrier_frequency_hz

            # Assign EM material properties to all scene geometry
            self._assign_radio_material(scene, rt)

            # Single isotropic antenna for TX and RX (BLE badge approximation)
            iso_array = rt.PlanarArray(
                num_rows=1, num_cols=1,
                vertical_spacing=0.5, horizontal_spacing=0.5,
                pattern="iso", polarization="V",
            )
            scene.tx_array = iso_array
            scene.rx_array = iso_array

            # Add one Transmitter per gateway
            for gw in env.gateways:
                scene.add(rt.Transmitter(
                    name=gw.id,
                    position=[gw.x_m, cfg.gateway_height_m, gw.y_m],
                ))

            n_gw = len(env.gateways)
            print(
                f"[SionnaRTSimulator] Computing coverage map "
                f"({room_w}×{room_d} m, res={res} m, {cfg.num_samples} rays, "
                f"{n_gw} gateways)…"
            )

            # coverage_map evaluates ALL transmitters simultaneously.
            # Sionna coords: Y = vertical axis, room floor = XZ plane.
            # cm_orientation=[0, 0, π/2] rotates the CM 90° around X so that
            # its local Y axis aligns with scene Z (room depth), placing the
            # CM in the horizontal XZ plane at height rx_height_m.
            # cm_size = [room_w, room_d] → spans the full room floor.
            import math
            cm = scene.coverage_map(
                rx_orientation=(0.0, 0.0, 0.0),
                max_depth=cfg.max_depth,
                cm_center=[room_w / 2.0, cfg.rx_height_m, room_d / 2.0],
                cm_orientation=[0.0, 0.0, math.pi / 2.0],
                cm_size=[room_w, room_d],
                cm_cell_size=[res, res],
                num_samples=cfg.num_samples,
                los=True,
                reflection=True,
                diffraction=False,
                scattering=False,
            )

        # path_gain: [n_tx, n_cells_x, n_cells_y]  (linear power ratio)
        # cell_centers: [n_cells_x, n_cells_y, 3]  (x, y_height, z in scene coords)
        path_gain_np = cm.path_gain.numpy()     # (n_tx, nx, ny)
        cell_centers = cm.cell_centers.numpy()  # (nx, ny, 3)

        nx, ny = cell_centers.shape[:2]
        n_points = nx * ny

        # Convert scene (x, height, z) → room (x, y): x_room=x_scene, y_room=z_scene
        grid_xy = np.column_stack([
            cell_centers[:, :, 0].ravel(),   # x_room
            cell_centers[:, :, 2].ravel(),   # y_room
        ])  # (N, 2)

        # Calibrate TX power so Sionna RSSI aligns with path-loss convention.
        # In path-loss models, tx_power_dbm is the RSSI at 1 m (reference RSSI), not
        # the radiated power. Sionna uses the actual radiated power. To recover the
        # same reference: actual_tx_dbm = ref_rssi_1m + FSPL(1m, f).
        # FSPL at 1 m = 20*log10(4π*f/c)  [dB]
        _c = 3e8
        _fspl_1m = 20.0 * np.log10(4.0 * np.pi * cfg.carrier_frequency_hz / _c)

        # Convert linear path gain to RSSI dBm per gateway
        rssi_grid = np.full((n_points, n_gw), -200.0, dtype=np.float64)
        for gw_idx, gw in enumerate(env.gateways):
            actual_tx_dbm = gw.tx_power_dbm + _fspl_1m
            gains = path_gain_np[gw_idx].ravel().astype(np.float64)
            valid = gains > 1e-30
            rssi_grid[valid, gw_idx] = 10.0 * np.log10(gains[valid]) + actual_tx_dbm

        valid_mask = rssi_grid > -200
        print(
            f"[SionnaRTSimulator] Done. Grid: {nx}×{ny}={n_points} cells. "
            f"RSSI range: {rssi_grid[valid_mask].min():.1f}…{rssi_grid[valid_mask].max():.1f} dBm"
        )
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
