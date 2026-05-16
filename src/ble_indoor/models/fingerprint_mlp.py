"""MLP fingerprint localizer: RSSI → (x, y) and zone id.

Uses sklearn MLPRegressor/MLPClassifier (no extra dependencies). Optional
``export_tflite()`` bakes the trained weights + scaler into a TFLite flatbuffer
ready for deployment on Nordic / Cortex-M4 via TFLite Micro.

Requires tensorflow only for export (pip install -r requirements-sionna.txt).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from ble_indoor.domain.environment import Environment
from ble_indoor.evaluation.fingerprint_eval import evaluate_predictions
from ble_indoor.models.features import position_matrix, rssi_feature_matrix
from ble_indoor.models.knn_zone import ZONE_ID_COLUMN
from ble_indoor.settings import ProjectConfig

_STATE_VERSION = 1

# sklearn → Keras activation name mapping
_ACTIVATION_MAP = {
    "relu": "relu",
    "tanh": "tanh",
    "logistic": "sigmoid",
    "identity": "linear",
}


@dataclass
class FingerprintMlpEstimator:
    """Two-headed MLP on scaled RSSI features: position regressor + zone classifier.

    Architecture: Input → [StandardScaler] → hidden layers (ReLU) → output.
    The scaler is baked into the TFLite export so the badge receives raw RSSI.
    """

    environment: Environment
    hidden_layer_sizes: tuple[int, ...]
    activation: str
    learning_rate_init: float
    max_iter: int
    random_state: int
    standardize_rssi: bool
    _scaler: StandardScaler | None = field(default=None, repr=False)
    _position: MLPRegressor | None = field(default=None, repr=False)
    _zone: MLPClassifier | None = field(default=None, repr=False)

    @classmethod
    def from_config(cls, config: ProjectConfig) -> FingerprintMlpEstimator:
        mlp = config.fingerprint_mlp
        return cls(
            environment=config.environment,
            hidden_layer_sizes=tuple(mlp.hidden_layer_sizes),
            activation=mlp.activation,
            learning_rate_init=mlp.learning_rate_init,
            max_iter=mlp.max_iter,
            random_state=mlp.random_state,
            standardize_rssi=mlp.standardize_rssi,
        )

    @property
    def fitted_zone(self) -> bool:
        return self._zone is not None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, *, fit_zone: bool = True) -> None:
        if len(train_df) < 2:
            raise ValueError("Training dataframe must have at least 2 rows.")

        X = rssi_feature_matrix(train_df, self.environment)
        y_xy = position_matrix(train_df)

        if self.standardize_rssi:
            self._scaler = StandardScaler()
            Xf = self._scaler.fit_transform(X).astype(np.float64)
        else:
            self._scaler = None
            Xf = X

        shared_kw: dict[str, Any] = dict(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=40,
        )

        self._position = MLPRegressor(**shared_kw, random_state=self.random_state)
        self._position.fit(Xf, y_xy)

        self._zone = None
        if fit_zone and ZONE_ID_COLUMN in train_df.columns:
            y_z = train_df[ZONE_ID_COLUMN].to_numpy(dtype=np.int64)
            self._zone = MLPClassifier(**shared_kw, random_state=self.random_state + 1)
            self._zone.fit(Xf, y_z)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self._scaler.transform(X) if self._scaler is not None else X

    def predict_xy(self, rssi_dbm: np.ndarray) -> np.ndarray:
        return self.predict_xy_batch(
            np.asarray(rssi_dbm, dtype=np.float64).reshape(1, -1)
        ).reshape(2,)

    def predict_xy_batch(self, rssi_dbm: np.ndarray) -> np.ndarray:
        if self._position is None:
            raise RuntimeError("Call fit() before predict_xy_batch.")
        return np.asarray(
            self._position.predict(self._transform(rssi_dbm)), dtype=np.float64
        )

    def predict_zone_id(self, rssi_dbm: np.ndarray) -> int:
        return int(
            self.predict_zone_batch(
                np.asarray(rssi_dbm, dtype=np.float64).reshape(1, -1)
            )[0]
        )

    def predict_zone_batch(self, rssi_dbm: np.ndarray) -> np.ndarray:
        if self._zone is None:
            raise RuntimeError("Zone model was not fitted.")
        return np.asarray(
            self._zone.predict(self._transform(rssi_dbm)), dtype=np.int64
        )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self, df: pd.DataFrame) -> dict[str, Any]:
        if self._position is None:
            raise RuntimeError("Call fit() before evaluate().")
        X = rssi_feature_matrix(df, self.environment)
        pred_xy = self.predict_xy_batch(X)
        pred_z = self.predict_zone_batch(X) if self._zone is not None else None
        return evaluate_predictions(df, pred_xy, pred_z)

    # ------------------------------------------------------------------
    # TFLite export
    # ------------------------------------------------------------------

    def export_tflite(
        self,
        path: str | Path,
        *,
        quantize: bool = True,
        calibration_data: np.ndarray | None = None,
    ) -> Path:
        """Export the position MLP to a TFLite flatbuffer.

        The StandardScaler is baked into the model as a Normalization layer, so
        the badge feeds raw RSSI values directly with no preprocessing.

        Args:
            path: output .tflite file path.
            quantize: INT8 post-training quantization (default True). Reduces
                model size ~4× with minimal accuracy loss.
            calibration_data: float32 array (N, n_features) of representative
                RSSI inputs for INT8 calibration. If None, 256 uniform random
                samples in [-110, -30] dBm are used.

        Requires tensorflow (pip install -r requirements-sionna.txt).
        """
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "TFLite export requires tensorflow.\n"
                "  pip install -r requirements-sionna.txt"
            ) from exc

        if self._position is None:
            raise RuntimeError("Call fit() before export_tflite.")

        sk = self._position
        n_features = len(self.environment.gateways)
        act = _ACTIVATION_MAP.get(sk.activation, "relu")

        # Build Keras Sequential: optional normalization + dense layers
        keras_layers: list[Any] = []

        if self._scaler is not None:
            keras_layers.append(
                tf.keras.layers.Normalization(
                    axis=-1,
                    mean=self._scaler.mean_.tolist(),
                    variance=self._scaler.var_.tolist(),
                )
            )

        for i, (W, b) in enumerate(zip(sk.coefs_, sk.intercepts_)):
            is_output = i == len(sk.coefs_) - 1
            keras_layers.append(
                tf.keras.layers.Dense(
                    W.shape[1],
                    activation="linear" if is_output else act,
                    use_bias=True,
                    name=f"output" if is_output else f"hidden_{i}",
                )
            )

        model = tf.keras.Sequential(keras_layers, name="mlp_position")
        model.build(input_shape=(None, n_features))

        # Transfer sklearn weights into Keras dense layers
        dense_idx = 0
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                W = sk.coefs_[dense_idx].astype(np.float32)
                b_vec = sk.intercepts_[dense_idx].astype(np.float32)
                layer.set_weights([W, b_vec])
                dense_idx += 1

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            if calibration_data is None:
                rng = np.random.default_rng(0)
                calibration_data = rng.uniform(
                    -110, -30, (256, n_features)
                ).astype(np.float32)
            calib = calibration_data.astype(np.float32)

            def _representative_dataset():
                for sample in calib:
                    yield [sample.reshape(1, -1)]

            converter.representative_dataset = _representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32

        tflite_bytes = converter.convert()

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(tflite_bytes)

        kb = len(tflite_bytes) / 1024
        quant_label = "INT8" if quantize else "float32"
        print(f"[MLP] TFLite exported → {out_path} ({kb:.1f} KB, {quant_label})")
        return out_path

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_type": "mlp",
                "version": _STATE_VERSION,
                "environment": self.environment,
                "hidden_layer_sizes": list(self.hidden_layer_sizes),
                "activation": self.activation,
                "learning_rate_init": self.learning_rate_init,
                "max_iter": self.max_iter,
                "random_state": self.random_state,
                "standardize_rssi": self.standardize_rssi,
                "scaler": self._scaler,
                "position_model": self._position,
                "zone_model": self._zone,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> FingerprintMlpEstimator:
        path = Path(path)
        data = joblib.load(path)
        if not isinstance(data, dict) or data.get("model_type") != "mlp":
            raise ValueError(f"Not an MLP fingerprint model: {path}")
        out = cls(
            environment=data["environment"],
            hidden_layer_sizes=tuple(data["hidden_layer_sizes"]),
            activation=str(data["activation"]),
            learning_rate_init=float(data["learning_rate_init"]),
            max_iter=int(data["max_iter"]),
            random_state=int(data["random_state"]),
            standardize_rssi=bool(data.get("standardize_rssi", True)),
        )
        out._scaler = data.get("scaler")
        out._position = data.get("position_model")
        out._zone = data.get("zone_model")
        return out
