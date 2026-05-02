# BLE indoor on-device localization

**v0.1.0** — baseline indoor localization from RSSI fingerprints: room geometry, gateways, spatial zones, kNN (position and zone).

## Setup

```bash
pip install -r requirements.txt
```

## Training data

Training rows must follow `simulations/omnet/EXPORT_FORMAT.txt` (columns `x_m`, `y_m`, `rssi_A1`, … matching `config/baseline_room.yaml` gateway ids).

Default path: `data/simulated/omnet_training_trace.csv` (`omnet.training_trace_csv` in `config/baseline_room.yaml`).

## EDA (notebook)

- `notebooks/eda_fingerprint_trace.ipynb` — cobertura espacial, RSSI por gateway, correlación, sentinel vs `n_visible`, RSSI vs distancia a gateways, PCA 2D por zona, y visual del split train/val (misma semilla que el script de entrenamiento).
- `notebooks/compare_fingerprint_models.ipynb` — entrena **kNN** y **RandomForest** en el mismo split, muestra sweep k, mapas de posición, confusiones, importancias (RF) y compara métricas al final.

Abrir con JupyterLab desde la raíz del repo.

## Run baseline

```bash
python scripts/run_baseline.py
```

Optional: `ALLOW_LEGACY_PATHLOSS=1` uses a synthetic trajectory (Python path loss) when no CSV is present.

## Synthetic training CSV (path loss)

```bash
python scripts/generate_pathloss_training_csv.py
```

Writes `data/simulated/omnet_training_trace.csv` (or `omnet.training_trace_csv` in YAML). Use `--force` to replace an existing file.

## Train fingerprint models (kNN or RandomForest)

```bash
python scripts/train_fingerprint_model.py --model knn
python scripts/train_fingerprint_model.py --model rf
```

Writes under **`data/results/knn/`** or **`data/results/rf/`**: `model.joblib`, `metrics.json`, and figures (`metrics_table.png`, `confusion_zone_*.png`, `position_validation.png`, `position_error_validation.png`, and for kNN only `validation_vs_k.png` unless `--no-sweep`). kNN hyperparameters come from `baseline` / `zone_knn` in `config/baseline_room.yaml`; RandomForest from `fingerprint_rf`.

## Package

`ble_indoor`: `BaselineStudy`, `FingerprintKnnEstimator`, `FingerprintRfEstimator`, `ProjectLayout`, `ProjectConfig`, `ChannelPerturbation`. Entrenamiento fingerprint: `ble_indoor.train.train_fingerprint_model` (el script `scripts/train_fingerprint_model.py` solo es CLI).
