# BLE Indoor On-Device Localization

Prototipo de investigación para **localización indoor BLE en el dispositivo** mediante fingerprinting de RSSI. El sistema entrena modelos compactos kNN / Random Forest sobre observaciones RSSI simuladas y evalúa su precisión y robustez, con el objetivo de ejecutar inferencia directamente en un badge Nordic con recursos limitados — sin conectividad al backend.

---

## Tabla de contenidos

- [Motivación](#motivación)
- [Arquitectura](#arquitectura)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Inicio rápido](#inicio-rápido)
- [Referencia CLI](#referencia-cli)
- [Configuración](#configuración)
- [Simuladores RSSI](#simuladores-rssi)
- [Modelos](#modelos)
- [Evaluación](#evaluación)
- [Notebooks](#notebooks)
- [Experimentos](#experimentos)

---

## Motivación

Los pipelines tradicionales de localización BLE requieren que los badges suban continuamente mediciones RSSI crudas a un servidor central para calcular la posición. Esto introduce latencia, dependencia de infraestructura y problemas de privacidad.

Este proyecto investiga si un **modelo de fingerprinting compacto ejecutado en el badge** puede lograr precisión de localización aceptable, permitiendo:

- Estimación de posición autónoma sin conectividad de red
- Reducción del tráfico BLE (transmitir posición, no RSSI en crudo)
- Baselines desplegables para clasificación de zonas y estimación continua de posición

Se comparan dos fuentes de datos de entrenamiento — un modelo analítico de path loss y un simulador de ray tracing físico (Sionna RT) — y se evalúan modelos kNN y Random Forest bajo condiciones de canal nominales y degradadas.

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                  Definición del entorno                      │
│        Habitación (12×8 m) · 4 gateways · modelo RSSI       │
└───────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │      Capa de simulación    │
              ├────────────────────────────┤
              │  PathLossSimulator         │  ← analítico, instantáneo
              │  SionnaRTSimulator         │  ← ray tracing (con caché)
              └─────────────┬─────────────┘
                            │
              ┌─────────────┴─────────────┐
              │    Generación de dataset   │
              ├────────────────────────────┤
              │  Grid fingerprints         │  ← grilla regular, N muestras/punto
              │  Trayectorias              │  ← random-walk, RX estocástico
              └─────────────┬─────────────┘
                            │
              ┌─────────────┴─────────────┐
              │    Split train / test      │  estratificado por zona
              └─────────────┬─────────────┘
                            │
              ┌─────────────┴─────────────┐
              │     Entrenamiento          │
              ├────────────────────────────┤
              │  kNN  → posición (x, y)    │
              │  kNN  → clasificación zona │
              │  RF   → posición + zona    │
              └─────────────┬─────────────┘
                            │
              ┌─────────────┴─────────────┐
              │         Evaluación         │
              ├────────────────────────────┤
              │  Métricas canal nominal    │
              │  Robustez a interferencia  │  ← escala de ruido, pérdida de paquetes
              └────────────────────────────┘
```

### Layout de la habitación

```
(0,8) A3 ·─────────────────────· A4 (12,8)
          │                     │
          │    12 m × 8 m       │
          │    6 zonas (3×2)    │
          │                     │
(0,0) A1 ·─────────────────────· A2 (12,0)
```

Cuatro gateways en las esquinas, cada uno transmitiendo a −59 dBm. La habitación se divide en una grilla uniforme de 3×2 zonas (6 celdas de 4×4 m cada una).

---

## Estructura del proyecto

```
.
├── config/
│   └── baseline_room.yaml          # Habitación, gateways y parámetros del modelo
├── data/
│   ├── simulated/                  # CSVs generados y caché Sionna RT (.npz)
│   └── results/                    # Artefactos del modelo entrenado y métricas
├── experiments/
│   ├── configs/                    # YAMLs de variantes de experimento
│   └── sweep.py                    # Runner de experimentos en lote
├── notebooks/
│   ├── dataset_building.ipynb      # Generación y exploración del dataset
│   ├── eda_fingerprint.ipynb       # EDA de fingerprints RSSI
│   └── fingerprint_models.ipynb    # Entrenamiento y comparación de modelos
├── simulations/
│   └── examples/                   # CSVs de ejemplo para pruebas
├── src/ble_indoor/
│   ├── domain/
│   │   ├── environment.py          # Room, Gateway, RssiModelParams
│   │   └── zones.py                # SpatialZoneMap (grilla uniforme)
│   ├── simulation/
│   │   ├── ports.py                # Protocolo RssiObservationSource
│   │   ├── path_loss.py            # PathLossSimulator (analítico)
│   │   ├── sionna_rt_simulator.py  # SionnaRTSimulator (ray tracing)
│   │   └── trace_loader.py         # Utilidades de parseo CSV
│   ├── data/
│   │   └── builders.py             # Constructores de dataset (grilla y trayectoria)
│   ├── models/
│   │   ├── fingerprint_knn.py      # FingerprintKnnEstimator
│   │   ├── fingerprint_rf.py       # FingerprintRfEstimator
│   │   ├── knn_position.py         # KnnFingerprintPositionModel
│   │   ├── knn_zone.py             # KnnZoneClassifier
│   │   └── features.py             # rssi_feature_matrix, position_matrix
│   ├── pipelines/
│   │   └── baseline_study.py       # Orquestador BaselineStudy
│   ├── train/
│   │   └── fingerprint.py          # train_fingerprint_model()
│   ├── evaluation/
│   │   ├── metrics.py              # position_errors_m, error_summary
│   │   └── interference.py         # ChannelPerturbation
│   ├── settings.py                 # Dataclasses de configuración desde YAML
│   └── __main__.py                 # Punto de entrada CLI
├── requirements.txt                # Dependencias principales
└── requirements-sionna.txt         # Opcional: Sionna RT + TensorFlow
```

---

## Instalación

Requiere **Python 3.11+**.

```bash
# Clonar y crear entorno virtual
git clone <repo-url>
cd ble-indoor-ondevice-localization
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Dependencias principales
pip install -r requirements.txt

# Opcional: simulador Sionna RT con ray tracing
pip install -r requirements-sionna.txt
```

> **Apple Silicon (arm64):** `requirements-sionna.txt` instala `tensorflow-macos==2.13.0` automáticamente via marcadores PEP 508. Linux/Windows reciben `tensorflow-cpu`.

---

## Inicio rápido

### 1. Generar datos de entrenamiento (path loss analítico, rápido)

```bash
PYTHONPATH=src python -m ble_indoor generate-csv
```

### 2. Generar datos de entrenamiento (Sionna RT ray tracing, alta fidelidad)

```bash
PYTHONPATH=src python -m ble_indoor generate-csv --simulator sionna --force
```

El primer run precomputa una grilla RSSI (1.536 puntos × 4 gateways a resolución 0.25 m) y la guarda en caché en `data/simulated/sionna_rt_cache.npz`. Corridas posteriores cargan desde caché instantáneamente.

### 3. Entrenar un modelo fingerprint

```bash
# kNN con k-sweep automático
PYTHONPATH=src python -m ble_indoor train --model knn

# Random Forest
PYTHONPATH=src python -m ble_indoor train --model rf
```

### 4. Correr el estudio baseline completo

```bash
PYTHONPATH=src python -m ble_indoor baseline
```

Imprime zona accuracy, RMSE de posición y métricas de robustez a interferencia.

---

## Referencia CLI

```
PYTHONPATH=src python -m ble_indoor <comando> [opciones]
```

| Comando | Descripción |
|---|---|
| `generate-csv` | Genera CSV de entrenamiento desde simulación |
| `train` | Entrena un modelo fingerprint (kNN o RF) |
| `baseline` | Estudio baseline completo (generar → entrenar → evaluar) |

### `generate-csv`

| Flag | Default | Descripción |
|---|---|---|
| `--simulator` | `pathloss` | Fuente RSSI: `pathloss` o `sionna` |
| `--force` | — | Sobrescribir CSV existente |
| `--config` | `config/baseline_room.yaml` | Ruta al YAML de configuración |

### `train`

| Flag | Default | Descripción |
|---|---|---|
| `--model` | `knn` | Tipo de modelo: `knn` o `rf` |
| `--no-sweep` | — | Omitir gráfico de k-sweep (solo kNN) |
| `--k-sweep-max` | `30` | k máximo a evaluar en el sweep |
| `--config` | `config/baseline_room.yaml` | Ruta al YAML de configuración |

---

## Configuración

Todos los parámetros se definen en `config/baseline_room.yaml`.

```yaml
room:
  width_m: 12.0
  height_m: 8.0

gateways:
  - {id: A1, x_m: 0.0,  y_m: 0.0,  tx_power_dbm: -59.0}
  - {id: A2, x_m: 12.0, y_m: 0.0,  tx_power_dbm: -59.0}
  - {id: A3, x_m: 0.0,  y_m: 8.0,  tx_power_dbm: -59.0}
  - {id: A4, x_m: 12.0, y_m: 8.0,  tx_power_dbm: -59.0}

rssi_model:
  path_loss_exponent: 2.2   # exponente del modelo log-distancia
  noise_sigma_db: 3.0       # desviación estándar del ruido RSSI gaussiano
  min_distance_m: 0.5       # clamp de singularidad

spatial_zones:
  nx: 3                     # columnas de zonas
  ny: 2                     # filas de zonas → 6 zonas en total

trajectory_dataset:
  n_sessions: 14
  steps_per_session: 320
  step_m: 0.35
  gateway_reception_prob: 0.86
  min_visible_gateways: 3
  missing_rssi_dbm: -105.0

sionna_rt:
  carrier_frequency_hz: 2.4e9
  max_depth: 5              # rebotes máximos de rayos
  num_samples: 1000000      # trayectorias Monte Carlo
  grid_resolution_m: 0.25
  wall_material: concrete   # ITU-R P.2040: ε=5.24, σ=0.014
  cache_file: data/simulated/sionna_rt_cache.npz
```

---

## Simuladores RSSI

El protocolo `RssiObservationSource` desacopla la generación de RSSI de la construcción del dataset y el entrenamiento. Cualquier clase que implemente el protocolo de tres métodos puede usarse como backend intercambiable.

### PathLossSimulator

Modelo analítico de path loss log-distancia:

```
RSSI(d) = TxPower_dBm − 10 · n · log₁₀(max(d, d_min)) + ε
donde  ε ~ N(0, σ²),  n = 2.2,  σ = 3 dB,  d_min = 0.5 m
```

Rápido y determinista. Adecuado para prototipos y verificaciones de reproducibilidad. No modela atenuación de paredes ni multipath.

### SionnaRTSimulator

Ray tracing físico usando [Sionna RT](https://nvlabs.github.io/sionna/) (NVIDIA). La habitación se modela como una escena rectangular cerrada Mitsuba 3 (suelo, techo, 4 paredes) con propiedades de material de hormigón ITU-R P.2040.

**Flujo de trabajo:**
1. Construye una grilla 2D de posiciones RX a la resolución configurada.
2. Ejecuta el `PathSolver` de Sionna una vez sobre todos los puntos de la grilla y gateways.
3. Convierte ganancias de path a dBm: `RSSI = 10·log₁₀(Σ|aᵢ|²) + TxPower_dBm`.
4. Guarda la grilla en caché en un archivo `.npz` (invalidado automáticamente al cambiar la config).
5. En tiempo de consulta, usa un KD-tree con ponderación inversa a la distancia (k=4) para interpolar RSSI en posiciones arbitrarias.

| | PathLoss | Sionna RT |
|---|---|---|
| Velocidad | Instantáneo | Lento (en caché tras el primer run) |
| Multipath / reflexiones | No | Sí (hasta 5 rebotes) |
| Atenuación de paredes | No | Sí (propiedades del material) |
| Requiere GPU | No | No (CPU soportado) |
| Requiere TensorFlow | No | Sí |

---

## Modelos

### kNN Fingerprint Estimator

Dos modelos sklearn separados comparten el mismo vector de features RSSI:

- **Regresión de posición** — `KNeighborsRegressor(k=3, weights="distance")` → `(x_m, y_m)`
- **Clasificación de zona** — `KNeighborsClassifier(k=15, weights="distance")` → `zone_id ∈ {0,…,5}`

Se aplica normalización opcional con `StandardScaler` antes del cómputo de distancias cuando `standardize_rssi: true`.

### Random Forest Estimator

- **Regresión de posición** — `RandomForestRegressor(n_estimators=150, max_depth=16)` multi-output → `(x_m, y_m)`
- **Clasificación de zona** — `RandomForestClassifier` con los mismos hiperparámetros → `zone_id`

Ambos estimadores se serializan con `joblib`.

---

## Evaluación

### Métricas

Los errores de posición se calculan como distancia euclídea entre coordenadas estimadas y reales:

```
error_i = ‖(x̂ᵢ, ŷᵢ) − (xᵢ, yᵢ)‖₂
```

Estadísticas reportadas: `mean`, `median`, `p90`, `p95`, `max`, `std`, `n`.

La zona accuracy es la exactitud multiclase estándar sobre el conjunto de validación.

### Robustez a interferencia

La evaluación con `ChannelPerturbation` re-ejecuta la inferencia bajo condiciones de canal degradadas escalando dos parámetros:

| Parámetro | Descripción |
|---|---|
| `noise_sigma_multiplier` | Escala la desviación estándar del ruido RSSI (ej. 1.6 = +60% ruido) |
| `reception_prob_multiplier` | Escala la probabilidad de visibilidad del gateway (ej. 0.85 = 15% pérdida adicional) |

### Artefactos de salida

```
data/results/<experimento>/<modelo>/
├── model.joblib      # estimador serializado
├── metrics.json      # todas las métricas (train, validación, k-sweep si kNN)
└── *.png             # validation_vs_k, mapas de error, CDFs
```

---

## Notebooks

| Notebook | Propósito |
|---|---|
| `notebooks/dataset_building.ipynb` | Generación de dataset, visualización de trayectorias, mapas de calor RSSI |
| `notebooks/eda_fingerprint.ipynb` | Análisis de distribución RSSI, cobertura de gateways, separabilidad de zonas |
| `notebooks/fingerprint_models.ipynb` | Entrenamiento de modelos, gráficos k-sweep, comparación de CDFs de error |

```bash
jupyter lab
```

---

## Experimentos

Las configuraciones de experimentos viven en `experiments/configs/`. Cada archivo YAML es una variante autocontenida que modifica la geometría de la habitación, la ubicación de los gateways o su cantidad. Los resultados se aíslan bajo `data/results/<experimento>/`.

### Configuraciones definidas

| Config | Habitación | Gateways | Layout |
|---|---|---|---|
| `extreme_4gw_12x8` | 12×8 m | 4 | Esquinas de la pared (0,0)…(12,8) |
| `corners_4gw_12x8` | 12×8 m | 4 | Esquinas con margen de 1 m |
| `wall_center_4gw_12x8` | 12×8 m | 4 | Centro de cada pared |
| `random_4gw_12x8` | 12×8 m | 4 | Aleatorio (semilla 7) |
| `corners_3gw_12x8` | 12×8 m | 3 | Triángulo (dos al sur, uno al norte) |
| `corners_6gw_12x8` | 12×8 m | 6 | Dos filas de tres |
| `corners_4gw_20x12` | 20×12 m | 4 | Habitación grande, esquinas con margen de 1 m |

### Correr el sweep

```bash
# Todos los experimentos, ambos modelos
PYTHONPATH=src python experiments/sweep.py --force

# Configs específicas
PYTHONPATH=src python experiments/sweep.py --configs corners_6gw_12x8 corners_4gw_12x8

# Un solo modelo
PYTHONPATH=src python experiments/sweep.py --models rf
```

Los resultados se acumulan en `data/results/sweep_summary.csv`.

### Resultados baseline (simulador path loss, 6 zonas, 25% holdout)

| Experimento | Modelo | Zone acc | RMSE (m) | Error medio (m) | P90 (m) |
|---|---|---|---|---|---|
| `corners_6gw_12x8` | RF | **77.8%** | **1.11** | **1.30** | **2.36** |
| `corners_6gw_12x8` | kNN | 75.9% | 1.21 | 1.43 | 2.46 |
| `corners_4gw_12x8` | RF | 72.9% | 1.31 | 1.57 | 2.91 |
| `corners_4gw_12x8` | kNN | 73.7% | 1.43 | 1.69 | 3.20 |
| `random_4gw_12x8` | RF | 70.8% | 1.49 | 1.66 | 3.34 |
| `random_4gw_12x8` | kNN | 70.8% | 1.55 | 1.72 | 3.44 |
| `wall_center_4gw_12x8` | RF | 67.1% | 1.45 | 1.73 | 3.24 |
| `wall_center_4gw_12x8` | kNN | 68.7% | 1.53 | 1.77 | 3.43 |
| `extreme_4gw_12x8` | RF | 68.1% | 1.44 | 1.72 | 3.18 |
| `extreme_4gw_12x8` | kNN | 67.9% | 1.57 | 1.85 | 3.55 |
| `corners_3gw_12x8` | RF | 65.8% | 1.70 | 1.91 | 3.68 |
| `corners_3gw_12x8` | kNN | 66.5% | 1.84 | 2.06 | 4.04 |
| `corners_4gw_20x12` | RF | 52.6% | 1.97 | 2.34 | 4.36 |
| `corners_4gw_20x12` | kNN | 54.0% | 2.13 | 2.54 | 4.71 |

**Conclusiones:**
- 6 gateways suma ~9 pp de zona accuracy sobre la mejor configuración con 4 gateways.
- Las esquinas con margen superan consistentemente a las posiciones en el centro de las paredes y en las esquinas exactas con igual cantidad de gateways.
- La habitación grande (20×12 m) con solo 4 gateways cae a 52–54% de zona accuracy — la densidad de cobertura importa más que el tamaño del espacio.
- RF supera a kNN en RMSE de posición en todas las configuraciones; las diferencias en zona accuracy están dentro del margen de ruido.
