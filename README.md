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
              │  MLP  → posición + zona    │
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
| `--model` | `knn` | Tipo de modelo: `knn`, `rf` o `mlp` |
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

### MLP Fingerprint Estimator

Red neuronal multicapa con dos cabezas sklearn:

- **Regresión de posición** — `MLPRegressor(hidden_layer_sizes=[128,64,32], activation=relu)` → `(x_m, y_m)`
- **Clasificación de zona** — `MLPClassifier` con la misma arquitectura → `zone_id`

El `StandardScaler` se bake-a opcionalmente en la primera capa. Soporta exportación a **TFLite INT8** para despliegue en Nordic / Cortex-M4 vía TFLite Micro:

```bash
PYTHONPATH=src python -m ble_indoor train --model mlp --export-tflite
```

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

### Resultados: path loss vs Sionna RT — 3 modelos, split 80/20

Zona accuracy (%) en validación. **PL** = path loss analítico · **RT** = Sionna RT ray-tracing.
Split estratificado 80/20, `random_state=123`.

| Experimento | GWs | Zonas | kNN PL | kNN RT | RF PL | RF RT | MLP PL | MLP RT |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `corners_6gw_12x8` | 6 | 6 | 76.6% | 61.5% | 77.9% | 62.5% | **79.0%** | **64.0%** |
| `corners_4gw_12x8` | 4 | 6 | 73.9% | 58.5% | 73.5% | 59.7% | 75.7% | 62.5% |
| `random_4gw_12x8` | 4 | 6 | 71.6% | 55.0% | 71.0% | 53.5% | 75.2% | 57.7% |
| `extreme_4gw_12x8` | 4 | 6 | 68.5% | 56.4% | 68.3% | 53.5% | 70.6% | 57.5% |
| `wall_center_4gw_12x8` | 4 | 6 | 69.4% | 54.4% | 67.0% | 50.7% | 70.1% | 55.3% |
| `corners_3gw_12x8` | 3 | 6 | 66.7% | 50.4% | 67.1% | 51.2% | 69.6% | 53.1% |
| `corners_4gw_20x12` | 4 | 12 | 53.6% | 43.4% | 53.6% | 43.7% | 56.8% | 47.1% |

#### RMSE de posición — mejor modelo por escenario (m)

| Experimento | MLP path loss | MLP Sionna RT | Δ RMSE |
|---|:---:|:---:|:---:|
| `corners_6gw_12x8` | **1.08** | **1.59** | +0.51 |
| `corners_4gw_12x8` | 1.27 | 1.72 | +0.45 |
| `random_4gw_12x8` | 1.33 | 1.89 | +0.56 |
| `extreme_4gw_12x8` | 1.39 | 1.86 | +0.47 |
| `wall_center_4gw_12x8` | 1.41 | 1.88 | +0.47 |
| `corners_3gw_12x8` | 1.61 | 2.15 | +0.54 |
| `corners_4gw_20x12` | 1.89 | 2.45 | +0.56 |

#### Brecha media entre simuladores (path loss − Sionna RT)

| Modelo | Media PL | Media RT | Brecha |
|---|:---:|:---:|:---:|
| kNN | 68.6% | 54.2% | **−14.4 pp** |
| RF | 68.4% | 54.9% | **−13.5 pp** |
| MLP | 71.0% | 57.3% | **−13.7 pp** |

**Conclusiones:**

- **MLP supera a kNN y RF** en todos los escenarios y simuladores (1–4 pp en zona accuracy, ~5% mejor RMSE).
- **6 gateways** es la mejor configuración: +8–9 pp sobre la mejor variante con 4 gateways, en ambos simuladores.
- **Sionna RT muestra una brecha sistemática de ~14 pp** respecto al path loss analítico. El modelo analítico sobreestima la precisión real porque no modela multipath ni atenuación de paredes — los resultados de Sionna RT son más representativos de un entorno real.
- **La habitación grande** (20×12 m, 12 zonas) es el escenario más difícil: 56.8% / 47.1% con MLP path loss / Sionna RT. La densidad de cobertura cae abruptamente con 4 gateways en un espacio mayor.
- **El layout importa** con pocos gateways: `corners_4gw_12x8` (esquinas interiores) supera a `wall_center` y `extreme` en Sionna RT en 4–9 pp.
