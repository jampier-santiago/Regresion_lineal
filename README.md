# Regresión Lineal — Accidentes de tránsito en Bogotá

Proyecto de ciencia de datos que predice el número diario de accidentes de tránsito por localidad en Bogotá, a partir del histórico de siniestros viales (2015-2021), usando un modelo de regresión lineal.

## Objetivo

A partir de los registros crudos de siniestros, construir una serie temporal diaria por localidad, explorar los factores que influyen en la cantidad de accidentes (día de la semana, mes, localidad) y entrenar un modelo de regresión lineal que estime `num_accidentes` para una fecha y localidad dadas.

## Datos

- **Fuente:** `data/raw/historico_siniestros_bogota.csv` — histórico de siniestros viales de Bogotá (2015-2021), separado por `;` y codificado en `latin-1`.
- **Variable objetivo:** `num_accidentes`, el conteo de accidentes por combinación fecha-localidad.
- **Features usadas:** `mes`, `anio`, `dia_semana` y `LOCALIDAD` (estas dos últimas codificadas con one-hot encoding).
- **Datos procesados** (generados por los notebooks, no versionados salvo `.gitkeep`):
  - `data/processed/accidentes_limpios.csv`: dataset crudo ya limpio.
  - `data/processed/accidentes_diarios_localidad.csv`: serie diaria agregada por localidad, con features de fecha.

## Estructura del proyecto

```text
├── data/
│   ├── raw/                  # Datos crudos (CSV original)
│   └── processed/            # Datos limpios/agregados (generados por los notebooks)
├── reports/
│   └── figures/               # Gráficas generadas por src/main.py
├── notebooks/
│   ├── 01_exploracion_y_limpieza.ipynb   # Carga, exploración y limpieza de datos
│   ├── 02_agregacion_features.ipynb      # Agregación diaria y features de fecha
│   └── 03_modelo_regresion.ipynb         # Split, entrenamiento y evaluación del modelo
├── src/
│   ├── helpers/
│   │   ├── generate_dataframe.py         # Carga el CSV crudo
│   │   └── clean_dataframe.py            # Limpieza (fechas, filtro de localidad)
│   ├── validate_info.py                  # Script de validación exploratoria
│   └── main.py                           # Pipeline completo: limpieza → features → entrenamiento → gráficas
├── models/
│   └── linear_regresion.py               # Entrenamiento y métricas del modelo (LinearRegression)
└── requirements.txt
```

## Pipeline

1. **Limpieza** (`clean_dataframe`): convierte `FECHA_HORA_ACC` a fecha y descarta registros sin `LOCALIDAD` o con `LOCALIDAD == 'SUMAPAZ'` (caso atípico con muy pocos registros).
2. **Agregación**: se cuenta el número de accidentes por combinación fecha-localidad y se completan con cero los días sin accidentes, generando una serie temporal completa.
3. **Features de fecha**: se derivan `dia_semana`, `mes` y `anio` a partir de la fecha.
4. **Filtro pre-pandemia**: se excluyen los datos posteriores al 1 de marzo de 2020 para no contaminar el modelo con la caída anómala de accidentes durante la cuarentena estricta por COVID-19.
5. **Split cronológico**: el conjunto de entrenamiento/prueba se divide por fecha (percentil 80), no de forma aleatoria, para evitar fuga de información propia de series de tiempo.
6. **Codificación**: `dia_semana` y `LOCALIDAD` se codifican con one-hot encoding (`drop_first=True`).
7. **Entrenamiento y evaluación** (`generate_model`): se entrena una `LinearRegression` de scikit-learn y se reportan R², MAE y RMSE sobre el set de prueba.
8. **Diagnóstico**: se analizan los residuos (vs. predicciones y su distribución) para validar los supuestos del modelo.

## Cómo ejecutarlo

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Luego, ejecuta los notebooks en orden (`01` → `02` → `03`) para reproducir el análisis paso a paso, o corre el pipeline completo directamente:

```bash
python src/main.py
```

Esto genera las siguientes gráficas en `reports/figures/`:

- `accidentes_por_dia.png` — promedio de accidentes por día de la semana
- `accidentes_por_mes.png` — promedio de accidentes por mes
- `accidentes_por_localidad.png` — promedio de accidentes por localidad
- `serie_tiempo_completa.png` — serie diaria total de accidentes (2015-2021)
- `residuos_vs_predicciones.png` — diagnóstico de residuos vs. predicciones
- `histograma_residuos.png` — distribución de los residuos

## Requisitos

Ver [requirements.txt](requirements.txt): `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `plotly`, `statsmodels`, `jupyter`.
