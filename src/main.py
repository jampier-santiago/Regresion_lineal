"""
Pipeline de regresión lineal para predecir el número diario de accidentes
de tránsito en Bogotá, por localidad.

Flujo: carga y limpieza de datos -> agregación diaria por localidad ->
relleno de combinaciones fecha/localidad sin accidentes -> exploración
gráfica -> split train/test temporal (pre-pandemia) -> entrenamiento y
evaluación del modelo -> análisis de residuos.

Genera las figuras en reports/figures y las guarda como efecto secundario
al ejecutar el script.
"""

# IMPORTS
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))

from helpers.generate_dataframe import dataframe
from helpers.clean_dataframe import clean_dataframe
from models.linear_regresion import generate_model

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Dataset ya limpio (tipos correctos, nulos/duplicados tratados en clean_dataframe)
df = clean_dataframe(dataframe())

# Colapsamos los registros individuales de accidentes a un conteo diario por localidad,
# que es la granularidad que usaremos como target del modelo.
group_accidents_by_date_and_location = (
    df.groupby([df['FECHA_HORA_ACC'].dt.date, 'LOCALIDAD'])
            .size()
            .reset_index(name='num_accidentes')
            .rename(columns={'FECHA_HORA_ACC': 'fecha'})
)

# rango completo de fechas del dataset
all_dates = pd.date_range(
    start=group_accidents_by_date_and_location['fecha'].min(),
    end=group_accidents_by_date_and_location['fecha'].max(),
    freq='D'
)

unique_locations = df['LOCALIDAD'].unique()

# Producto cartesiano fecha x localidad: es el universo completo de combinaciones
# que debería existir si cada localidad tuviera un registro (aunque sea en cero) cada día.
all_combinations = pd.MultiIndex.from_product(
    [all_dates, unique_locations],
    names=['fecha', 'LOCALIDAD']
)

group_accidents_by_date_and_location['fecha'] = pd.to_datetime(group_accidents_by_date_and_location['fecha'])

# El groupby anterior solo genera filas para combinaciones fecha/localidad donde SÍ hubo
# accidentes. Sin este reindex, los días sin accidentes en una localidad simplemente no
# aparecerían en el dataset, sesgando el promedio hacia arriba. Reindexar contra
# all_combinations fuerza la fila para cada combinación posible, rellenando con 0 los casos
# ausentes (conserva el conteo real donde ya existía).
all_registers = (
    group_accidents_by_date_and_location.set_index(['fecha', 'LOCALIDAD'])
                     .reindex(all_combinations, fill_value=0)
                     .reset_index()
)

# Features temporales derivadas de la fecha, usadas más adelante como variables del modelo.
all_registers['dia_semana'] = all_registers['fecha'].dt.day_name()
all_registers['mes'] = all_registers['fecha'].dt.month
all_registers['anio'] = all_registers['fecha'].dt.year

# --- Exploración gráfica ---
# Todas las figuras se guardan en FIGURES_DIR para el reporte; no se muestran interactivamente.

# Distribución general del número de accidentes por combinación día-localidad
all_registers[['num_accidentes']].hist(bins=27, figsize=(8,5))

# accidentes promedio por día de la semana
plt.figure(figsize=(8,5))
all_registers.groupby('dia_semana')['num_accidentes'].mean().sort_values().plot(kind='barh')
plt.title('Promedio de accidentes por día de la semana')
plt.savefig(FIGURES_DIR / 'accidentes_por_dia.png')
plt.close()

# accidentes promedio por mes
all_registers.groupby('mes')['num_accidentes'].mean().plot(kind='bar', figsize=(8,5))
plt.title('Promedio de accidentes por mes')
plt.savefig(FIGURES_DIR / 'accidentes_por_mes.png')
plt.close()

# accidentes promedio por localidad (alta variación esperada por diferencias de tamaño/tráfico entre localidades)
all_registers.groupby('LOCALIDAD')['num_accidentes'].mean().sort_values().plot(kind='barh', figsize=(10,6))
plt.title('Promedio de accidentes por localidad')
plt.savefig(FIGURES_DIR / 'accidentes_por_localidad.png')
plt.close()

serie_diaria = all_registers.groupby('fecha')['num_accidentes'].sum()

# Serie de tiempo del total de accidentes diarios (todas las localidades sumadas).
# La línea vertical marca el inicio de la cuarentena estricta por COVID-19, que introduce
# una caída abrupta y sostenida en el patrón normal de accidentalidad.
plt.figure(figsize=(14,5))
serie_diaria.plot()
plt.axvline(pd.Timestamp('2020-03-20'), color='red', linestyle='--', label='Inicio cuarentena estricta')
plt.title('Total de accidentes diarios en Bogotá (2015-2021)')
plt.legend()
plt.savefig(FIGURES_DIR / 'serie_tiempo_completa.png')
plt.close()

# --- Preparación para el modelo ---
# Se descarta el periodo de pandemia: es un evento anómalo (caída abrupta por cuarentena)
# que no sigue el patrón estacional que el modelo busca capturar, y lo contaminaría.
all_registers_pre_pandemia = all_registers[all_registers['fecha'] < '2020-03-01'].copy()

# Split train/test temporal (no aleatorio): se ordenan las fechas únicas y se corta en el
# percentil 80, dejando el tramo más reciente como test. Esto evita fuga de información
# (usar el "futuro" para predecir el "pasado") y simula el escenario real de predecir
# accidentes de días que aún no han ocurrido.
unique_dates = all_registers_pre_pandemia['fecha'].drop_duplicates().sort_values()
date_for_cut = unique_dates.iloc[int(len(unique_dates) * 0.8)]

train_mask = all_registers_pre_pandemia['fecha'] < date_for_cut
test_mask = all_registers_pre_pandemia['fecha'] >= date_for_cut

df_train = all_registers_pre_pandemia[train_mask]
df_test = all_registers_pre_pandemia[test_mask]

# One-hot encoding de las variables categóricas (día de la semana y localidad).
# Nota: al codificar train y test por separado se corre el riesgo de que una categoría
# presente en un conjunto no aparezca en el otro, generando columnas distintas entre
# X_train y X_test. Aquí no se observa el problema porque ambos conjuntos cubren el
# rango completo de localidades y días de la semana, pero en datasets más pequeños o con
# categorías raras conviene codificar sobre el DataFrame completo antes de dividir.
X_train = pd.get_dummies(df_train[['mes', 'anio', 'dia_semana', 'LOCALIDAD']], columns=['dia_semana', 'LOCALIDAD'], drop_first=True)
X_test = pd.get_dummies(df_test[['mes', 'anio', 'dia_semana', 'LOCALIDAD']], columns=['dia_semana', 'LOCALIDAD'], drop_first=True)

Y_train = df_train['num_accidentes']
Y_test = df_test['num_accidentes']

# Entrena la regresión lineal y ya imprime R², MAE y RMSE sobre el set de test
# (ver models/linear_regresion.py).
model = generate_model(X_train, Y_train, X_test, Y_test)

# Coeficientes ordenados de mayor a menor: indican qué categorías (localidad/día) o
# variables numéricas (mes/año) empujan más al alza la predicción de accidentes.
coeficientes = pd.Series(model.coef_, index=X_train.columns).sort_values(ascending=False)

Y_pred = model.predict(X_test)

# Residuos = valor real - valor predicho. Se analizan para validar los supuestos de la
# regresión lineal (varianza constante, ausencia de patrones sistemáticos).
residuos = Y_test - Y_pred

# 1. Residuos vs valores predichos: una nube sin patrón alrededor de 0 sugiere que el
# modelo no está sesgado sistemáticamente; un patrón (curva, embudo) indicaría
# heterocedasticidad o relaciones no lineales no capturadas.
plt.figure(figsize=(8,5))
plt.scatter(Y_pred, residuos, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Valores predichos')
plt.ylabel('Residuos')
plt.title('Residuos vs Predicciones')
plt.savefig(FIGURES_DIR / 'residuos_vs_predicciones.png')
plt.close()

# 2. Histograma de residuos: se espera una distribución aproximadamente simétrica y
# centrada en 0 si el modelo no tiene sesgo sistemático.
plt.figure(figsize=(8,5))
residuos.hist(bins=30)
plt.title('Distribución de los residuos')
plt.savefig(FIGURES_DIR / 'histograma_residuos.png')
plt.close()

# Resumen estadístico de los residuos (media, desviación, cuartiles) para chequeo rápido en consola.
print(residuos.describe())