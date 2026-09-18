# IMPORTS 
import pandas as pd
import matplotlib.pyplot as plt

from helpers.generate_dataframe import dataframe
from helpers.clean_dataframe import clean_dataframe

df = clean_dataframe(dataframe())

# Agrupamos cantidad todos los accidentes por fecha y localidad
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

all_combinations = pd.MultiIndex.from_product(
    [all_dates, unique_locations],
    names=['fecha', 'LOCALIDAD']
)

group_accidents_by_date_and_location['fecha'] = pd.to_datetime(group_accidents_by_date_and_location['fecha'])

# Fuerza al DataFrame a tener exactamente las filas de all_combinations.
# Para las combinaciones que ya existían conserva su conteo real; para las que no existían (no hubo accidentes ese día en esa localidad), crea la fila con num_accidentes = 0.
all_registers = (
    group_accidents_by_date_and_location.set_index(['fecha', 'LOCALIDAD'])
                     .reindex(all_combinations, fill_value=0)
                     .reset_index()
)

all_registers['dia_semana'] = all_registers['fecha'].dt.day_name()
all_registers['mes'] = all_registers['fecha'].dt.month
all_registers['anio'] = all_registers['fecha'].dt.year

all_registers[['num_accidentes']].hist(bins=27, figsize=(8,5))
plt.show()

print(all_registers['num_accidentes'].skew())