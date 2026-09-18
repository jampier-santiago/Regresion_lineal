# IMPORTS
import pandas as pd


def clean_dataframe(df):
    df = df.copy()

    # Ajustar las fechas para que tengan un formato valido
    df['FECHA_HORA_ACC'] = pd.to_datetime(df['FECHA_HORA_ACC'], errors='coerce')

    # Filtramos nulos de localidad y Sumapaz
    df = df[df['LOCALIDAD'].notna() & (df['LOCALIDAD'] != 'SUMAPAZ')].copy()

    return df
