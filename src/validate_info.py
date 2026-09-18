# IMPORTS
from helpers.generate_dataframe import dataframe
from helpers.clean_dataframe import clean_dataframe

def main():
    df = dataframe()

    # Validar cuantas localidades tenemos
    print(df['LOCALIDAD'].unique())

    # Contamos cuantos registros hay por localidad incluyendo los valores nulos
    print(df['LOCALIDAD'].value_counts(dropna=False))

    # Aplicamos la limpieza (fechas + filtro de localidad nula/Sumapaz)
    df_clean = clean_dataframe(df)

    print(df_clean.shape)


if __name__ == "__main__":
    main()