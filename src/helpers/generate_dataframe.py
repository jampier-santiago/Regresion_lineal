# IMPORTS 
from pathlib import Path
import pandas as pd

RUTA_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "historico_siniestros_bogota.csv"

def dataframe():
    return pd.read_csv(RUTA_CSV, encoding="latin-1", sep=";")