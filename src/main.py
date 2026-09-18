from helpers.generate_dataframe import dataframe
from helpers.clean_dataframe import clean_dataframe

df = clean_dataframe(dataframe())
print(df['LOCALIDAD'].unique())