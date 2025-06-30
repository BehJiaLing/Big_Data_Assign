import pandas as pd

# import dataset
df = pd.read_csv('ObesityDataSet_Ori.csv')

# View first few rows
print(df.head())

# Check for data types and missing values
print(df.info())

# Summary statistics
print(df.describe(include='all'))

