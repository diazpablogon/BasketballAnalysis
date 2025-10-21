import pandas as pd

df = pd.read_parquet('/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00c_final/2024-25/teamgamelogs_by_game.parquet')
print("Columnas del archivo:")
for columna in df.columns:
    print(f"- {columna}")