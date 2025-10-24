import pandas as pd

df = pd.read_parquet('/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00c_final/2024-25/teamgamelogs_by_game.parquet')
df2 = pd.read_parquet('/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00c_final/2024-25/dashboards/team_dashboard_by_general_splits__dataset_5.parquet')
print("Columnas del archivo:")
for columna in df2.columns:
    print(f"- {columna}")