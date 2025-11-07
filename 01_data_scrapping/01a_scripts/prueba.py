import pandas as pd

df = pd.read_parquet('/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00c_final/2024-25/teamgamelogs_by_game.parquet')
df2 = pd.read_parquet('/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00c_final/2024-25/dashboards/team_dashboard_by_general_splits__dataset_5.parquet')
df3 = pd.read_parquet('/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00d_featurized/00a_WL_prediction/teamgamelogs.parquet')
df4 = pd.read_parquet('/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00c_final/2024-25/teamgamelogs_by_game.parquet')
print("Columnas del archivo:")
for columna in df4.columns:
    print(f"- {columna}")