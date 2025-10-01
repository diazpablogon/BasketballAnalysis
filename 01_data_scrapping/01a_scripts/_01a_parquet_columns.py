import pandas as pd
import os

files = [
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dash_lineups/2024-25/Regular Season/team_dash_lineups__1610612737__dataset_0.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dash_lineups/2024-25/Regular Season/team_dash_lineups__1610612737__dataset_1.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dash_pt_pass/2024-25/Regular Season/team_dash_pt_pass__1610612737__dataset_0.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dash_pt_pass/2024-25/Regular Season/team_dash_pt_pass__1610612737__dataset_1.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dash_pt_reb/2024-25/Regular Season/team_dash_pt_reb__1610612737__dataset_0.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dash_pt_reb/2024-25/Regular Season/team_dash_pt_reb__1610612737__dataset_1.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dash_pt_reb/2024-25/Regular Season/team_dash_pt_reb__1610612737__dataset_2.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dash_pt_reb/2024-25/Regular Season/team_dash_pt_reb__1610612737__dataset_3.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dash_pt_reb/2024-25/Regular Season/team_dash_pt_reb__1610612737__dataset_4.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dash_pt_shots/2024-25/Regular Season/team_dash_pt_shots__1610612737__dataset_0.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dash_pt_shots/2024-25/Regular Season/team_dash_pt_shots__1610612737__dataset_1.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dash_pt_shots/2024-25/Regular Season/team_dash_pt_shots__1610612737__dataset_2.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dash_pt_shots/2024-25/Regular Season/team_dash_pt_shots__1610612737__dataset_3.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dash_pt_shots/2024-25/Regular Season/team_dash_pt_shots__1610612737__dataset_4.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dash_pt_shots/2024-25/Regular Season/team_dash_pt_shots__1610612737__dataset_5.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dashboard_by_general_splits/2024-25/Regular Season/team_dashboard_by_general_splits__1610612737__dataset_0.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dashboard_by_general_splits/2024-25/Regular Season/team_dashboard_by_general_splits__1610612737__dataset_1.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dashboard_by_general_splits/2024-25/Regular Season/team_dashboard_by_general_splits__1610612737__dataset_2.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dashboard_by_general_splits/2024-25/Regular Season/team_dashboard_by_general_splits__1610612737__dataset_3.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dashboard_by_general_splits/2024-25/Regular Season/team_dashboard_by_general_splits__1610612737__dataset_4.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dashboard_by_general_splits/2024-25/Regular Season/team_dashboard_by_general_splits__1610612737__dataset_5.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dashboard_by_shooting_splits/2024-25/Regular Season/team_dashboard_by_shooting_splits__1610612737__dataset_0.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dashboard_by_shooting_splits/2024-25/Regular Season/team_dashboard_by_shooting_splits__1610612737__dataset_1.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dashboard_by_shooting_splits/2024-25/Regular Season/team_dashboard_by_shooting_splits__1610612737__dataset_2.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dashboard_by_shooting_splits/2024-25/Regular Season/team_dashboard_by_shooting_splits__1610612737__dataset_3.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dashboard_by_shooting_splits/2024-25/Regular Season/team_dashboard_by_shooting_splits__1610612737__dataset_4.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dashboard_by_shooting_splits/2024-25/Regular Season/team_dashboard_by_shooting_splits__1610612737__dataset_5.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_dashboard_by_shooting_splits/2024-25/Regular Season/team_dashboard_by_shooting_splits__1610612737__dataset_6.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_player_dashboard/2024-25/Regular Season/team_player_dashboard__1610612737__dataset_0.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_player_dashboard/2024-25/Regular Season/team_player_dashboard__1610612737__dataset_1.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_player_on_off_details/2024-25/Regular Season/team_player_on_off_details__1610612737__dataset_0.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_player_on_off_details/2024-25/Regular Season/team_player_on_off_details__1610612737__dataset_1.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_player_on_off_summary/2024-25/Regular Season/team_player_on_off_summary__1610612737__dataset_0.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_player_on_off_summary/2024-25/Regular Season/team_player_on_off_summary__1610612737__dataset_1.parquet",
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/team_dashboard/team_player_on_off_summary/2024-25/Regular Season/team_player_on_off_summary__1610612737__dataset_2.parquet",

]

col_sets = {}
errores = []

for f in files:
    try:
        df = pd.read_parquet(f)  # si te falta pyarrow: pip install pyarrow
        col_sets[f] = set(map(str, df.columns))
    except Exception as e:
        errores.append((f, str(e)))

if not col_sets:
    print("No se pudo leer ninguna tabla.")
    if errores:
        for f, msg in errores:
            print(f"⚠️ {os.path.basename(f)} -> {msg}")
    raise SystemExit()

# 1) Columnas comunes a TODOS
common_cols = set.intersection(*col_sets.values())
print(f"\n=== COLUMNAS COMUNES ({len(common_cols)}) ===")
for c in sorted(common_cols):
    print(f"  - {c}")

# 2) Columnas NO comunes (extras) por archivo
for f in files:
    if f not in col_sets:
        continue
    extras = sorted(col_sets[f] - common_cols)
    base = os.path.basename(f)
    print(f"\n=== EXTRAS en {base} ({len(extras)}) ===")
    if extras:
        for c in extras:
            print(f"  - {c}")
    else:
        print("  (ninguna)")

# 3) (Opcional) Línea “copiar-pegar” por archivo con solo EXTRAS
#    Descomenta si lo quieres:
# for f in files:
#     if f in col_sets:
#         extras = sorted(col_sets[f] - common_cols)
#         print(f"{os.path.basename(f)} | {', '.join(extras)}")

# Mostrar errores de lectura (si hubo)
if errores:
    print("\n=== ERRORES DE LECTURA ===")
    for f, msg in errores:

        print(f"⚠️ {os.path.basename(f)} -> {msg}")