import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    average_precision_score, brier_score_loss, log_loss,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.calibration import calibration_curve
import ipywidgets as widgets
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
# === CONFIGURACIÓN PRINCIPAL ===
DATA_PATH = "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00c_final/2024-25/teamgamelogs_by_game.parquet"
OUTPUT_DIR = "/Users/pablo/Documents/BigData/BasketballAnalysis/02_processing_data/02a_WL_prediction/preds"
CUTOFF_DATE = "2025-02-15"
SELECTED_MODEL = "GB"  # "LR" o "GB"

# Features seleccionadas (sin recalcular correlación)
FEATURE_LIST = [
    'ROLL10_NET_RATING','ROLL10_E_NET_RATING','ROLL10_DEF_RATING','ROLL10_E_DEF_RATING',
    'ROLL10_PTS','ROLL10_FGM','ROLL10_FG_PCT','ROLL10_EFG_PCT','ROLL10_TS_PCT',
    'ROLL10_TOV','ROLL10_TM_TOV_PCT','ROLL10_OREB_PCT','ROLL10_DREB','ROLL10_POSS',
    'ROLL10_PACE','ROLL10_PCT_FGA_2PT','ROLL10_PCT_FGA_3PT','ROLL10_PCT_PTS_FT'
]

# === OPCIONAL: Parquets externos ===
# ExternalFeaturesConfig = [
#     {
#         "path": "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00c_final/2024-25/dashboards/team_dash_pt_pass__dataset_0.parquet",
#         "on": ["TEAM_ID","GAME_DATE"],
#         "select": ["AST_PCT","PCT_AST_2PM","PCT_AST_3PM"],
#         "prefix": "PASS_"
#     }
# ]
ExternalFeaturesConfig = []
df = pd.read_parquet(DATA_PATH)
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

if 'WL_NUM' not in df.columns and 'WL' in df.columns:
    df['WL_NUM'] = df['WL'].map({'W': 1, 'L': 0})

selected_columns = ['TEAM_ID', 'GAME_DATE', 'WL_NUM'] + [c for c in FEATURE_LIST if c in df.columns]
df = df[selected_columns].copy()

print(f"Filas totales: {len(df)}")
print(f"Fechas: {df['GAME_DATE'].min().date()} → {df['GAME_DATE'].max().date()}")
print(f"Equipos únicos: {df['TEAM_ID'].nunique()}")
df.head(3)
# def add_external_features(df, cfg):
#     ext = pd.read_parquet(cfg["path"])
#     ext = ext[cfg["on"] + cfg["select"]]
#     new_cols = [cfg["prefix"] + c for c in cfg["select"]]
#     ext.columns = cfg["on"] + new_cols
#     return df.merge(ext, on=cfg["on"], how="left")

# for cfg in ExternalFeaturesConfig:
#     df = add_external_features(df, cfg)
# print("Datos extendidos con features externas (si las hubiera).")
for c in FEATURE_LIST:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())
cutoff_dt = pd.Timestamp(CUTOFF_DATE)
train = df[df['GAME_DATE'] < cutoff_dt].copy()
test = df[df['GAME_DATE'] >= cutoff_dt].copy()

X_train = train[FEATURE_LIST]
y_train = train['WL_NUM']
X_test = test[FEATURE_LIST]
y_test = test['WL_NUM']

print(f"Train: {len(train)} partidos  |  Test: {len(test)} partidos")
print("Distribución Train:", y_train.value_counts(normalize=True).round(3).to_dict())
print("Distribución Test:", y_test.value_counts(normalize=True).round(3).to_dict())
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

model_gb = GradientBoostingClassifier(random_state=42)

pipe_lr.fit(X_train, y_train)
model_gb.fit(X_train, y_train)
def evaluate_model(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "avg_precision": average_precision_score(y, proba),
        "brier": brier_score_loss(y, proba),
        "log_loss": log_loss(y, proba)
    }

results = {
    "Logistic Regression": evaluate_model(pipe_lr, X_test, y_test),
    "Gradient Boosting": evaluate_model(model_gb, X_test, y_test)
}

pd.DataFrame(results).T
best_model = model_gb if SELECTED_MODEL == "GB" else pipe_lr
model_name = "Gradient Boosting" if SELECTED_MODEL == "GB" else "Logistic Regression"

proba_best = best_model.predict_proba(X_test)[:, 1]
preds_best = (proba_best >= 0.5).astype(int)
cm = confusion_matrix(y_test, preds_best)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=axes[0])
axes[0].set_title(f"Curva ROC - {model_name}")

prob_true, prob_pred = calibration_curve(y_test, proba_best, n_bins=10, strategy='uniform')
axes[1].plot(prob_pred, prob_true, marker='o', label='Calibración')
axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal')
axes[1].set_xlabel('Probabilidad predicha')
axes[1].set_ylabel('Proporción real de victorias')
axes[1].set_title('Curva de calibración')
axes[1].legend()

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[2])
axes[2].set_title('Matriz de confusión (umbral 0.5)')
axes[2].set_xlabel('Predicción')
axes[2].set_ylabel('Valor real')
axes[2].set_xticklabels(['Derrota', 'Victoria'])
axes[2].set_yticklabels(['Derrota', 'Victoria'], rotation=0)

plt.tight_layout()
plt.show()
from pathlib import Path

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

best_model = model_gb if SELECTED_MODEL == "GB" else pipe_lr

test_preds = pd.DataFrame({
    "TEAM_ID": test["TEAM_ID"],
    "GAME_DATE": test["GAME_DATE"],
    "y_true": y_test,
    "proba_W": best_model.predict_proba(X_test)[:, 1]
})

test_preds["pred_W"] = (test_preds["proba_W"] >= 0.5).astype(int)

output_path = Path(OUTPUT_DIR) / "test_preds.csv"
test_preds.to_csv(output_path, index=False)
print(f"Predicciones guardadas en {output_path}")
