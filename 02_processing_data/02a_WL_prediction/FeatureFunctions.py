"""Utility functions to compute basketball features and train win/loss estimators."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

DEFAULT_DAYS_REST = 5


__all__ = [
    'DEFAULT_DAYS_REST',
    'features_roll10',
    'parse_days_rest_value',
    'load_days_rest_reference',
    'calculate_current_streak',
    'features_enhanced',
    'features_venue',
    'build_match_dataset_enhanced',
    'fit_and_eval',
]


def features_roll10(df: pd.DataFrame) -> pd.DataFrame:
    """Selects and cleans pre-computed rolling features with a 10-game window."""
    df = df.copy()

    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    id_columns = [
        'TEAM_ID',
        'TEAM_ABBREVIATION',
        'GAME_ID',
        'GAME_DATE',
        'MATCHUP',
        'WL',
        'WL_NUM',
    ]

    roll_features = sorted([c for c in df.columns if c.startswith('ROLL10_')])
    if not roll_features:
        print("⚠️ No se encontraron columnas ROLL10_ en el parquet.")
    else:
        print(f"Columnas ROLL10 detectadas: {len(roll_features)}")

    selected = [c for c in id_columns if c in df.columns] + roll_features
    df = df[selected].copy()

    if 'GAME_DATE' in df.columns:
        print(f"Filas totales tras selección: {len(df)}")
        print(
            "Rango de fechas tras selección: "
            f"{df['GAME_DATE'].min().date()} → {df['GAME_DATE'].max().date()}"
        )
    if 'TEAM_ID' in df.columns:
        print(f"Equipos únicos tras selección: {df['TEAM_ID'].nunique()}")

    if {'TEAM_ID', 'GAME_DATE'}.issubset(df.columns):
        df = df.sort_values(['TEAM_ID', 'GAME_DATE'])

    roll_cols = sorted([c for c in df.columns if c.startswith('ROLL10_')])
    if not roll_cols:
        print("⚠️ No hay columnas ROLL10_ para imputar valores faltantes.")
    else:
        for col in roll_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        print(f"Columnas ROLL10 imputadas (con la mediana): {len(roll_cols)}")

    return df


def parse_days_rest_value(value):
    """Converts the rest range text to a numeric approximation."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"\d+", text)
    if match:
        return float(match.group())
    lowered = text.lower()
    if 'back' in lowered:
        return 0.0
    return None


def load_days_rest_reference(path: str | Path) -> pd.DataFrame:
    """Loads a parquet with rest information and normalises the key columns."""
    path = Path(path)
    if not path.exists():
        print(
            f"⚠️ No se encontró el parquet de days rest en {path}. "
            "Se usará la diferencia de fechas como respaldo."
        )
        return pd.DataFrame()

    rest_df = pd.read_parquet(path)
    if rest_df.empty:
        return pd.DataFrame()

    team_col = None
    for cand in ['TEAM_ID', 'TEAM_ID_x', 'TEAM_ID_y']:
        if cand in rest_df.columns:
            team_col = cand
            break
    if team_col is None:
        print("⚠️ El parquet de days rest no contiene TEAM_ID. Se omite el merge.")
        return pd.DataFrame()

    date_col = None
    for cand in ['GAME_DATE', 'TEAM_GAME_DATE', 'DATE']:
        if cand in rest_df.columns:
            date_col = cand
            break
    if date_col is None:
        print("⚠️ El parquet de days rest no contiene GAME_DATE. Se omite el merge.")
        return pd.DataFrame()

    range_col = None
    for cand in ['TEAM_DAYS_REST_RANGE', 'DAYS_REST_RANGE', 'TEAM_DAYS_REST']:
        if cand in rest_df.columns:
            range_col = cand
            break
    if range_col is None:
        print(
            "⚠️ El parquet de days rest no contiene TEAM_DAYS_REST_RANGE. "
            "Se omite el merge."
        )
        return pd.DataFrame()

    out = rest_df[[team_col, date_col, range_col]].copy()
    out = out.rename(
        columns={team_col: 'TEAM_ID', date_col: 'GAME_DATE', range_col: 'TEAM_DAYS_REST_RANGE'}
    )
    out['GAME_DATE'] = pd.to_datetime(out['GAME_DATE'])
    out['DAYS_REST_SOURCE'] = out['TEAM_DAYS_REST_RANGE'].apply(parse_days_rest_value)
    return out[['TEAM_ID', 'GAME_DATE', 'DAYS_REST_SOURCE']]


def calculate_current_streak(results: Sequence[float]) -> List[int]:
    """Calculates the pre-game streak based on past win/loss values."""
    streaks: List[int] = []
    current = 0
    for value in results:
        if pd.isna(value):
            current = 0
        elif value == 1:
            current = current + 1 if current >= 0 else 1
        else:
            current = current - 1 if current <= 0 else -1
        streaks.append(current)
    return streaks


def features_enhanced(df: pd.DataFrame, config: Dict[str, object]) -> pd.DataFrame:
    """Generates advanced features for each team before every game."""
    df = df.copy()

    if 'GAME_DATE' not in df.columns:
        raise ValueError('GAME_DATE es obligatorio para calcular las nuevas features.')

    days_rest_path = str(config.get('days_rest_path', ''))
    if days_rest_path:
        days_rest_ref = load_days_rest_reference(days_rest_path)
    else:
        days_rest_ref = pd.DataFrame()

    if not days_rest_ref.empty:
        df = df.merge(days_rest_ref, on=['TEAM_ID', 'GAME_DATE'], how='left')

    if 'ROLL10_PACE' in df.columns and 'PACE' not in df.columns:
        df['PACE'] = df['ROLL10_PACE']
    if 'ROLL10_TOV' in df.columns and 'ROLL10_POSS' in df.columns:
        ratio = df['ROLL10_TOV'] / df['ROLL10_POSS'].replace({0: np.nan})
        df['TURNOVER_RATIO'] = ratio.replace([np.inf, -np.inf], np.nan)

    season_col = None
    for cand in ['SEASON_KEY', 'SEASON', 'SEASON_ID', 'SEASON_YEAR']:
        if cand in df.columns:
            season_col = cand
            break

    group_cols = ['TEAM_ID'] + ([season_col] if season_col else [])
    df = df.sort_values(group_cols + ['GAME_DATE'])

    def process_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values('GAME_DATE').copy()
        results = group['WL_NUM'].astype(float)
        prev_results = results.shift(1)

        group['WIN_STREAK'] = calculate_current_streak(prev_results)

        last5 = prev_results.rolling(5, min_periods=1).mean()
        group['LAST_5_PCT'] = last5.fillna(0.5)

        wins = prev_results.fillna(0).cumsum()
        games = prev_results.expanding().count()
        losses = games - wins
        pct = wins.div(games.where(games > 0, np.nan))
        group['SEASON_W_PCT'] = pct.fillna(0.5)
        group['SEASON_WINS'] = wins
        group['SEASON_LOSSES'] = losses

        if 'DAYS_REST_SOURCE' in group.columns:
            group['DAYS_REST'] = group['DAYS_REST_SOURCE']
        else:
            group['DAYS_REST'] = np.nan
        fallback = group['GAME_DATE'].diff().dt.days
        group['DAYS_REST'] = group['DAYS_REST'].fillna(fallback)
        group['DAYS_REST'] = group['DAYS_REST'].fillna(DEFAULT_DAYS_REST).clip(lower=0)

        return group

    df = df.groupby(group_cols, group_keys=False).apply(process_group).reset_index(drop=True)
    df = df.drop(columns=['DAYS_REST_SOURCE'], errors='ignore')

    return df


def features_venue(
    df: pd.DataFrame,
    venue_path: str = "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00c_final/2024-25/dashboards/team_dashboard_by_general_splits__dataset_1.parquet",
) -> pd.DataFrame:
    """Merges home/road split information into the dataframe."""
    df = df.copy()

    def _to_key_str(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip()
        return s.str.replace(r"\.0$", "", regex=True)

    def _flatten_scalar(x):
        if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
            return x[0] if len(x) > 0 else np.nan
        return x

    if 'TEAM_ID' not in df.columns and 'team_id' in df.columns:
        df = df.rename(columns={'team_id': 'TEAM_ID'})
    elif 'TEAM_ID' in df.columns and 'team_id' in df.columns:
        df = df.drop(columns=['team_id'], errors='ignore')

    if 'TEAM_ID' not in df.columns:
        raise ValueError('Tu df no tiene TEAM_ID para poder hacer el merge.')

    df['TEAM_ID'] = df['TEAM_ID'].map(_flatten_scalar)
    df['_TEAM_KEY'] = _to_key_str(df['TEAM_ID'])

    venue_path = Path(venue_path)
    if not venue_path.exists():
        raise FileNotFoundError(f"No existe el parquet: {venue_path}")

    v = pd.read_parquet(venue_path).copy()
    v.columns = [str(c).strip().upper() for c in v.columns]
    v = v.loc[:, ~pd.Index(v.columns).duplicated(keep='first')]

    if 'TEAM_ID' not in v.columns and 'TEAMID' in v.columns:
        v = v.rename(columns={'TEAMID': 'TEAM_ID'})
    elif 'TEAM_ID' not in v.columns and 'TEAM_ID'.lower() in [c.lower() for c in v.columns]:
        cand = [c for c in v.columns if c.lower() == 'team_id']
        if cand:
            v = v.rename(columns={cand[0]: 'TEAM_ID'})

    if 'TEAM_ID' not in v.columns:
        raise ValueError(f"No se encontró TEAM_ID en el parquet. Columnas: {sorted(v.columns)}")

    v['TEAM_ID'] = v['TEAM_ID'].map(_flatten_scalar)
    v['_TEAM_KEY'] = _to_key_str(v['TEAM_ID'])

    if 'W_PCT' not in v.columns:
        if {'W', 'L'}.issubset(v.columns):
            tot = v['W'] + v['L']
            v['W_PCT'] = np.where(tot > 0, v['W'] / tot, np.nan)
        elif {'WINS', 'LOSSES'}.issubset(v.columns):
            tot = v['WINS'] + v['LOSSES']
            v['W_PCT'] = np.where(tot > 0, v['WINS'] / tot, np.nan)
        else:
            raise ValueError('No hay W_PCT ni (W,L) ni (WINS,LOSSES) para calcular W_PCT.')

    needed = {'GROUP_SET', 'GROUP_VALUE', '_TEAM_KEY', 'W_PCT'}
    if not needed.issubset(v.columns):
        raise ValueError(f"Faltan columnas en splits: {needed} | Reales: {sorted(v.columns)}")

    v['GROUP_SET'] = v['GROUP_SET'].astype(str).str.strip().str.lower()
    v['GROUP_VALUE'] = v['GROUP_VALUE'].astype(str).str.strip().str.title()

    map_val = {
        'Home': 'Home',
        'Local': 'Home',
        'Casa': 'Home',
        'Road': 'Road',
        'Away': 'Road',
        'Visitor': 'Road',
        'Visita': 'Road',
        'Fuera': 'Road',
    }
    v['GROUP_VALUE'] = v['GROUP_VALUE'].map(lambda x: map_val.get(x, x))

    vloc = v[(v['GROUP_SET'] == 'location') & (v['GROUP_VALUE'].isin(['Home', 'Road']))].copy()

    agg = (
        vloc.groupby(['_TEAM_KEY', 'GROUP_VALUE'], as_index=False)['W_PCT']
        .mean()
        .rename(columns={'W_PCT': 'VAL'})
    )

    wide = (
        agg.pivot(index='_TEAM_KEY', columns='GROUP_VALUE', values='VAL')
        .rename(columns={'Home': 'VENUE_HOME_W_PCT', 'Road': 'VENUE_ROAD_W_PCT'})
        .reset_index()
    )

    venue_cols = ['VENUE_HOME_W_PCT', 'VENUE_ROAD_W_PCT']
    df = df.drop(columns=[c for c in venue_cols if c in df.columns], errors='ignore')
    before_shape = df.shape
    right_cols = ['_TEAM_KEY'] + [c for c in venue_cols if c in wide.columns]
    df = df.merge(wide[right_cols], how='left', on='_TEAM_KEY')

    imput: Dict[str, float] = {}
    for c in venue_cols:
        if c in df.columns:
            med = df[c].median()
            if pd.notna(med):
                df[c] = df[c].fillna(med)
                imput[c] = float(med)
            else:
                df[c] = df[c].fillna(0.5)
                imput[c] = 0.5

    df = df.drop(columns=['_TEAM_KEY'], errors='ignore')

    print(f"OK: Splits por sede añadidos desde '{venue_path.name}'")
    print(f"Shape: {before_shape} -> {df.shape} (merge left; columnas añadidas al final)")
    print("Columnas nuevas presentes:", [c for c in venue_cols if c in df.columns])
    if imput:
        print("Imputaciones aplicadas:", imput)

    return df


def build_match_dataset_enhanced(
    df: pd.DataFrame,
    numeric_feature_prefixes: Sequence[str] = ("ROLL10_", "VENUE_"),
    advanced_features: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Builds the home/away match-level dataset with differential features."""
    d = df.copy()

    d['IS_HOME'] = d['MATCHUP'].astype(str).str.contains('vs', case=False).astype(int)

    home = d[d['IS_HOME'] == 1].copy().add_prefix('HOME_')
    away = d[d['IS_HOME'] == 0].copy().add_prefix('AWAY_')

    merged = pd.merge(
        home,
        away,
        left_on='HOME_GAME_ID',
        right_on='AWAY_GAME_ID',
        how='inner',
    )

    merged['y'] = (merged['HOME_WL'].astype(str).str.upper().str.strip() == 'W').astype(int)

    bases: List[str] = []
    for col in merged.columns:
        if not (col.startswith('HOME_') or col.startswith('AWAY_')):
            continue
        base = col.replace('HOME_', '').replace('AWAY_', '')
        if base.startswith(tuple(numeric_feature_prefixes)) and merged[col].dtype.kind in 'if':
            bases.append(base)
    bases = sorted(set(bases))

    X_rel = pd.DataFrame(index=merged.index)
    for base in bases:
        h, a = f'HOME_{base}', f'AWAY_{base}'
        if h in merged.columns and a in merged.columns:
            X_rel[f'DIFF_{base}'] = merged[h] - merged[a]

    X_rel['HOME_COURT'] = 1

    if 'HOME_VENUE_HOME_W_PCT' in merged.columns and 'AWAY_VENUE_ROAD_W_PCT' in merged.columns:
        X_rel['DIFF_VENUE_W_PCT'] = (
            merged['HOME_VENUE_HOME_W_PCT'] - merged['AWAY_VENUE_ROAD_W_PCT']
        )

    if advanced_features is None:
        advanced_features = [
            'WIN_STREAK',
            'LAST_5_PCT',
            'DAYS_REST',
            'SEASON_W_PCT',
            'PACE',
            'TURNOVER_RATIO',
        ]

    for feat in advanced_features:
        h_feat = f'HOME_{feat}'
        a_feat = f'AWAY_{feat}'
        if h_feat in merged.columns and a_feat in merged.columns:
            X_rel[f'DIFF_{feat}'] = merged[h_feat] - merged[a_feat]

    if 'HOME_DAYS_REST' in merged.columns and 'AWAY_DAYS_REST' in merged.columns:
        home_b2b = (merged['HOME_DAYS_REST'].fillna(DEFAULT_DAYS_REST) == 0).astype(int)
        away_b2b = (merged['AWAY_DAYS_REST'].fillna(DEFAULT_DAYS_REST) == 0).astype(int)
        X_rel['B2B_ADVANTAGE'] = away_b2b - home_b2b

    y = merged['y'].values

    aux_cols = [
        'HOME_TEAM_ABBREVIATION',
        'AWAY_TEAM_ABBREVIATION',
        'HOME_GAME_DATE',
        'HOME_TEAM_ID',
        'AWAY_TEAM_ID',
        'HOME_SEASON_KEY',
        'AWAY_SEASON_KEY',
    ]
    aux_cols = [c for c in aux_cols if c in merged.columns]
    meta = merged[aux_cols].copy()

    allowed_exact = {'HOME_COURT', 'B2B_ADVANTAGE'}
    allowed_prefixes = (
        'DIFF_ROLL10_',
        'DIFF_VENUE_',
        'DIFF_WIN_STREAK',
        'DIFF_LAST_5_PCT',
        'DIFF_DAYS_REST',
        'DIFF_SEASON_',
        'DIFF_PACE',
        'DIFF_TURNOVER_RATIO',
    )
    for col in X_rel.columns:
        if col in allowed_exact:
            continue
        if any(col.startswith(pref) for pref in allowed_prefixes):
            continue
        raise AssertionError(
            f'Se detectó una columna no permitida: {col}. Revisa los prefijos aceptados.'
        )

    return X_rel, y, meta


def fit_and_eval(
    X_tr: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_tr: Sequence[int],
    y_val: Sequence[int],
    y_test: Sequence[int],
    model_name: str,
    random_state: int = 42,
):
    """Fits an XGBoost model and evaluates it on validation/test splits."""
    common_kwargs = dict(
        n_estimators=1200,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=2,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=-1,
    )

    try:
        model = XGBClassifier(early_stopping_rounds=75, **common_kwargs)
    except TypeError:
        model = XGBClassifier(**common_kwargs)

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    use_ntree_limit = {}
    if hasattr(model, 'best_iteration_') and model.best_iteration_ is not None:
        use_ntree_limit = {'iteration_range': (0, model.best_iteration_ + 1)}

    proba_val = model.predict_proba(X_val, **use_ntree_limit)[:, 1]
    proba_test = model.predict_proba(X_test, **use_ntree_limit)[:, 1]

    fpr_val, tpr_val, thresholds_val = roc_curve(y_val, proba_val)
    youden_scores = tpr_val - fpr_val
    best_idx = int(np.argmax(youden_scores))
    if 0 <= best_idx < len(thresholds_val) and np.isfinite(thresholds_val[best_idx]):
        best_thresh = float(thresholds_val[best_idx])
    else:
        best_thresh = 0.5

    y_pred = (proba_test >= best_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    metrics_dict = {
        'accuracy': float((y_pred == y_test).mean()),
        'balanced_acc': float(
            np.nanmean([
                (y_pred[y_test == 1] == 1).mean() if np.any(y_test == 1) else np.nan,
                (y_pred[y_test == 0] == 0).mean() if np.any(y_test == 0) else np.nan,
            ])
        ),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, proba_test)),
        'avg_precision': float(average_precision_score(y_test, proba_test)),
        'brier': float(brier_score_loss(y_test, proba_test)),
        'log_loss': float(log_loss(y_test, proba_test)),
        'best_threshold': float(best_thresh),
    }

    return {
        'model': model,
        'proba_val': proba_val,
        'proba_test': proba_test,
        'y_pred': y_pred,
        'best_threshold': best_thresh,
        'confusion_matrix': cm,
        'metrics': metrics_dict,
        'model_name': model_name,
    }
