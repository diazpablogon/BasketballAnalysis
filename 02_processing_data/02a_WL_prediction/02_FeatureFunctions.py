"""Utility functions to compute basketball features and train win/loss estimators.
Versión con MERGE SEGURO para Days Rest usando clave string '_REST_KEY' = TEAM_ID|YYYY-MM-DD
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
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

DEFAULT_LINEUP_CONFIG = {
    'N_ROLL_MIN': 5,
    'TEAM_PCTL': (5, 95),
    'USE_ON_OFF': False,
    'USE_LINEUPS': True,
    'MIN_EXP_FALLBACK': 15,
    'MIN_BENCH_THRESHOLD': 15,
    'LINEUP_WEIGHTS': {
        'EFF_RATING': 0.55,
        'BENCH_DEPTH': 0.25,
        'AVAIL_PENALTY': 0.20,
    },
}

# Métricas externas por días de descanso que se consideran útiles para el modelo.
# Formato: columna_origen -> (función_agg, nombre_feature_destino)
REST_METRIC_RULES: Dict[str, Tuple[str, str]] = {
    'GP': ('sum', 'REST_BUCKET_GP'),
    'W_PCT': ('mean', 'REST_BUCKET_WIN_PCT'),
    'PLUS_MINUS': ('mean', 'REST_BUCKET_PLUS_MINUS'),
    'PTS': ('mean', 'REST_BUCKET_POINTS'),
}

__all__ = [
    'DEFAULT_DAYS_REST',
    'DEFAULT_LINEUP_CONFIG',
    'ensure_teamid_and_date',
    'features_roll10',
    'parse_days_rest_value',
    'load_days_rest_reference',
    'add_days_rest_from_reference',
    'calculate_current_streak',
    'features_enhanced',
    'features_venue',
    'build_match_dataset_enhanced',
    'fit_and_eval',
    'prep_dates',
    'build_player_to_date_in_memory',
    'compute_on_off_to_date_in_memory',
    'process_lineups_data',
    'compute_lineup_metrics_for_game',
    'add_lineup_features_in_memory',
]


# =========================
# Helpers de normalización
# =========================
def _flatten_scalar(x):
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return x[0] if len(x) > 0 else np.nan
    return x

def _to_key_str(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return s.str.replace(r"\.0$", "", regex=True)

def ensure_teamid_and_date(df: pd.DataFrame, team_col: str = 'TEAM_ID', date_col: str = 'GAME_DATE') -> pd.DataFrame:
    """Normaliza TEAM_ID a Int64 (nullable) y GAME_DATE a datetime (naive). Devuelve copia."""
    d = df.copy()
    if team_col in d.columns:
        d[team_col] = d[team_col].map(_flatten_scalar)
        d[team_col] = pd.to_numeric(d[team_col], errors='coerce').astype('Int64')
    if date_col in d.columns:
        d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
    return d

def _build_rest_key(df: pd.DataFrame, team_col='TEAM_ID', date_col='GAME_DATE', key='_REST_KEY') -> pd.DataFrame:
    """Crea una clave string segura TEAM_ID|YYYY-MM-DD para merges sin conflictos de dtype."""
    d = df.copy()
    if team_col not in d.columns or date_col not in d.columns:
        raise ValueError(f"Para construir {key} faltan columnas: {team_col} y/o {date_col}.")
    # Asegura tipos y luego pasa a string canónica
    d = ensure_teamid_and_date(d, team_col=team_col, date_col=date_col)
    team_str = _to_key_str(d[team_col].astype(object))
    date_str = pd.to_datetime(d[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
    d[key] = team_str.fillna('NA') + '|' + date_str.fillna('NA')
    return d


# =========================
# ROLL10 selector/limpieza
# =========================
def features_roll10(df: pd.DataFrame) -> pd.DataFrame:
    """Selects and cleans pre-computed rolling features with a 10-game window."""
    df = ensure_teamid_and_date(df)

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

    if 'GAME_DATE' in df.columns and df['GAME_DATE'].notna().any():
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
        print(f"Columnas ROLL10 imputadas (mediana): {len(roll_cols)}")

    return df


# =========================
# Days Rest utilities
# =========================
def parse_days_rest_value(value):
    """Converts the rest range text to a numeric approximation (simple)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    m = re.search(r"\d+", text)
    if m:
        return float(m.group())
    lowered = text.lower()
    if 'back' in lowered:
        return 0.0
    return None


def _normalize_rest_range_label(value: object) -> Optional[str]:
    """Normaliza etiquetas como "1 Day Rest"/"3+ Days Rest" -> formato consistente."""

    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    lowered = text.lower()
    if 'back' in lowered:
        return '0 Days Rest'

    num = parse_days_rest_value(text)
    if num is None:
        return text.title()

    is_plus = '+' in text or 'plus' in lowered or 'more' in lowered or '>=' in lowered

    num_int = int(max(0, round(num)))
    if is_plus and num_int >= 0:
        return f"{num_int}+ Days Rest"

    if num_int == 1:
        return '1 Day Rest'

    return f"{num_int} Days Rest"

def _find_column_case_insensitive(columns: pd.Index, *aliases: str) -> str | None:
    """Return the first matching column name regardless of case."""

    col_map = {str(c).strip(): c for c in columns}
    lower_map = {str(c).strip().lower(): c for c in columns}

    for cand in aliases:
        if cand in col_map:
            return col_map[cand]
        cand_lower = cand.lower()
        if cand_lower in lower_map:
            return lower_map[cand_lower]
    return None


# =========================
# Lineup feature utilities
# =========================
def _resolve_lineup_config(config: Optional[Dict]) -> Dict:
    cfg = DEFAULT_LINEUP_CONFIG.copy()
    if config is None:
        return cfg
    for key, value in config.items():
        if key == 'LINEUP_WEIGHTS':
            weights = cfg['LINEUP_WEIGHTS'].copy()
            if isinstance(value, dict):
                weights.update(value)
            cfg['LINEUP_WEIGHTS'] = weights
        else:
            cfg[key] = value
    return cfg


def prep_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Return a copy with normalized naive datetimes sorted ascending."""

    if date_col not in df.columns:
        raise KeyError(f"La columna de fecha '{date_col}' no existe en el DataFrame.")

    result = df.copy()
    dates = pd.to_datetime(result[date_col], errors='coerce')
    if ptypes.is_datetime64tz_dtype(dates.dtype):
        dates = dates.dt.tz_localize(None)
    result[date_col] = dates
    result = result.sort_values(date_col).reset_index(drop=True)
    return result


def build_player_to_date_in_memory(
    player_box: pd.DataFrame,
    team_games: pd.DataFrame,
    mean_cols: Sequence[str],
    roll_cols: Sequence[str],
    config: Optional[Dict] = None,
) -> pd.DataFrame:
    """Build historical player metrics with strict temporal anti-leakage safeguards."""

    cfg = _resolve_lineup_config(config)
    required = {'GAME_ID', 'TEAM_ID', 'PLAYER_ID'}
    missing = required - set(player_box.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en player_box: {sorted(missing)}")

    if {'GAME_ID', 'TEAM_ID', 'GAME_DATE'}.difference(team_games.columns):
        raise ValueError("team_games debe contener GAME_ID, TEAM_ID y GAME_DATE")

    team_dates = team_games[['GAME_ID', 'TEAM_ID', 'GAME_DATE']].drop_duplicates()
    team_dates['GAME_DATE'] = pd.to_datetime(team_dates['GAME_DATE'], errors='coerce')

    base = player_box.copy()
    for col in ['GAME_ID', 'TEAM_ID', 'PLAYER_ID']:
        base[col] = pd.to_numeric(base[col], errors='coerce')

    merged = base.merge(team_dates, on=['GAME_ID', 'TEAM_ID'], how='inner', validate='many_to_one')
    merged['GAME_DATE'] = pd.to_datetime(merged['GAME_DATE'], errors='coerce')
    merged['TEAM_ID'] = pd.to_numeric(merged['TEAM_ID'], errors='coerce').astype('Int64')
    merged['PLAYER_ID'] = pd.to_numeric(merged['PLAYER_ID'], errors='coerce').astype('Int64')
    merged['GAME_ID'] = pd.to_numeric(merged['GAME_ID'], errors='coerce').astype('Int64')
    merged = merged.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

    # Prepare numeric conversions for computation
    for col in set(mean_cols) | set(roll_cols):
        if col not in merged.columns:
            merged[col] = np.nan
        else:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')

    group_key = merged.groupby('PLAYER_ID', group_keys=False)

    for col in mean_cols:
        col_name = f"{col}_to_date"

        def _mean_shift(series: pd.Series) -> pd.Series:
            return series.expanding().mean().shift(1)

        merged[col_name] = group_key[col].transform(_mean_shift)

    window = int(cfg.get('N_ROLL_MIN', 5))
    window = max(1, window)
    for col in roll_cols:
        col_name = f"{col}_roll{window}_prev"

        def _roll_shift(series: pd.Series) -> pd.Series:
            return series.rolling(window=window, min_periods=1).mean().shift(1)

        merged[col_name] = group_key[col].transform(_roll_shift)

    return merged


def compute_on_off_to_date_in_memory(
    on_df: Optional[pd.DataFrame],
    off_df: Optional[pd.DataFrame],
    team_games: pd.DataFrame,
) -> pd.DataFrame:
    """Compute rolling on/off metrics when per-game data is available.

    The function inspects the structure of the provided DataFrames and only
    computes historical averages when GAME_ID is present. If the supplied data
    is aggregated at the season level (lacking GAME_ID), it returns an empty
    DataFrame to avoid temporal leakage while signalling the condition.
    """

    empty = pd.DataFrame(
        columns=['PLAYER_ID', 'GAME_DATE', 'NET_ON_to_date', 'NET_OFF_to_date', 'DELTA_NET_to_date']
    )

    if on_df is None or off_df is None:
        return empty

    if on_df.empty or off_df.empty:
        return empty

    has_game_id = 'GAME_ID' in on_df.columns and 'GAME_ID' in off_df.columns

    if not has_game_id:
        print(
            "⚠️  ON/OFF DESACTIVADO - Datos agregados por temporada sin GAME_ID."
            " Se omiten para prevenir leakage."
        )
        return empty

    required_cols = {'PLAYER_ID', 'GAME_ID', 'NET_RATING'}

    missing_on = required_cols - set(on_df.columns)
    missing_off = required_cols - set(off_df.columns)
    if missing_on:
        raise ValueError(f"Faltan columnas en on_df: {sorted(missing_on)}")
    if missing_off:
        raise ValueError(f"Faltan columnas en off_df: {sorted(missing_off)}")

    if {'GAME_ID', 'GAME_DATE'}.difference(team_games.columns):
        raise ValueError("team_games debe contener GAME_ID y GAME_DATE para cruzar on/off")

    team_dates = team_games[['GAME_ID', 'GAME_DATE']].drop_duplicates()
    team_dates['GAME_DATE'] = pd.to_datetime(team_dates['GAME_DATE'], errors='coerce')

    def _prep(df: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
        subset = df[['PLAYER_ID', 'GAME_ID', value_col]].copy()
        subset['PLAYER_ID'] = pd.to_numeric(subset['PLAYER_ID'], errors='coerce').astype('Int64')
        subset['GAME_ID'] = pd.to_numeric(subset['GAME_ID'], errors='coerce').astype('Int64')
        subset[value_col] = pd.to_numeric(subset[value_col], errors='coerce')
        subset = subset.merge(team_dates, on='GAME_ID', how='inner', validate='many_to_one')
        subset = subset.dropna(subset=['PLAYER_ID', 'GAME_ID', 'GAME_DATE'])
        subset = subset.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
        subset = subset.groupby(['PLAYER_ID', 'GAME_DATE'], as_index=False)[value_col].mean()
        subset[out_col] = (
            subset.groupby('PLAYER_ID')[value_col]
            .expanding(min_periods=1)
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
        )
        subset = subset.drop(columns=[value_col])
        return subset

    on_hist = _prep(on_df, 'NET_RATING', 'NET_ON_to_date')
    off_hist = _prep(off_df, 'NET_RATING', 'NET_OFF_to_date')

    hist = on_hist.merge(off_hist, on=['PLAYER_ID', 'GAME_DATE'], how='outer')
    hist = hist.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
    hist['DELTA_NET_to_date'] = hist['NET_ON_to_date'] - hist['NET_OFF_to_date']

    return hist


def process_lineups_data(
    lineups_df: Optional[pd.DataFrame],
    team_games: pd.DataFrame,
) -> pd.DataFrame:
    """Identify historical starters per team-date using either game-level or season-level lineups."""

    if lineups_df is None or lineups_df.empty:
        return pd.DataFrame(columns=['TEAM_ID', 'GAME_DATE', 'STARTING_LINEUP'])

    has_game_level = {'GAME_ID', 'PLAYER_ID'}.issubset(lineups_df.columns)

    if has_game_level:
        if {'GAME_ID', 'TEAM_ID', 'GAME_DATE'}.difference(team_games.columns):
            raise ValueError("team_games debe contener GAME_ID, TEAM_ID y GAME_DATE")

        team_dates = team_games[['GAME_ID', 'TEAM_ID', 'GAME_DATE']].drop_duplicates()
        team_dates['GAME_DATE'] = pd.to_datetime(team_dates['GAME_DATE'], errors='coerce')

        lineups = lineups_df[['TEAM_ID', 'GAME_ID', 'PLAYER_ID', 'MIN']].copy()
        for col in ['TEAM_ID', 'GAME_ID', 'PLAYER_ID']:
            lineups[col] = pd.to_numeric(lineups[col], errors='coerce')
        lineups['MIN'] = pd.to_numeric(lineups['MIN'], errors='coerce').fillna(0.0)

        agg = (
            lineups.groupby(['TEAM_ID', 'GAME_ID', 'PLAYER_ID'], as_index=False)['MIN']
            .sum()
        )
        agg = agg.merge(team_dates, on=['TEAM_ID', 'GAME_ID'], how='inner', validate='many_to_one')
        agg['GAME_DATE'] = pd.to_datetime(agg['GAME_DATE'], errors='coerce')
        agg['TEAM_ID'] = pd.to_numeric(agg['TEAM_ID'], errors='coerce').astype('Int64')
        agg['PLAYER_ID'] = pd.to_numeric(agg['PLAYER_ID'], errors='coerce').astype('Int64')
        agg['GAME_ID'] = pd.to_numeric(agg['GAME_ID'], errors='coerce').astype('Int64')
        agg = agg.sort_values(['TEAM_ID', 'PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

        agg['MIN_CUM'] = agg.groupby(['TEAM_ID', 'PLAYER_ID'])['MIN'].cumsum()
        agg['MIN_CUM_PREV'] = (agg['MIN_CUM'] - agg['MIN']).clip(lower=0)
        agg = agg.drop(columns=['MIN_CUM'])

        agg = agg.sort_values(['TEAM_ID', 'GAME_DATE', 'MIN_CUM_PREV'], ascending=[True, True, False])

        starters = (
            agg.groupby(['TEAM_ID', 'GAME_DATE'])['PLAYER_ID']
            .apply(lambda s: s.head(5).dropna().astype('Int64').tolist())
            .reset_index(name='STARTING_LINEUP')
        )

        return starters

    required = {'TEAM_ID', 'GROUP_ID', 'GP'}
    missing = required - set(lineups_df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en lineups_df: {sorted(missing)}")

    if {'TEAM_ID', 'GAME_DATE'}.difference(team_games.columns):
        raise ValueError("team_games debe contener TEAM_ID y GAME_DATE")

    team_dates = team_games[['TEAM_ID', 'GAME_DATE']].drop_duplicates()
    team_dates['TEAM_ID'] = pd.to_numeric(team_dates['TEAM_ID'], errors='coerce')
    team_dates = team_dates.dropna(subset=['TEAM_ID'])
    team_dates['TEAM_ID'] = team_dates['TEAM_ID'].astype('Int64')
    team_dates['GAME_DATE'] = pd.to_datetime(team_dates['GAME_DATE'], errors='coerce')
    team_dates = team_dates.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)

    aggregated = lineups_df[list(required)].copy()
    aggregated['TEAM_ID'] = pd.to_numeric(aggregated['TEAM_ID'], errors='coerce')
    aggregated = aggregated.dropna(subset=['TEAM_ID'])
    aggregated['TEAM_ID'] = aggregated['TEAM_ID'].astype('Int64')
    aggregated['GP'] = pd.to_numeric(aggregated['GP'], errors='coerce').fillna(0.0)

    def _parse_group(group_id: object) -> List[int]:
        if pd.isna(group_id):
            return []
        tokens = str(group_id).strip().split('-')
        parsed: List[int] = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            token = re.sub(r"\.0$", "", token)
            try:
                value = int(float(token))
            except (TypeError, ValueError):
                continue
            parsed.append(value)
        return parsed

    aggregated['STARTING_LINEUP'] = aggregated['GROUP_ID'].apply(_parse_group)
    aggregated = aggregated[aggregated['STARTING_LINEUP'].map(len) > 0]

    if aggregated.empty:
        team_dates['STARTING_LINEUP'] = [[] for _ in range(len(team_dates))]
        return team_dates[['TEAM_ID', 'GAME_DATE', 'STARTING_LINEUP']]

    sort_cols = ['TEAM_ID', 'GP']
    ascending = [True, False]

    if 'MIN' in lineups_df.columns:
        aggregated['MIN'] = pd.to_numeric(lineups_df.loc[aggregated.index, 'MIN'], errors='coerce').fillna(0.0)
        sort_cols.append('MIN')
        ascending.append(False)

    aggregated = aggregated.sort_values(sort_cols, ascending=ascending)
    aggregated = aggregated.drop_duplicates('TEAM_ID', keep='first')

    starters = team_dates.merge(aggregated[['TEAM_ID', 'STARTING_LINEUP']], on='TEAM_ID', how='left')
    starters['STARTING_LINEUP'] = starters['STARTING_LINEUP'].apply(
        lambda x: [int(v) for v in x if pd.notna(v)] if isinstance(x, (list, tuple, np.ndarray)) else []
    )

    starters = starters[['TEAM_ID', 'GAME_DATE', 'STARTING_LINEUP']]
    starters = starters.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)

    return starters


def compute_lineup_metrics_for_game(
    team_id: int,
    game_id: int,
    game_date: pd.Timestamp,
    player_hist: pd.DataFrame,
    onoff_hist: Optional[pd.DataFrame],
    starters_lookup,
    config: Optional[Dict] = None,
) -> Dict[str, float]:
    """Compute lineup metrics for a single team-game ensuring no temporal leakage."""

    cfg = _resolve_lineup_config(config)
    fallback = {
        'LINEUP_EFF_RATING': np.nan,
        'LINEUP_EFF_ADJ': 0.0,
        'LINEUP_STARTERS_OUT': 0.0,
        'LINEUP_BENCH_DEPTH': 0.0,
        'LINEUP_MIN_VAR': np.nan,
        'LINEUP_AVAIL_PENALTY': 0.0,
    }

    if player_hist is None or player_hist.empty:
        return fallback

    mask = (player_hist['TEAM_ID'] == team_id) & (player_hist['GAME_ID'] == game_id)
    players = player_hist.loc[mask].copy()
    if players.empty:
        return fallback

    players['GAME_DATE'] = pd.to_datetime(players['GAME_DATE'], errors='coerce')
    players['PLAYER_ID'] = pd.to_numeric(players['PLAYER_ID'], errors='coerce').astype('Int64')

    use_on_off = bool(cfg.get('USE_ON_OFF', False))
    if use_on_off and onoff_hist is not None and not onoff_hist.empty:
        onoff_hist = onoff_hist.copy()
        onoff_hist['GAME_DATE'] = pd.to_datetime(onoff_hist['GAME_DATE'], errors='coerce')
        onoff_hist['PLAYER_ID'] = pd.to_numeric(onoff_hist['PLAYER_ID'], errors='coerce').astype('Int64')
        players = players.merge(
            onoff_hist,
            on=['PLAYER_ID', 'GAME_DATE'],
            how='left',
        )

    roll_col = f"MIN_roll{int(cfg.get('N_ROLL_MIN', 5))}_prev"
    if roll_col in players.columns:
        min_exp = pd.to_numeric(players[roll_col], errors='coerce')
    else:
        min_exp = pd.Series(np.nan, index=players.index)
    players['MIN_EXPECTED'] = min_exp.fillna(cfg['MIN_EXP_FALLBACK'])

    weight = players['MIN_EXPECTED'].clip(lower=0)
    total_minutes = weight.sum()

    def _weighted_average(values: pd.Series) -> float:
        vals = pd.to_numeric(values, errors='coerce')
        mask_valid = vals.notna() & weight.notna() & (weight > 0)
        if mask_valid.any():
            local_weights = weight[mask_valid]
            if local_weights.sum() > 0:
                return float(np.average(vals[mask_valid], weights=local_weights))
        return np.nan

    eff_rating = np.nan
    if 'NET_RATING_to_date' in players.columns:
        eff_rating = _weighted_average(players['NET_RATING_to_date'])

    if np.isnan(eff_rating):
        proxy_components = {
            'PTS_to_date': 1.0,
            'REB_to_date': 1.0,
            'AST_to_date': 1.0,
            'STL_to_date': 1.0,
            'BLK_to_date': 1.0,
            'TOV_to_date': -1.0,
        }
        proxy = pd.Series(0.0, index=players.index, dtype=float)
        for col, sign in proxy_components.items():
            if col in players.columns:
                proxy = proxy + pd.to_numeric(players[col], errors='coerce').fillna(0.0) * sign
        mask_weights = weight > 0
        if total_minutes > 0 and mask_weights.any():
            eff_rating = float(np.average(proxy[mask_weights], weights=weight[mask_weights]))

    eff_adj = 0.0
    if use_on_off and 'DELTA_NET_to_date' in players.columns:
        eff_adj_val = _weighted_average(players['DELTA_NET_to_date'])
        if not np.isnan(eff_adj_val):
            eff_adj = float(eff_adj_val)

    starters_out = 0.0
    starters_list: List[int] = []
    lookup_value = None
    if starters_lookup:
        if isinstance(starters_lookup, dict):
            lookup_value = starters_lookup.get((team_id, pd.Timestamp(game_date) if pd.notna(game_date) else game_date))
        elif isinstance(starters_lookup, pd.DataFrame) and not starters_lookup.empty:
            mask_lookup = (starters_lookup['TEAM_ID'] == team_id) & (
                pd.to_datetime(starters_lookup['GAME_DATE']) == pd.Timestamp(game_date)
            )
            if mask_lookup.any():
                lookup_value = starters_lookup.loc[mask_lookup, 'STARTING_LINEUP'].iloc[0]
    if lookup_value is not None:
        if isinstance(lookup_value, list):
            starters_list = [int(x) for x in lookup_value if pd.notna(x)]
        elif pd.notna(lookup_value):
            starters_list = [int(lookup_value)]

    active_players = players['PLAYER_ID'].dropna().astype(int).tolist()
    active_set = set(active_players)
    if starters_list:
        starters_out = float(sum(1 for pid in starters_list if pid not in active_set))

    min_threshold = cfg.get('MIN_BENCH_THRESHOLD', 15)
    bench_candidates = players.copy()
    if starters_list:
        bench_candidates = bench_candidates[~bench_candidates['PLAYER_ID'].isin(starters_list)]
    bench_depth = float(
        bench_candidates.loc[bench_candidates['MIN_EXPECTED'] >= min_threshold, 'PLAYER_ID'].nunique()
    )

    top8 = players.sort_values('MIN_EXPECTED', ascending=False).head(8)
    if top8.empty or top8['MIN_EXPECTED'].isna().all():
        min_var = np.nan
    else:
        mins = pd.to_numeric(top8['MIN_EXPECTED'], errors='coerce').dropna()
        if len(mins) >= 2:
            min_var = float(np.var(mins, ddof=0))
        else:
            min_var = 0.0

    avail_penalty = 0.15 * starters_out

    return {
        'LINEUP_EFF_RATING': float(eff_rating) if not np.isnan(eff_rating) else np.nan,
        'LINEUP_EFF_ADJ': float(eff_adj),
        'LINEUP_STARTERS_OUT': starters_out,
        'LINEUP_BENCH_DEPTH': bench_depth,
        'LINEUP_MIN_VAR': min_var,
        'LINEUP_AVAIL_PENALTY': float(avail_penalty),
    }


def add_lineup_features_in_memory(
    df_teamgames: pd.DataFrame,
    df_player_box: pd.DataFrame,
    df_on: Optional[pd.DataFrame] = None,
    df_off: Optional[pd.DataFrame] = None,
    df_lineups: Optional[pd.DataFrame] = None,
    config: Optional[Dict] = None,
) -> pd.DataFrame:
    """Augment team games with lineup-based features computed without leakage."""

    cfg = _resolve_lineup_config(config)

    required_team_cols = {'GAME_ID', 'TEAM_ID', 'GAME_DATE'}
    missing_team = required_team_cols - set(df_teamgames.columns)
    if missing_team:
        raise ValueError(f"Faltan columnas en df_teamgames: {sorted(missing_team)}")

    required_player_cols = {'GAME_ID', 'TEAM_ID', 'PLAYER_ID'}
    missing_player = required_player_cols - set(df_player_box.columns)
    if missing_player:
        raise ValueError(f"Faltan columnas en df_player_box: {sorted(missing_player)}")

    df_team = prep_dates(df_teamgames, 'GAME_DATE')
    df_team['TEAM_ID'] = pd.to_numeric(df_team['TEAM_ID'], errors='coerce').astype('Int64')
    df_team['GAME_ID'] = pd.to_numeric(df_team['GAME_ID'], errors='coerce').astype('Int64')

    mean_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN', 'NET_RATING', 'TS_PCT', 'USG_PCT']
    roll_cols = ['MIN', 'PTS', 'AST', 'REB']

    player_hist = build_player_to_date_in_memory(
        player_box=df_player_box,
        team_games=df_team,
        mean_cols=mean_cols,
        roll_cols=roll_cols,
        config=cfg,
    )

    if cfg.get('USE_ON_OFF', True):
        onoff_hist = compute_on_off_to_date_in_memory(df_on, df_off, df_team)
    else:
        onoff_hist = pd.DataFrame(columns=['PLAYER_ID', 'GAME_DATE', 'NET_ON_to_date', 'NET_OFF_to_date', 'DELTA_NET_to_date'])

    if cfg.get('USE_LINEUPS', True):
        starters_df = process_lineups_data(df_lineups, df_team)
        if not starters_df.empty:
            starters_df['TEAM_ID'] = pd.to_numeric(starters_df['TEAM_ID'], errors='coerce').astype('Int64')
        starters_lookup = {
            (int(row.TEAM_ID) if pd.notna(row.TEAM_ID) else row.TEAM_ID, pd.Timestamp(row.GAME_DATE)):
            row.STARTING_LINEUP
            for row in starters_df.itertuples()
        }
    else:
        starters_lookup = {}

    teamgames_sorted = df_team.sort_values('GAME_DATE').reset_index(drop=True)

    lineup_records = []
    for row in teamgames_sorted.itertuples(index=False):
        metrics = compute_lineup_metrics_for_game(
            team_id=int(row.TEAM_ID) if pd.notna(row.TEAM_ID) else row.TEAM_ID,
            game_id=int(row.GAME_ID) if pd.notna(row.GAME_ID) else row.GAME_ID,
            game_date=row.GAME_DATE,
            player_hist=player_hist,
            onoff_hist=onoff_hist,
            starters_lookup=starters_lookup,
            config=cfg,
        )
        metrics.update({'TEAM_ID': row.TEAM_ID, 'GAME_ID': row.GAME_ID, 'GAME_DATE': row.GAME_DATE})
        lineup_records.append(metrics)

    if not lineup_records:
        return df_teamgames.copy()

    lineup_df = pd.DataFrame(lineup_records)

    low_pct, high_pct = cfg.get('TEAM_PCTL', (5, 95))

    def _percentile_norm(group: pd.DataFrame, column: str) -> pd.Series:
        history: List[float] = []
        normalized: List[float] = []
        for value in group[column]:
            valid_hist = [h for h in history if not np.isnan(h)]
            if valid_hist and not pd.isna(value):
                p_low = float(np.percentile(valid_hist, low_pct))
                p_high = float(np.percentile(valid_hist, high_pct))
                if np.isclose(p_high, p_low):
                    norm_val = 0.5
                else:
                    norm_val = float(
                        np.clip((float(value) - p_low) / (p_high - p_low), 0.0, 1.0)
                    )
            else:
                norm_val = 0.5

            if pd.isna(value):
                normalized.append(0.5)
            else:
                normalized.append(norm_val if not np.isnan(norm_val) else 0.5)
                history.append(float(value))

        return pd.Series(normalized, index=group.index, dtype=float)

    metrics_to_normalize = {
        'LINEUP_EFF_RATING': None,
        'LINEUP_EFF_ADJ': None,
        'LINEUP_BENCH_DEPTH': None,
        'LINEUP_AVAIL_PENALTY': None,
    }

    lineup_sorted = lineup_df.sort_values(['TEAM_ID', 'GAME_DATE'])
    for metric in metrics_to_normalize:
        lineup_df[f'{metric}_NORM'] = (
            lineup_sorted.groupby('TEAM_ID', group_keys=False)
            .apply(lambda g, col=metric: _percentile_norm(g, col))
        )

    score = np.zeros(len(lineup_df))
    for key, weight in cfg['LINEUP_WEIGHTS'].items():
        metric_name = f'LINEUP_{key}'
        norm_col = f'{metric_name}_NORM'
        if norm_col in lineup_df.columns:
            vals = lineup_df[norm_col].fillna(0.5)
            score += weight * vals

    lineup_df['LINEUP_SCORE'] = np.clip(score, 0.0, 1.0)

    output_cols = [
        'TEAM_ID',
        'GAME_ID',
        'LINEUP_EFF_RATING',
        'LINEUP_EFF_ADJ',
        'LINEUP_STARTERS_OUT',
        'LINEUP_BENCH_DEPTH',
        'LINEUP_MIN_VAR',
        'LINEUP_AVAIL_PENALTY',
        'LINEUP_SCORE',
    ]

    result = df_teamgames.copy()
    result = result.merge(lineup_df[output_cols], on=['TEAM_ID', 'GAME_ID'], how='left')
    return result


def _infer_datetime_column(df: pd.DataFrame, *, min_valid_ratio: float = 0.6) -> str | None:
    """Try to infer a datetime column when an explicit alias is not available."""

    for col in df.columns:
        series = df[col]

        if ptypes.is_datetime64_any_dtype(series):
            return col

        if not (ptypes.is_object_dtype(series) or ptypes.is_string_dtype(series)):
            # Avoid trying to coerce rank/metric numeric columns which can produce bogus datetimes.
            continue

        parsed = pd.to_datetime(series, errors='coerce', utc=True)
        valid_ratio = parsed.notna().mean()
        if valid_ratio >= min_valid_ratio and parsed.notna().any():
            return col

    return None


def load_days_rest_reference(path: str | Path) -> pd.DataFrame:
    """
    Carga un parquet con splits de rendimiento por días de descanso.

    Devuelve columnas agregadas por TEAM_ID + bucket de descanso (DAYS_REST_BUCKET).
    Incluye métricas prefijadas con REST_ (por ejemplo REST_W_PCT) cuando estén disponibles.
    """

    path = Path(path)
    if not path.exists():
        print(
            f"⚠️ No se encontró el parquet de days rest en {path}. Se usará la diferencia de fechas como respaldo."
        )
        return pd.DataFrame()

    rest_df = pd.read_parquet(path)
    if rest_df.empty:
        return pd.DataFrame()

    # Normaliza nombres para detección case-insensitive
    rest_df = rest_df.copy()
    rest_df.columns = [str(c).strip() for c in rest_df.columns]

    team_col = _find_column_case_insensitive(rest_df.columns, 'TEAM_ID', 'TEAMID')
    group_set_col = _find_column_case_insensitive(rest_df.columns, 'GROUP_SET')
    group_value_col = _find_column_case_insensitive(rest_df.columns, 'GROUP_VALUE')
    range_col = _find_column_case_insensitive(
        rest_df.columns,
        'TEAM_DAYS_REST_RANGE',
        'DAYS_REST_RANGE',
        'TEAM_DAYS_REST',
    )

    if range_col is None and group_set_col and group_value_col:
        mask = (
            rest_df[group_set_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq('team_days_rest_range')
        )
        filtered = rest_df.loc[mask].copy()
        if filtered.empty:
            print(
                "⚠️ El parquet de days rest no contiene filas con GROUP_SET == 'team_days_rest_range'."
            )
            return pd.DataFrame()
        rest_df = filtered
        range_col = group_value_col

    if team_col is None or range_col is None:
        missing = []
        if team_col is None:
            missing.append('TEAM_ID')
        if range_col is None:
            missing.append('TEAM_DAYS_REST_RANGE')
        print(
            "⚠️ El parquet de days rest no contiene las columnas requeridas "
            f"{', '.join(missing)}. Se omite. Columnas disponibles: {sorted(map(str, rest_df.columns))}"
        )
        return pd.DataFrame()

    rest_df = rest_df.rename(columns={team_col: 'TEAM_ID', range_col: 'REST_RANGE_LABEL'})
    rest_df['TEAM_ID'] = pd.to_numeric(rest_df['TEAM_ID'].map(_flatten_scalar), errors='coerce').astype('Int64')
    rest_df = rest_df.dropna(subset=['TEAM_ID'])

    rest_df['REST_RANGE_LABEL'] = rest_df['REST_RANGE_LABEL'].map(_normalize_rest_range_label)
    rest_df = rest_df.dropna(subset=['REST_RANGE_LABEL'])

    # Calcula bucket numérico para merge (p.e. 3+ Days Rest -> 3)
    rest_df['DAYS_REST_BUCKET'] = rest_df['REST_RANGE_LABEL'].map(parse_days_rest_value)
    rest_df['DAYS_REST_BUCKET'] = (
        pd.to_numeric(rest_df['DAYS_REST_BUCKET'], errors='coerce').round().astype('Int64')
    )
    rest_df = rest_df.dropna(subset=['DAYS_REST_BUCKET'])

    # Selecciona únicamente las métricas consideradas útiles
    selected_metrics: Dict[str, Tuple[str, str]] = {}
    for src_name, (agg_fn, alias) in REST_METRIC_RULES.items():
        real_col = _find_column_case_insensitive(rest_df.columns, src_name)
        if real_col:
            selected_metrics[real_col] = (agg_fn, alias)

    if not selected_metrics:
        print(
            "⚠️ El parquet de days rest no contiene las métricas necesarias ("
            f"{sorted(REST_METRIC_RULES)})."
        )
        return pd.DataFrame()

    metric_cols = list(selected_metrics.keys())

    for col in metric_cols:
        rest_df[col] = pd.to_numeric(rest_df[col], errors='coerce')

    agg_dict = {col: selected_metrics[col][0] for col in metric_cols}
    agg_dict['REST_RANGE_LABEL'] = 'first'

    grouped = (
        rest_df[['TEAM_ID', 'DAYS_REST_BUCKET', 'REST_RANGE_LABEL'] + metric_cols]
        .groupby(['TEAM_ID', 'DAYS_REST_BUCKET'], as_index=False)
        .agg(agg_dict)
    )

    rename_metrics = {col: selected_metrics[col][1] for col in metric_cols}

    grouped = grouped.rename(columns=rename_metrics)
    grouped['REST_DAYS_VALUE'] = grouped['DAYS_REST_BUCKET'].astype(float)

    keep_cols = ['TEAM_ID', 'DAYS_REST_BUCKET', 'REST_RANGE_LABEL', 'REST_DAYS_VALUE']
    keep_cols.extend(rename_metrics.values())

    grouped = grouped[keep_cols]

    print(
        "OK: Referencia de rendimiento por días de descanso cargada "
        f"({len(grouped)} combinaciones TEAM_ID/bucket). "
        f"Métricas: {sorted(rename_metrics.values())}"
    )

    return grouped


def add_days_rest_from_reference(df: pd.DataFrame, rest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combina el dataframe principal con métricas externas por días de descanso.

    Se hace merge por TEAM_ID + DAYS_REST_BUCKET para evitar depender de GAME_DATE.
    """

    if rest_df is None or rest_df.empty:
        return df

    required = {'TEAM_ID', 'DAYS_REST_BUCKET'}
    if not required.issubset(rest_df.columns):
        print(
            "⚠️ La referencia de days rest no tiene las columnas requeridas "
            f"{sorted(required)}. Columnas reales: {sorted(rest_df.columns)}"
        )
        return df

    if not required.issubset(df.columns):
        print(
            "⚠️ El dataframe base no tiene columnas para merge de days rest. Se omite merge externo."
        )
        return df

    left = df.copy()
    if 'DAYS_REST_BUCKET' in left.columns:
        min_bucket = rest_df['DAYS_REST_BUCKET'].min()
        max_bucket = rest_df['DAYS_REST_BUCKET'].max()
        if pd.notna(min_bucket) and pd.notna(max_bucket):
            left['DAYS_REST_BUCKET'] = (
                pd.to_numeric(left['DAYS_REST_BUCKET'], errors='coerce')
                .round()
                .clip(lower=int(min_bucket), upper=int(max_bucket))
                .astype('Int64')
            )

    before_shape = left.shape
    merged = left.merge(rest_df, how='left', on=['TEAM_ID', 'DAYS_REST_BUCKET'])
    after_shape = merged.shape

    added_cols = [c for c in merged.columns if c not in df.columns]

    print(
        "OK: Referencia externa de Days Rest mergeada por TEAM_ID + bucket "
        f"({before_shape} -> {after_shape}). Columnas añadidas: {added_cols}"
    )

    return merged


# =========================
# Streaks
# =========================
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


# =========================
# Features avanzadas
# =========================
def features_enhanced(df: pd.DataFrame, config: Dict[str, object]) -> pd.DataFrame:
    """
    Genera features avanzadas por equipo y partido:
      - WIN_STREAK, LAST_5_PCT
      - SEASON_W_PCT, SEASON_WINS, SEASON_LOSSES
      - DAYS_REST (diff capado a 6) y DAYS_REST_RANGE ("0..6 Days Rest"), con posible refuerzo por parquet externo
      - PACE (de ROLL10_PACE) y TURNOVER_RATIO (ROLL10_TOV/ROLL10_POSS)
    Si config contiene 'days_rest_path', intentará mergear esa referencia con clave segura (_REST_KEY).
    """
    d = ensure_teamid_and_date(df)

    if 'GAME_DATE' not in d.columns:
        raise ValueError('GAME_DATE es obligatorio para calcular las nuevas features.')
    if 'TEAM_ID' not in d.columns:
        raise ValueError('TEAM_ID es obligatorio para calcular las nuevas features.')

    # WL_NUM (1=W, 0=L)
    if 'WL_NUM' not in d.columns:
        if 'WL' in d.columns:
            d['WL_NUM'] = d['WL'].astype(str).str.strip().str.upper().map({'W': 1, 'L': 0})
        else:
            raise ValueError('Falta WL o WL_NUM.')

    # PACE / TURNOVER_RATIO desde rollings si aplican
    if 'ROLL10_PACE' in d.columns and 'PACE' not in d.columns:
        d['PACE'] = d['ROLL10_PACE']
    if 'ROLL10_TOV' in d.columns and 'ROLL10_POSS' in d.columns and 'TURNOVER_RATIO' not in d.columns:
        ratio = d['ROLL10_TOV'] / d['ROLL10_POSS'].replace({0: np.nan})
        d['TURNOVER_RATIO'] = ratio.replace([np.inf, -np.inf], np.nan)

    # Detecta temporada si existe
    season_col = None
    for cand in ['SEASON_KEY', 'SEASON', 'SEASON_ID', 'SEASON_YEAR']:
        if cand in d.columns:
            season_col = cand
            break

    group_cols = ['TEAM_ID'] + ([season_col] if season_col else [])
    d = d.sort_values(group_cols + ['GAME_DATE'])

    # === Procesado por grupo ===
    def process_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values('GAME_DATE').copy()
        prev_results = g['WL_NUM'].astype(float).shift(1)

        # racha previa (positiva en victorias, negativa en derrotas)
        wins_flag = (prev_results == 1).astype(int).fillna(0)
        g['WIN_STREAK'] = wins_flag * (wins_flag.groupby((wins_flag == 0).cumsum()).cumcount() + 1)

        # % victorias últimos 5 previos
        g['LAST_5_PCT'] = prev_results.rolling(5, min_periods=1).mean().fillna(0.5)

        # acumulados temporada previos
        wins_cum = prev_results.fillna(0).cumsum()
        games_cum = prev_results.expanding().count()
        losses_cum = games_cum - wins_cum
        pct_prev = wins_cum.div(games_cum.where(games_cum > 0, np.nan))
        g['SEASON_W_PCT'] = pct_prev.fillna(0.5)
        g['SEASON_WINS'] = wins_cum
        g['SEASON_LOSSES'] = losses_cum
        g['RECENT_FORM_DELTA'] = g['LAST_5_PCT'] - g['SEASON_W_PCT']

        # descanso base por diferencia de fechas (capado 0..6)
        days_rest = g['GAME_DATE'].diff().dt.days
        g['DAYS_REST'] = days_rest.fillna(DEFAULT_DAYS_REST).clip(lower=0).clip(upper=6).astype(int)
        g['DAYS_REST_BUCKET'] = g['DAYS_REST'].clip(lower=0).astype(int)
        g['BACK_TO_BACK_FLAG'] = (g['DAYS_REST'] == 0).astype(int)

        def _format_range(v: object) -> Optional[str]:
            if pd.isna(v):
                return None
            try:
                value = int(v)
            except (TypeError, ValueError):
                return None
            if value == 1:
                return '1 Day Rest'
            return f'{value} Days Rest'

        g['DAYS_REST_RANGE'] = g['DAYS_REST'].map(_format_range)
        return g

    d = d.groupby(group_cols, group_keys=False).apply(process_group).reset_index(drop=True)

    # === Refuerzo con parquet de descanso (opcional) usando MERGE SEGURO ===
    days_path = (config or {}).get('days_rest_path') if isinstance(config, dict) else None
    if days_path:
        ref = load_days_rest_reference(days_path)
        if not ref.empty:
            d = add_days_rest_from_reference(d, ref)
            if 'REST_BUCKET_WIN_PCT' in d.columns:
                d['REST_BUCKET_WIN_PCT'] = pd.to_numeric(
                    d['REST_BUCKET_WIN_PCT'], errors='coerce'
                )
                d['REST_BUCKET_WIN_PCT_DELTA'] = d['REST_BUCKET_WIN_PCT'] - d['SEASON_W_PCT']
            if 'REST_BUCKET_POINTS' in d.columns:
                d['REST_BUCKET_POINTS'] = pd.to_numeric(
                    d['REST_BUCKET_POINTS'], errors='coerce'
                )
            if 'REST_BUCKET_PLUS_MINUS' in d.columns:
                d['REST_BUCKET_PLUS_MINUS'] = pd.to_numeric(
                    d['REST_BUCKET_PLUS_MINUS'], errors='coerce'
                )
        else:
            print("⚠️ No se aplicó referencia externa de Days Rest (archivo vacío o inválido).")

    if isinstance(config, dict):
        feature_list = config.setdefault('new_features', [])
        if isinstance(feature_list, list):
            desired = [
                'DAYS_REST',
                'LAST_5_PCT',
                'BACK_TO_BACK_FLAG',
            ]
            for feat in desired:
                if feat in d.columns and feat not in feature_list:
                    feature_list.append(feat)

    return d


# =========================
# Venue splits (home/road)
# =========================
def features_venue(
    df: pd.DataFrame,
    venue_path: str = "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00c_final/2024-25/dashboards/team_dashboard_by_general_splits__dataset_1.parquet",
) -> pd.DataFrame:
    """Merges home/road split information into the dataframe."""
    df = ensure_teamid_and_date(df)

    # Clave segura tipo string para merge
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

    v['TEAM_ID'] = pd.to_numeric(v['TEAM_ID'].map(_flatten_scalar), errors='coerce').astype('Int64')
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
        'Home': 'Home', 'Local': 'Home', 'Casa': 'Home',
        'Road': 'Road', 'Away': 'Road', 'Visitor': 'Road', 'Visita': 'Road', 'Fuera': 'Road',
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


# =========================
# Match-level dataset
# =========================
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
        home, away,
        left_on='HOME_GAME_ID', right_on='AWAY_GAME_ID',
        how='inner',
    )

    merged['y'] = (merged['HOME_WL'].astype(str).str.upper().str.strip() == 'W').astype(int)

    bases: List[str] = []
    extra_meta: Dict[str, pd.Series] = {}
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
        X_rel['DIFF_VENUE_W_PCT'] = merged['HOME_VENUE_HOME_W_PCT'] - merged['AWAY_VENUE_ROAD_W_PCT']

    if advanced_features is None:
        advanced_features = []

    for feat in advanced_features:
        h_feat = f'HOME_{feat}'
        a_feat = f'AWAY_{feat}'
        if h_feat in merged.columns and a_feat in merged.columns:
            X_rel[f'DIFF_{feat}'] = merged[h_feat] - merged[a_feat]

    if {'HOME_DAYS_REST', 'AWAY_DAYS_REST'}.issubset(merged.columns):
        home_rest = pd.to_numeric(merged['HOME_DAYS_REST'], errors='coerce').fillna(DEFAULT_DAYS_REST)
        away_rest = pd.to_numeric(merged['AWAY_DAYS_REST'], errors='coerce').fillna(DEFAULT_DAYS_REST)
        X_rel['DIFF_DAYS_REST'] = home_rest - away_rest

        home_b2b = (home_rest == 0).astype(int)
        away_b2b = (away_rest == 0).astype(int)
        X_rel['DIFF_B2B_FLAG'] = home_b2b - away_b2b

        extra_meta['HOME_B2B_FLAG'] = home_b2b
        extra_meta['AWAY_B2B_FLAG'] = away_b2b

    if {'HOME_LAST_5_PCT', 'AWAY_LAST_5_PCT'}.issubset(merged.columns):
        home_last5 = pd.to_numeric(merged['HOME_LAST_5_PCT'], errors='coerce')
        away_last5 = pd.to_numeric(merged['AWAY_LAST_5_PCT'], errors='coerce')
        X_rel['DIFF_LAST_5_PCT'] = home_last5.fillna(0.5) - away_last5.fillna(0.5)

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
    for col_name, series in extra_meta.items():
        meta[col_name] = series.values

    allowed_exact = {'HOME_COURT', 'DIFF_DAYS_REST', 'DIFF_LAST_5_PCT', 'DIFF_B2B_FLAG'}
    allowed_prefixes = list(numeric_feature_prefixes) + ['HOME_COURT', 'DIFF_']

    for col in X_rel.columns:
        if col in allowed_exact:
            continue
        if any(col.startswith(pref) for pref in allowed_prefixes):
            continue
        raise AssertionError(f'Se detectó una columna no permitida: {col}. Revisa los prefijos aceptados.')

    return X_rel, y, meta


# =========================
# Train & Eval
# =========================
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

    strategy = 'youden'
    if isinstance(model_name, str) and 'enhanced' in model_name.lower():
        strategy = 'min_error'

    if strategy == 'min_error':
        threshold_candidates = np.linspace(0.0, 1.0, 2001)
        best_thresh = 0.5
        best_error = np.inf
        best_bal_acc = -np.inf
        best_balance_gap = np.inf

        for thr in threshold_candidates:
            preds_val = (proba_val >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, preds_val, labels=[0, 1]).ravel()

            error = fp + fn
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            bal_acc = 0.5 * (tpr + tnr)
            balance_gap = abs(tp - tn)

            if (
                error < best_error
                or (np.isclose(error, best_error) and bal_acc > best_bal_acc + 1e-6)
                or (
                    np.isclose(error, best_error)
                    and np.isclose(bal_acc, best_bal_acc)
                    and balance_gap < best_balance_gap - 1e-6
                )
                or (
                    np.isclose(error, best_error)
                    and np.isclose(bal_acc, best_bal_acc)
                    and np.isclose(balance_gap, best_balance_gap)
                    and abs(thr - 0.5) < abs(best_thresh - 0.5)
                )
            ):
                best_error = error
                best_bal_acc = bal_acc
                best_balance_gap = balance_gap
                best_thresh = float(thr)
    else:
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
