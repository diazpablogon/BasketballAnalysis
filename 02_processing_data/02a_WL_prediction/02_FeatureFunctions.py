"""Utility functions to compute basketball features and train win/loss estimators.
Versión con MERGE SEGURO para Days Rest usando clave string '_REST_KEY' = TEAM_ID|YYYY-MM-DD
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

# Métricas externas por días de descanso que se consideran útiles para el modelo.
# Formato: columna_origen -> (función_agg, nombre_feature_destino)
REST_METRIC_RULES: Dict[str, Tuple[str, str]] = {
    'GP': ('sum', 'REST_BUCKET_GP'),
    'W_PCT': ('mean', 'REST_BUCKET_WIN_PCT'),
    'PLUS_MINUS': ('mean', 'REST_BUCKET_PLUS_MINUS'),
    'PTS': ('mean', 'REST_BUCKET_POINTS'),
}

logger = logging.getLogger(__name__)

__all__ = [
    'DEFAULT_DAYS_REST',
    'ensure_teamid_and_date',
    'features_baseline',
    'features_roll10',
    'features_venue',
    'features_enhanced',
    'features_lineup',
    'features_elo',
    'add_calendar_no_leak',
    'add_elo_no_leak',
    'build_match_level',
    'impute_roll10_inplace',
    'impute_advanced_inplace',
    'impute_calendar_inplace',
    'impute_match_differentials_inplace',
    'parse_days_rest_value',
    'load_days_rest_reference',
    'add_days_rest_from_reference',
    'calculate_current_streak',
    'build_match_dataset_enhanced',
    'fit_and_eval',
    'build_lineup_id',
    'extract_game_lineups_from_boxscore',
    'compute_player_quality_from_onoff',
    'compute_lineup_synergy_from_dashboard',
    'score_game_lineups',
    'build_lineup_scores_for_games',
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
# Baseline (ROLL10) block
# =========================
def features_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Genera métricas ROLL10_* a partir del boxscore crudo y conserva identificadores básicos."""

    df = ensure_teamid_and_date(df)

    required_ids = [
        'TEAM_ID',
        'TEAM_ABBREVIATION',
        'GAME_ID',
        'GAME_DATE',
        'MATCHUP',
    ]

    missing_ids = [col for col in required_ids if col not in df.columns]
    if missing_ids:
        raise ValueError(f"Faltan columnas obligatorias para baseline: {missing_ids}")

    if 'WL_NUM' not in df.columns:
        if 'WL' in df.columns:
            df['WL_NUM'] = (
                df['WL']
                .astype(str)
                .str.strip()
                .str.upper()
                .map({'W': 1, 'L': 0})
            )
        else:
            raise ValueError("Faltan 'WL' y 'WL_NUM' para construir features baseline.")

    df['WL_NUM'] = pd.to_numeric(df['WL_NUM'], errors='coerce')

    numeric_candidates = [
        'FG_PCT',
        'FG3_PCT',
        'FT_PCT',
        'REB',
        'AST',
        'TOV',
        'PTS',
        'PLUS_MINUS',
        'FGM',
        'FG3M',
        'FGA',
        'FTA',
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        num = pd.to_numeric(numerator, errors='coerce')
        denom = pd.to_numeric(denominator, errors='coerce').replace({0: np.nan})
        result = num / denom
        return result.replace([np.inf, -np.inf], np.nan)

    metric_sources: Dict[str, str] = {}
    temp_columns: List[str] = []

    base_metrics = [
        'FG_PCT',
        'FG3_PCT',
        'FT_PCT',
        'REB',
        'AST',
        'TOV',
        'PTS',
        'PLUS_MINUS',
    ]

    for col in base_metrics:
        if col in df.columns:
            metric_sources[col] = col

    if {'FGM', 'FG3M', 'FGA'}.issubset(df.columns):
        temp_col = '__BASELINE_EFG_PCT'
        df[temp_col] = _safe_ratio(df['FGM'] + 0.5 * df['FG3M'], df['FGA'])
        metric_sources['EFG_PCT'] = temp_col
        temp_columns.append(temp_col)

    if {'PTS', 'FGA', 'FTA'}.issubset(df.columns):
        temp_col = '__BASELINE_TS_PCT'
        denominator = 2 * (df['FGA'] + 0.44 * df['FTA'])
        df[temp_col] = _safe_ratio(df['PTS'], denominator)
        metric_sources['TS_PCT'] = temp_col
        temp_columns.append(temp_col)

    if {'AST', 'TOV'}.issubset(df.columns):
        temp_col = '__BASELINE_AST_TOV'
        df[temp_col] = _safe_ratio(df['AST'], df['TOV'])
        metric_sources['AST_TOV'] = temp_col
        temp_columns.append(temp_col)

    df = df.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)

    roll_columns: List[str] = []
    if 'TEAM_ID' in df.columns:
        grouped = df.groupby('TEAM_ID', group_keys=False)
        for metric_name, source_col in metric_sources.items():
            roll_col = f'ROLL10_{metric_name}'

            def _compute(series: pd.Series) -> pd.Series:
                shifted = series.shift(1)
                return shifted.rolling(10, min_periods=1).mean()

            df[roll_col] = grouped[source_col].transform(_compute)
            roll_columns.append(roll_col)

    for temp_col in temp_columns:
        df = df.drop(columns=temp_col, errors='ignore')

    if not roll_columns:
        print('⚠️ BASELINE: No se pudieron generar columnas ROLL10_ a partir del boxscore disponible.')
    else:
        impute_roll10_inplace(df, roll_columns)
        print(f"BASELINE: Columnas ROLL10 generadas: {len(roll_columns)}")
        if df['GAME_DATE'].notna().any():
            min_date = df['GAME_DATE'].min()
            max_date = df['GAME_DATE'].max()
            print(
                "BASELINE: Rango de fechas procesado: "
                f"{min_date.date()} → {max_date.date()}"
            )

    id_columns = [
        'TEAM_ID',
        'TEAM_ABBREVIATION',
        'GAME_ID',
        'GAME_DATE',
        'MATCHUP',
        'WL_NUM',
    ]

    keep_cols = id_columns + sorted(roll_columns)
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df.loc[:, keep_cols].copy()

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
# Lineup (static team block)
# =========================
def features_lineup(
    df: pd.DataFrame,
    lineup_path: str | Path = "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00c_final/2024-25/dashboards/team_dash_lineups__dataset_1.parquet",
) -> pd.DataFrame:
    """Añade métricas estáticas de lineups por equipo si existe el parquet indicado."""

    if 'TEAM_ID' not in df.columns:
        raise ValueError(
            "features_lineup requiere TEAM_ID; revisa features_enhanced para no eliminar columnas de agrupación."
        )

    df = ensure_teamid_and_date(df)

    lineup_path = Path(lineup_path)
    if not lineup_path.exists():
        print(f"⚠️ Archivo de lineups no encontrado ({lineup_path}). Se omite merge.")
        return df

    try:
        lineups = pd.read_parquet(lineup_path).copy()
    except Exception as exc:  # pragma: no cover - robustez en entorno productivo
        print(f"⚠️ No se pudo leer el parquet de lineups: {exc}. Se omite merge.")
        return df

    if lineups.empty:
        print("⚠️ Parquet de lineups vacío. Se omite merge.")
        return df

    lineups.columns = [str(c).strip() for c in lineups.columns]

    team_col = _find_column_case_insensitive(lineups.columns, 'TEAM_ID', 'TEAMID')
    if team_col is None:
        print("⚠️ No se encontró columna TEAM_ID en el parquet de lineups. Se omite merge.")
        return df

    lineups['_TEAM_ID_TMP'] = pd.to_numeric(
        lineups[team_col].map(_flatten_scalar), errors='coerce'
    )
    lineups = lineups[lineups['_TEAM_ID_TMP'].notna()].copy()
    if lineups.empty:
        print("⚠️ No hay registros de lineups válidos tras limpiar TEAM_ID. Se omite merge.")
        return df

    expected_columns: Dict[str, Tuple[str, ...]] = {
        'LINEUP_WEIGHTED_NET': ('LINEUP_WEIGHTED_NET', 'WEIGHTED_NET_RATING'),
        'LINEUP_TOP5_NET': ('LINEUP_TOP5_NET', 'TOP5_NET_RATING'),
        'LINEUP_TOP5_MIN': ('LINEUP_TOP5_MIN', 'TOP5_MINUTES'),
        'LINEUP_STABILITY_HHI': ('LINEUP_STABILITY_HHI', 'LINEUP_HHI'),
        'LINEUP_VARIETY_5MIN': ('LINEUP_VARIETY_5MIN', 'LINEUP_VARIETY_5MIN'),
        'LINEUP_TOTAL_MIN': ('LINEUP_TOTAL_MIN', 'TOTAL_MINUTES'),
    }

    extracted = {'TEAM_ID': lineups['_TEAM_ID_TMP'].astype('int64')}
    for target, aliases in expected_columns.items():
        col_name = _find_column_case_insensitive(lineups.columns, *aliases)
        if col_name is None:
            extracted[target] = pd.Series(np.nan, index=lineups.index, dtype=float)
            continue
        extracted[target] = pd.to_numeric(lineups[col_name], errors='coerce')

    features_df = pd.DataFrame(extracted)
    aggregated = (
        features_df.groupby('TEAM_ID', as_index=False)
        .median(numeric_only=True)
        .rename(columns={'TEAM_ID': '_TEAM_ID_AGG'})
    )
    aggregated['_TEAM_KEY'] = _to_key_str(aggregated['_TEAM_ID_AGG'])

    df['_TEAM_KEY'] = _to_key_str(df['TEAM_ID'])
    before_shape = df.shape

    merge_cols = ['_TEAM_KEY'] + [c for c in expected_columns if c in aggregated.columns]
    df = df.merge(aggregated[merge_cols], on='_TEAM_KEY', how='left')

    df = df.drop(columns=['_TEAM_KEY'], errors='ignore')

    added_cols = [c for c in expected_columns if c in df.columns]
    impute_map: Dict[str, float] = {}
    for col in added_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().all():
            df[col] = 0.0
            impute_map[col] = 0.0
            continue
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(float(median_val))
        impute_map[col] = float(median_val)

    print(
        "OK LINEUP: métricas estáticas añadidas"
        f" ({before_shape} -> {df.shape}). Columnas: {added_cols}"
    )
    if impute_map:
        print("Imputaciones LINEUP aplicadas:", impute_map)

    return df


# =========================
# Imputaciones auxiliares
# =========================
def impute_roll10_inplace(df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Imputa columnas ROLL10_ con la mediana (0.0 si no existe)."""

    if columns is None:
        columns = [c for c in df.columns if c.startswith('ROLL10_')]

    for col in columns:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().all():
            df[col] = 0.0
            continue
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(float(median_val))

    return df


def impute_advanced_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa columnas avanzadas calculadas en features_enhanced."""

    defaults = {
        'WIN_STREAK': 0.0,
        'LAST_5_PCT': 0.5,
        'DAYS_REST': DEFAULT_DAYS_REST,
        'SEASON_W_PCT': 0.5,
        'TURNOVER_RATIO': 0.0,
    }

    if 'PACE' in df.columns:
        df['PACE'] = pd.to_numeric(df['PACE'], errors='coerce')
        pace_median = df['PACE'].median()
        if pd.isna(pace_median):
            pace_median = 0.0
        df['PACE'] = df['PACE'].fillna(float(pace_median))

    for col, default_val in defaults.items():
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(float(default_val))

    return df


def impute_calendar_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa métricas de calendario para evitar fugas."""

    calendar_defaults = {
        'DAYS_REST_CALC': DEFAULT_DAYS_REST,
        'IS_B2B_CALC': 0.0,
        'GAMES_IN_3D_CALC': 0.0,
        'GAMES_IN_5D_CALC': 0.0,
        'GAMES_IN_7D_CALC': 0.0,
    }

    for col, default_val in calendar_defaults.items():
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(float(default_val))

    return df


def impute_match_differentials_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa diferenciales de partido y asegura HOME_COURT y P_ELO_HOME válidos."""

    diff_cols = [c for c in df.columns if c.startswith('DIFF_')]
    for col in diff_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(0.0)

    if 'P_ELO_HOME' in df.columns:
        df['P_ELO_HOME'] = pd.to_numeric(df['P_ELO_HOME'], errors='coerce')
        df['P_ELO_HOME'] = df['P_ELO_HOME'].clip(lower=0.0, upper=1.0)
        df['P_ELO_HOME'] = df['P_ELO_HOME'].fillna(0.5)

    if 'HOME_COURT' in df.columns:
        df['HOME_COURT'] = 1

    if 'LINEUP_SCORE_DIFF' in df.columns:
        df['LINEUP_SCORE_DIFF'] = pd.to_numeric(df['LINEUP_SCORE_DIFF'], errors='coerce')

    return df


# =========================
# Lineup score utilities
# =========================
def build_lineup_id(players: list[int]) -> str:
    """Devuelve '-p1-p2-p3-p4-p5-' con los PLAYER_ID ordenados asc."""

    if players is None:
        players = []

    normalized: List[int] = []
    for p in players:
        if pd.isna(p):
            continue
        normalized.append(int(p))

    if not normalized:
        return '-'

    normalized = sorted(set(normalized))
    return '-' + '-'.join(str(p) for p in normalized) + '-'


def _cast_ids_for_lineup(df: pd.DataFrame, *, require_player: bool = False) -> pd.DataFrame:
    d = df.copy()

    if 'game_id' in d.columns:
        d = d.drop(columns=['game_id'])

    if 'GAME_ID' in d.columns:
        d['GAME_ID'] = d['GAME_ID'].astype('string')

    if 'TEAM_ID' in d.columns:
        d['TEAM_ID'] = pd.to_numeric(d['TEAM_ID'], errors='coerce')
        d = d[d['TEAM_ID'].notna()].copy()
        d['TEAM_ID'] = d['TEAM_ID'].astype('int64')

    if require_player and 'PLAYER_ID' in d.columns:
        d['PLAYER_ID'] = pd.to_numeric(d['PLAYER_ID'], errors='coerce')
        d = d[d['PLAYER_ID'].notna()].copy()
        d['PLAYER_ID'] = d['PLAYER_ID'].astype('int64')

    if require_player and 'PLAYER_ID' not in d.columns:
        raise ValueError('Se requiere PLAYER_ID para esta operación de lineups.')

    return d


def _coerce_minutes_series(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    """Devuelve una serie de minutos como float a partir de columnas conocidas."""

    for cand in candidates:
        if cand in df.columns:
            series = df[cand]
            break
    else:
        return pd.Series(0.0, index=df.index, dtype=float)

    def _parse_minutes(value) -> float:
        if pd.isna(value):
            return 0.0
        if isinstance(value, (int, float, np.number)):
            return float(value)
        text = str(value).strip()
        if not text:
            return 0.0
        if text.count(':') == 1:
            minutes, seconds = text.split(':')
            try:
                return float(minutes) + float(seconds) / 60.0
            except ValueError:
                pass
        numeric_val = pd.to_numeric(text, errors='coerce')
        if pd.isna(numeric_val):
            return 0.0
        return float(numeric_val)

    return series.apply(_parse_minutes).astype(float)


def _winsorize(values: Sequence[float], limits: Tuple[float, float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr

    lower_q, upper_q = limits
    lower_q = max(0.0, min(0.5, lower_q))
    upper_q = max(0.5, min(1.0, upper_q))

    if arr.size == 1:
        lower_val = upper_val = arr[0]
    else:
        lower_val = np.nanquantile(arr, lower_q)
        upper_val = np.nanquantile(arr, upper_q)

    return np.clip(arr, lower_val, upper_val)


def extract_game_lineups_from_boxscore(df_box: pd.DataFrame) -> pd.DataFrame:
    """Devuelve alineaciones por partido usando titulares o top-5 por minutos."""

    required_cols = {'GAME_ID', 'TEAM_ID', 'PLAYER_ID'}
    if not required_cols.issubset(df_box.columns):
        raise ValueError(f'Faltan columnas necesarias en boxscore: {required_cols - set(df_box.columns)}')

    d = _cast_ids_for_lineup(df_box, require_player=True)
    d['_MINUTES_'] = _coerce_minutes_series(d, ('MIN', 'MIN__sum', 'MINUTES', 'MIN_SUM'))
    if 'PTS' not in d.columns:
        d['PTS'] = 0.0
    d['PTS'] = pd.to_numeric(d['PTS'], errors='coerce').fillna(0.0)
    if 'USG_PCT' not in d.columns:
        d['USG_PCT'] = 0.0
    d['USG_PCT'] = pd.to_numeric(d['USG_PCT'], errors='coerce').fillna(0.0)

    records: List[Dict[str, object]] = []

    valid_starter_positions = {'C', 'F', 'G', 'F-C', 'F-G', 'G-F', 'C-F', 'C-G'}

    for (game_id, team_id), group in d.groupby(['GAME_ID', 'TEAM_ID'], sort=False):
        lineup_source = 'starters'
        starters = group[group['START_POSITION'].astype(str).str.strip().isin(valid_starter_positions)]
        starters_ids = list(dict.fromkeys(starters['PLAYER_ID'].tolist()))

        if len(starters_ids) != 5:
            lineup_source = 'top5_min'
            print(
                f"⚠️ Titulares incompletos ({len(starters_ids)}) para GAME_ID={game_id}, TEAM_ID={team_id}. "
                "Uso top-5 por minutos."
            )
            ordered = group.sort_values(
                by=['_MINUTES_', 'PTS', 'USG_PCT'], ascending=[False, False, False]
            )
            starters_ids = list(dict.fromkeys(ordered['PLAYER_ID'].tolist()))[:5]

        if len(starters_ids) < 5:
            print(
                f"⚠️ Solo {len(starters_ids)} jugadores disponibles para GAME_ID={game_id}, TEAM_ID={team_id}."
            )

        players_sorted = sorted({int(pid) for pid in starters_ids})
        lineup_id = build_lineup_id(players_sorted)
        records.append(
            {
                'GAME_ID': game_id,
                'TEAM_ID': team_id,
                'players': players_sorted,
                'lineup_id': lineup_id,
                'lineup_source': lineup_source,
            }
        )

    return pd.DataFrame.from_records(records)


def compute_player_quality_from_onoff(
    df_on: pd.DataFrame,
    df_off: pd.DataFrame,
    m0_player_minutes: int = 400,
    winsor_limits: Tuple[float, float] = (0.05, 0.95),
) -> pd.DataFrame:
    """Calcula impacto on/off con shrinkage por minutos."""

    if m0_player_minutes <= 0:
        raise ValueError('m0_player_minutes debe ser positivo.')

    on = _cast_ids_for_lineup(df_on, require_player=True)
    off = _cast_ids_for_lineup(df_off, require_player=True)

    on['minutes_on'] = _coerce_minutes_series(on, ('MIN', 'MIN__sum', 'MINUTES', 'MIN_SUM'))
    net_on_col = _find_column_case_insensitive(on.columns, 'NET_RATING')
    net_off_col = _find_column_case_insensitive(off.columns, 'NET_RATING')
    if net_on_col is None or net_off_col is None:
        raise ValueError('No se encontraron columnas NET_RATING en on/off.')

    on['NET_RATING_on'] = pd.to_numeric(on[net_on_col], errors='coerce').fillna(0.0)
    off['NET_RATING_off'] = pd.to_numeric(off[net_off_col], errors='coerce').fillna(0.0)

    merged = on[['TEAM_ID', 'PLAYER_ID', 'NET_RATING_on', 'minutes_on']].merge(
        off[['TEAM_ID', 'PLAYER_ID', 'NET_RATING_off']], how='left', on=['TEAM_ID', 'PLAYER_ID']
    )
    merged['NET_RATING_off'] = merged['NET_RATING_off'].fillna(0.0)
    merged['impact_raw'] = merged['NET_RATING_on'] - merged['NET_RATING_off']

    team_means = (
        merged.groupby('TEAM_ID')['impact_raw'].mean().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    merged['team_mean'] = merged['TEAM_ID'].map(team_means).fillna(0.0)

    merged['minutes_on'] = merged['minutes_on'].fillna(0.0)
    merged['weight'] = merged['minutes_on'] / (merged['minutes_on'] + float(m0_player_minutes))
    merged['quality_player'] = (
        merged['weight'] * merged['impact_raw'] + (1.0 - merged['weight']) * merged['team_mean']
    )

    low_weight = merged[merged['weight'] < 0.25]
    if not low_weight.empty:
        sample_players = low_weight.head(5)['PLAYER_ID'].tolist()
        print(
            f"⚠️ Shrink fuerte para {len(low_weight)} jugadores (w<0.25). Ejemplos: {sample_players}."
        )

    result = merged[['TEAM_ID', 'PLAYER_ID', 'impact_raw', 'minutes_on', 'quality_player']].copy()
    result.attrs['winsor_limits'] = winsor_limits
    result.attrs['m0_player_minutes'] = m0_player_minutes

    return result


def compute_lineup_synergy_from_dashboard(
    df_lineups: pd.DataFrame,
    M0_lineup_minutes: int = 300,
) -> pd.DataFrame:
    """Calcula la sinergia bayesiana de las alineaciones históricas usando PLUS_MINUS."""

    if M0_lineup_minutes <= 0:
        raise ValueError('M0_lineup_minutes debe ser positivo.')

    d = _cast_ids_for_lineup(df_lineups)
    if 'GROUP_ID' not in d.columns:
        raise ValueError('El dataframe de lineups debe contener GROUP_ID.')

    d['GROUP_ID'] = d['GROUP_ID'].astype('string')
    d['minutes_lineup'] = _coerce_minutes_series(d, ('MIN', 'MIN__sum', 'MINUTES', 'MIN_SUM'))

    plus_minus_col = 'PLUS_MINUS'
    if plus_minus_col not in d.columns:
        raise ValueError(f'No se encontró columna {plus_minus_col} en el dashboard de lineups.')

    d['plus_minus_lineup'] = pd.to_numeric(d[plus_minus_col], errors='coerce').fillna(0.0)

    team_minutes = d.groupby('TEAM_ID')['minutes_lineup'].sum()
    team_weighted_plus_minus = d.groupby('TEAM_ID').apply(
        lambda x: np.average(
            x['plus_minus_lineup'],
            weights=np.clip(x['minutes_lineup'], 1e-6, None),
        )
        if (x['minutes_lineup'] > 0).any()
        else 0.0
    )
    plus_minus_team = team_weighted_plus_minus.reindex(team_minutes.index).fillna(0.0)

    d['plus_minus_team'] = d['TEAM_ID'].map(plus_minus_team).fillna(0.0)
    d['weight'] = d['minutes_lineup'] / (d['minutes_lineup'] + float(M0_lineup_minutes))
    d['synergy'] = d['weight'] * d['plus_minus_lineup'] + (1.0 - d['weight']) * d['plus_minus_team']

    result = d[
        ['TEAM_ID', 'GROUP_ID', 'synergy', 'minutes_lineup', 'plus_minus_lineup', 'plus_minus_team']
    ].copy()
    result.attrs['M0_lineup_minutes'] = M0_lineup_minutes

    return result


def score_game_lineups(
    df_game_lineups: pd.DataFrame,
    df_quality_players: pd.DataFrame,
    df_synergy: pd.DataFrame,
    alpha_quality: float = 0.75,
) -> pd.DataFrame:
    """Une quality + synergy para puntuar alineaciones por partido."""

    if not 0 <= alpha_quality <= 1:
        raise ValueError('alpha_quality debe estar entre 0 y 1.')

    game_lineups = _cast_ids_for_lineup(df_game_lineups)
    quality = _cast_ids_for_lineup(df_quality_players, require_player=True)
    synergy = _cast_ids_for_lineup(df_synergy)
    if 'GROUP_ID' in synergy.columns:
        synergy['GROUP_ID'] = synergy['GROUP_ID'].astype('string')

    winsor_limits = df_quality_players.attrs.get('winsor_limits', (0.05, 0.95))

    quality_lookup = quality.set_index(['TEAM_ID', 'PLAYER_ID'])['quality_player']
    team_fallback = (
        quality.groupby('TEAM_ID')['quality_player'].mean().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    synergy_lookup = synergy.set_index(['TEAM_ID', 'GROUP_ID'])
    team_synergy = synergy.groupby('TEAM_ID')['plus_minus_team'].mean().fillna(0.0)

    scored_records: List[Dict[str, object]] = []

    for _, row in game_lineups.iterrows():
        team_id = row['TEAM_ID']
        lineup_id = str(row['lineup_id'])
        raw_players = row.get('players', [])
        if isinstance(raw_players, str):
            raw_players = [int(p) for p in re.findall(r'-?\d+', raw_players)]
        elif isinstance(raw_players, (set, tuple, list, np.ndarray, pd.Series)):
            raw_players = list(raw_players)
        elif pd.isna(raw_players):
            raw_players = []
        else:
            raw_players = [raw_players]
        players_list = [int(p) for p in raw_players if not pd.isna(p)]

        player_qualities: List[float] = []
        for pid in players_list:
            key = (team_id, int(pid))
            if key in quality_lookup.index:
                player_qualities.append(float(quality_lookup.loc[key]))
            else:
                fallback = float(team_fallback.get(team_id, 0.0))
                player_qualities.append(fallback)

        if len(players_list) != 5:
            print(
                f"⚠️ Esperaba 5 jugadores para TEAM_ID={team_id}, GAME_ID={row['GAME_ID']}. "
                f"Recibí {len(players_list)}."
            )

        winsorized = _winsorize(player_qualities, winsor_limits)
        lineup_quality = float(np.nanmean(winsorized)) if winsorized.size else 0.0

        minutes_hist = 0.0
        lineup_synergy = float(team_synergy.get(team_id, 0.0))
        plus_minus_lineup = np.nan
        if (team_id, lineup_id) in synergy_lookup.index:
            match_row = synergy_lookup.loc[(team_id, lineup_id)]
            minutes_hist = float(match_row.get('minutes_lineup', 0.0))
            lineup_synergy = float(match_row.get('synergy', lineup_synergy))
            plus_minus_lineup = float(match_row.get('plus_minus_lineup', np.nan))
        else:
            print(
                f"⚠️ Alineación {lineup_id} no encontrada para TEAM_ID={team_id}. "
                "Uso plus_minus_team como fallback."
            )

        lineup_score = alpha_quality * lineup_quality + (1.0 - alpha_quality) * lineup_synergy

        scored_records.append(
            {
                'GAME_ID': row['GAME_ID'],
                'TEAM_ID': team_id,
                'lineup_id': lineup_id,
                'players': players_list,
                'lineup_source': row.get('lineup_source', 'unknown'),
                'lineup_quality': lineup_quality,
                'lineup_synergy': lineup_synergy,
                'lineup_score': lineup_score,
                'minutes_hist': minutes_hist,
                'plus_minus_lineup_hist': plus_minus_lineup,
            }
        )

    return pd.DataFrame.from_records(scored_records)


def build_lineup_scores_for_games(
    df_box: pd.DataFrame,
    df_lineups: pd.DataFrame,
    df_on: pd.DataFrame,
    df_off: pd.DataFrame,
    params: dict | None = None,
) -> pd.DataFrame:
    """Orquesta la construcción de LINEUP_SCORE_DIFF a nivel partido."""

    params = params or {}
    m0_player_minutes = int(params.get('m0_player_minutes', 400))
    M0_lineup_minutes = int(params.get('M0_lineup_minutes', 300))
    alpha_quality = float(params.get('alpha_quality', 0.75))
    winsor_limits = tuple(params.get('winsor_limits', (0.05, 0.95)))

    box = _cast_ids_for_lineup(df_box, require_player=True)

    required_cols = ['GAME_ID', 'TEAM_ID']
    missing_cols = [col for col in required_cols if col not in box.columns]
    if missing_cols:
        raise ValueError(
            f"Se necesitan las columnas {missing_cols} para identificar equipos por juego."
        )

    game_lineups = extract_game_lineups_from_boxscore(box)
    quality_players = compute_player_quality_from_onoff(
        df_on, df_off, m0_player_minutes=m0_player_minutes, winsor_limits=winsor_limits
    )
    quality_players.attrs['winsor_limits'] = winsor_limits
    synergy = compute_lineup_synergy_from_dashboard(df_lineups, M0_lineup_minutes=M0_lineup_minutes)

    scored = score_game_lineups(
        game_lineups,
        quality_players,
        synergy,
        alpha_quality=alpha_quality,
    )

    meta = box[['GAME_ID', 'TEAM_ID']].drop_duplicates()

    team_counts = meta.groupby('GAME_ID')['TEAM_ID'].nunique()
    invalid_games = team_counts[team_counts != 2]
    if not invalid_games.empty:
        sample = invalid_games.index.tolist()[:5]
        raise ValueError(
            'Cada GAME_ID debe tener exactamente 2 TEAM_ID distintos. '
            f'Problemas detectados en: {sample}'
        )

    meta_scored = meta.merge(
        scored,
        on=['GAME_ID', 'TEAM_ID'],
        how='left',
        suffixes=('', '_scored'),
    )

    meta_scored = meta_scored.sort_values(['GAME_ID', 'TEAM_ID']).reset_index(drop=True)
    meta_scored['TEAM_ORDER'] = meta_scored.groupby('GAME_ID').cumcount()

    if meta_scored['TEAM_ORDER'].max() != 1:
        raise ValueError('No se pudieron asignar exactamente dos equipos por GAME_ID.')

    home = (
        meta_scored[meta_scored['TEAM_ORDER'] == 0]
        .drop(columns='TEAM_ORDER')
        .add_prefix('home_')
    )
    away = (
        meta_scored[meta_scored['TEAM_ORDER'] == 1]
        .drop(columns='TEAM_ORDER')
        .add_prefix('away_')
    )

    combined = pd.merge(
        home,
        away,
        left_on='home_GAME_ID',
        right_on='away_GAME_ID',
        how='inner',
        suffixes=('_home', '_away'),
    )

    combined['LINEUP_SCORE_DIFF'] = combined['home_lineup_score'] - combined['away_lineup_score']

    result = combined[
        [
            'home_GAME_ID',
            'home_TEAM_ID',
            'away_TEAM_ID',
            'home_lineup_score',
            'away_lineup_score',
            'LINEUP_SCORE_DIFF',
            'home_lineup_quality',
            'away_lineup_quality',
            'home_lineup_synergy',
            'away_lineup_synergy',
            'home_minutes_hist',
            'away_minutes_hist',
            'home_lineup_id',
            'away_lineup_id',
            'home_players',
            'away_players',
        ]
    ].copy()

    result = result.rename(
        columns={
            'home_GAME_ID': 'GAME_ID',
            'home_TEAM_ID': 'HOME_TEAM_ID',
            'away_TEAM_ID': 'AWAY_TEAM_ID',
        }
    )

    result['GAME_ID'] = result['GAME_ID'].astype('string')
    result['HOME_TEAM_ID'] = pd.to_numeric(result['HOME_TEAM_ID'], errors='coerce').astype('int64')
    result['AWAY_TEAM_ID'] = pd.to_numeric(result['AWAY_TEAM_ID'], errors='coerce').astype('int64')

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
        numerator = pd.to_numeric(d['ROLL10_TOV'], errors='coerce')
        denominator = pd.to_numeric(d['ROLL10_POSS'], errors='coerce')
        ratio = numerator / denominator.replace({0: np.nan})
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

    # --- Asegurar que las columnas de agrupación se conservan ---
    non_group_cols = [c for c in d.columns if c not in group_cols]
    grouped = (
        d.loc[:, group_cols + non_group_cols]
        .groupby(group_cols, as_index=False, group_keys=False, sort=False)
    )

    processed_chunks: List[pd.DataFrame] = []
    for _, chunk in grouped:
        processed = process_group(chunk.copy())
        processed_chunks.append(processed)

    if processed_chunks:
        d = pd.concat(processed_chunks, ignore_index=True)
    else:
        d = d.loc[:, group_cols + non_group_cols].copy()

    # Sanidad: TEAM_ID no debe perderse
    if 'TEAM_ID' not in d.columns:
        raise AssertionError(
            "features_enhanced perdió TEAM_ID; conserva group_cols al aplicar."
        )

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
# Calendario sin fuga
# =========================
def add_calendar_no_leak(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas retrospectivas de calendario evitando fuga de información."""

    df = ensure_teamid_and_date(df)

    if 'TEAM_ID' not in df.columns or 'GAME_DATE' not in df.columns:
        raise ValueError('TEAM_ID y GAME_DATE son necesarios para calcular calendario.')

    calendar_cols = [
        'DAYS_REST_CALC',
        'IS_B2B_CALC',
        'GAMES_IN_3D_CALC',
        'GAMES_IN_5D_CALC',
        'GAMES_IN_7D_CALC',
    ]

    df = df.drop(columns=[c for c in calendar_cols if c in df.columns])

    df = df.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)

    updates: List[pd.DataFrame] = []
    for _, group in df.groupby('TEAM_ID', sort=False):
        g = group.sort_values('GAME_DATE').reset_index()
        dates = g['GAME_DATE'].values.astype('datetime64[D]')
        ordinals = dates.astype('int64')

        diff_days = g['GAME_DATE'].diff().dt.days
        rest_raw = diff_days - 1
        rest_vals = rest_raw.fillna(DEFAULT_DAYS_REST).clip(lower=0)
        b2b_flag = rest_raw.fillna(DEFAULT_DAYS_REST).le(0).astype(int)

        idx = np.arange(len(ordinals))
        counts = {}
        for window in (3, 5, 7):
            lower_bounds = ordinals - window
            start_idx = np.searchsorted(ordinals, lower_bounds, side='left')
            counts[window] = (idx - start_idx).clip(min=0)

        updates.append(
            pd.DataFrame(
                {
                    'index': g['index'],
                    'DAYS_REST_CALC': rest_vals.astype(int),
                    'IS_B2B_CALC': b2b_flag.astype(int),
                    'GAMES_IN_3D_CALC': counts[3],
                    'GAMES_IN_5D_CALC': counts[5],
                    'GAMES_IN_7D_CALC': counts[7],
                }
            )
        )

    if updates:
        merged = pd.concat(updates, ignore_index=True).set_index('index')
        for col in calendar_cols:
            df.loc[merged.index, col] = merged[col].values

    impute_calendar_inplace(df)

    print("OK: Calendario sin fuga calculado (DAYS_REST_CALC, IS_B2B_CALC, ventanas de 3/5/7 días).")

    return df


# =========================
# Elo sin fuga
# =========================
def features_elo(
    df: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """Wrapper canónico para calcular Elo pre-partido sin fuga."""

    return add_elo_no_leak(df, **kwargs)


def add_elo_no_leak(
    df: pd.DataFrame,
    *,
    base_rating: float = 1500.0,
    k_factor: float = 20.0,
    home_advantage: float = 65.0,
    carry_over: float = 0.75,
) -> pd.DataFrame:
    """Calcula Elo pre-partido y probabilidades sin fuga de información."""

    df = ensure_teamid_and_date(df)

    required_cols = {'TEAM_ID', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL_NUM'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas necesarias para Elo: {sorted(missing)}")

    df = df.sort_values(['GAME_DATE', 'GAME_ID', 'TEAM_ID']).reset_index(drop=True)

    season_col = None
    for cand in ['SEASON_KEY', 'SEASON', 'SEASON_ID', 'SEASON_YEAR']:
        if cand in df.columns:
            season_col = cand
            break

    elo_pre = pd.Series(np.nan, index=df.index, dtype=float)
    elo_opp_pre = pd.Series(np.nan, index=df.index, dtype=float)
    elo_diff_pre = pd.Series(np.nan, index=df.index, dtype=float)
    p_elo_home = pd.Series(np.nan, index=df.index, dtype=float)

    current_ratings: Dict[Tuple[int, object], float] = {}
    last_rating_global: Dict[int, float] = {}
    last_season_seen: Dict[int, object] = {}

    df['GAME_ID'] = df['GAME_ID'].astype('string')

    for game_id, game in df.groupby('GAME_ID', sort=False):
        game = game.sort_values('GAME_DATE')
        if len(game) < 2:
            continue

        home_mask = game['MATCHUP'].astype(str).str.contains('vs', case=False, na=False)
        if not home_mask.any():
            continue

        try:
            home_row = game[home_mask].iloc[0]
            away_row = game[~home_mask].iloc[0]
        except IndexError:
            continue

        home_idx = home_row.name
        away_idx = away_row.name

        home_team = int(home_row['TEAM_ID'])
        away_team = int(away_row['TEAM_ID'])

        home_season = home_row[season_col] if season_col else home_row['GAME_DATE'].year
        away_season = away_row[season_col] if season_col else away_row['GAME_DATE'].year

        for team, season in ((home_team, home_season), (away_team, away_season)):
            key = (team, season)
            if key not in current_ratings:
                prev_final = last_rating_global.get(team, base_rating)
                if last_season_seen.get(team) == season:
                    starting = current_ratings.get(key, prev_final)
                else:
                    starting = carry_over * prev_final + (1.0 - carry_over) * base_rating
                current_ratings[key] = float(starting)
                last_season_seen[team] = season

        home_key = (home_team, home_season)
        away_key = (away_team, away_season)

        home_rating = current_ratings[home_key]
        away_rating = current_ratings[away_key]

        expected_home = 1.0 / (
            1.0 + 10 ** ((away_rating - home_rating - home_advantage) / 400.0)
        )
        expected_home = float(np.clip(expected_home, 0.0, 1.0))
        expected_away = float(1.0 - expected_home)

        home_result = float(home_row['WL_NUM'])
        away_result = float(away_row['WL_NUM'])

        new_home = home_rating + k_factor * (home_result - expected_home)
        new_away = away_rating + k_factor * (away_result - expected_away)

        elo_pre.loc[home_idx] = home_rating
        elo_pre.loc[away_idx] = away_rating
        elo_opp_pre.loc[home_idx] = away_rating
        elo_opp_pre.loc[away_idx] = home_rating
        elo_diff_pre.loc[home_idx] = home_rating - away_rating
        elo_diff_pre.loc[away_idx] = away_rating - home_rating

        p_elo_home.loc[home_idx] = expected_home
        p_elo_home.loc[away_idx] = expected_away

        current_ratings[home_key] = float(new_home)
        current_ratings[away_key] = float(new_away)
        last_rating_global[home_team] = float(new_home)
        last_rating_global[away_team] = float(new_away)

    df['ELO_PRE'] = elo_pre
    df['ELO_OPP_PRE'] = elo_opp_pre
    df['ELO_DIFF_PRE'] = elo_diff_pre
    df['P_ELO_HOME'] = p_elo_home.clip(lower=0.0, upper=1.0)

    print("OK: Elo sin fuga calculado (ELO_PRE, ELO_OPP_PRE, ELO_DIFF_PRE, P_ELO_HOME).")

    return df


# =========================
# Match-level builder
# =========================
def build_match_level(
    df: pd.DataFrame,
    lineup_scores: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Construye dataset a nivel partido con diferenciales HOME–AWAY."""

    df = ensure_teamid_and_date(df)

    required_cols = {'GAME_ID', 'MATCHUP', 'TEAM_ID', 'WL_NUM', 'GAME_DATE'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas necesarias para match-level: {sorted(missing)}")

    df['GAME_ID'] = df['GAME_ID'].astype('string')
    df['IS_HOME'] = df['MATCHUP'].astype(str).str.contains('vs', case=False, na=False).astype(int)

    home = df[df['IS_HOME'] == 1].copy()
    away = df[df['IS_HOME'] == 0].copy()

    home_pref = home.add_prefix('HOME_')
    away_pref = away.add_prefix('AWAY_')

    merged = home_pref.merge(
        away_pref,
        left_on='HOME_GAME_ID',
        right_on='AWAY_GAME_ID',
        how='inner',
    )

    if merged.empty:
        raise ValueError('No se pudieron emparejar partidos home/away. Revisa MATCHUP.')

    match_df = pd.DataFrame(
        {
            'GAME_ID': merged['HOME_GAME_ID'],
            'GAME_DATE': merged['HOME_GAME_DATE'],
            'HOME_TEAM_ID': pd.to_numeric(merged['HOME_TEAM_ID'], errors='coerce').astype('Int64'),
            'AWAY_TEAM_ID': pd.to_numeric(merged['AWAY_TEAM_ID'], errors='coerce').astype('Int64'),
            'WL_NUM': pd.to_numeric(merged['HOME_WL_NUM'], errors='coerce'),
            'HOME_COURT': 1,
        }
    )

    def _diff(base_cols: Iterable[str]) -> None:
        for col in base_cols:
            home_col = f'HOME_{col}'
            away_col = f'AWAY_{col}'
            if home_col in merged.columns and away_col in merged.columns:
                match_df[f'DIFF_{col}'] = (
                    pd.to_numeric(merged[home_col], errors='coerce')
                    - pd.to_numeric(merged[away_col], errors='coerce')
                )

    roll_cols = sorted([c for c in df.columns if c.startswith('ROLL10_')])
    venue_cols = sorted([c for c in df.columns if c.startswith('VENUE_')])
    _diff(roll_cols)
    _diff(venue_cols)

    enhanced_cols = [
        'TURNOVER_RATIO',
        'PACE',
        'LAST_5_PCT',
        'DAYS_REST',
        'WIN_STREAK',
        'SEASON_W_PCT',
    ]
    _diff(enhanced_cols)

    if 'HOME_P_ELO_HOME' in merged.columns:
        match_df['P_ELO_HOME'] = pd.to_numeric(merged['HOME_P_ELO_HOME'], errors='coerce')

    elo_pairs = [
        ('ELO_PRE', 'DIFF_ELO_PRE'),
        ('ELO_OPP_PRE', 'DIFF_ELO_OPP_PRE'),
        ('ELO_DIFF_PRE', 'DIFF_ELO_DIFF_PRE'),
    ]
    for base_col, diff_name in elo_pairs:
        home_col = f'HOME_{base_col}'
        away_col = f'AWAY_{base_col}'
        if home_col in merged.columns and away_col in merged.columns:
            match_df[diff_name] = (
                pd.to_numeric(merged[home_col], errors='coerce')
                - pd.to_numeric(merged[away_col], errors='coerce')
            )

    if lineup_scores is not None and not lineup_scores.empty:
        print(
            "ℹ️  lineup_scores proporcionado, pero se ignora en esta versión sin métricas de lineups."
        )

    impute_match_differentials_inplace(match_df)

    allowed_exact = {
        'GAME_ID',
        'GAME_DATE',
        'HOME_TEAM_ID',
        'AWAY_TEAM_ID',
        'WL_NUM',
        'HOME_COURT',
        'P_ELO_HOME',
    }

    for col in match_df.columns:
        if col in allowed_exact:
            continue
        if col.startswith('DIFF_'):
            continue
        raise AssertionError(
            f"Se detectó una columna no permitida en match-level: {col}. Revisa prefijos aceptados."
        )

    match_df['WL_NUM'] = pd.to_numeric(match_df['WL_NUM'], errors='coerce').fillna(0.0)

    return match_df


# =========================
# Match-level dataset
# =========================
def build_match_dataset_enhanced(
    df: pd.DataFrame,
    numeric_feature_prefixes: Sequence[str] = ("ROLL10_", "VENUE_"),
    advanced_features: Optional[Sequence[str]] = None,
    lineup_scores: Optional[pd.DataFrame] = None,
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

    if lineup_scores is not None:
        try:
            lineup_df = lineup_scores.copy()
            if 'GAME_ID' not in lineup_df.columns:
                raise ValueError('lineup_scores necesita columna GAME_ID')
            rename_map = {
                'GAME_ID': 'LINEUP_GAME_ID',
                'home_lineup_score': 'HOME_LINEUP_SCORE',
                'away_lineup_score': 'AWAY_LINEUP_SCORE',
                'home_lineup_quality': 'HOME_LINEUP_QUALITY',
                'away_lineup_quality': 'AWAY_LINEUP_QUALITY',
                'home_lineup_synergy': 'HOME_LINEUP_SYNERGY',
                'away_lineup_synergy': 'AWAY_LINEUP_SYNERGY',
                'home_minutes_hist': 'HOME_LINEUP_MINUTES_HIST',
                'away_minutes_hist': 'AWAY_LINEUP_MINUTES_HIST',
                'home_lineup_id': 'HOME_LINEUP_ID',
                'away_lineup_id': 'AWAY_LINEUP_ID',
                'home_players': 'HOME_PLAYERS',
                'away_players': 'AWAY_PLAYERS',
            }
            lineup_df = lineup_df.rename(columns={k: v for k, v in rename_map.items() if k in lineup_df.columns})
            lineup_df['LINEUP_GAME_ID'] = lineup_df['LINEUP_GAME_ID'].astype('string')
            merged['HOME_GAME_ID'] = merged['HOME_GAME_ID'].astype('string')
            merged = merged.merge(
                lineup_df,
                left_on='HOME_GAME_ID',
                right_on='LINEUP_GAME_ID',
                how='left',
                suffixes=('', '_LINEUP'),
            )
            merged = merged.drop(columns=[c for c in ['LINEUP_GAME_ID'] if c in merged.columns])
            for col in [
                'HOME_LINEUP_SCORE',
                'AWAY_LINEUP_SCORE',
                'HOME_LINEUP_QUALITY',
                'AWAY_LINEUP_QUALITY',
                'HOME_LINEUP_SYNERGY',
                'AWAY_LINEUP_SYNERGY',
                'HOME_LINEUP_MINUTES_HIST',
                'AWAY_LINEUP_MINUTES_HIST',
            ]:
                if col in merged.columns:
                    merged[col] = pd.to_numeric(merged[col], errors='coerce')
        except Exception as exc:
            print(f"Omito lineup score: {exc}")


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

    if 'LINEUP_SCORE_DIFF' in merged.columns:
        X_rel['LINEUP_SCORE_DIFF'] = pd.to_numeric(merged['LINEUP_SCORE_DIFF'], errors='coerce')

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
    for col in [
        'LINEUP_SCORE_DIFF',
        'HOME_LINEUP_SCORE',
        'AWAY_LINEUP_SCORE',
        'HOME_LINEUP_QUALITY',
        'AWAY_LINEUP_QUALITY',
        'HOME_LINEUP_SYNERGY',
        'AWAY_LINEUP_SYNERGY',
        'HOME_LINEUP_ID',
        'AWAY_LINEUP_ID',
        'HOME_PLAYERS',
        'AWAY_PLAYERS',
        'HOME_LINEUP_MINUTES_HIST',
        'AWAY_LINEUP_MINUTES_HIST',
    ]:
        if col in merged.columns:
            meta[col] = merged[col]
    for col_name, series in extra_meta.items():
        meta[col_name] = series.values

    allowed_exact = {'HOME_COURT', 'DIFF_DAYS_REST', 'DIFF_LAST_5_PCT', 'DIFF_B2B_FLAG', 'LINEUP_SCORE_DIFF'}
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
