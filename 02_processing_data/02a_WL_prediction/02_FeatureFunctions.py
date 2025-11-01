"""Utility functions to compute basketball features and train win/loss estimators.
Versión con MERGE SEGURO para Days Rest usando clave string '_REST_KEY' = TEAM_ID|YYYY-MM-DD
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

DEFAULT_DAYS_REST = 5

DEFAULT_LINEUP_CONFIG = {
    'N_ROLL_MIN': 5,
    'TEAM_PCTL': (5, 95),
    'USE_ON_OFF': True,
    'USE_LINEUPS': True,
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
    'estimate_expected_minutes',
    'compute_lineup_metrics_for_row',
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


# =========================
# LINEUP utilities (in-memory, anti-leakage)
# =========================
def prep_dates(df: Optional[pd.DataFrame], date_col: str) -> Optional[pd.DataFrame]:
    """Return a copy of ``df`` with ``date_col`` cast to naive datetime64[ns]."""

    if df is None:
        return None

    d = df.copy()
    if date_col not in d.columns:
        logger.warning("No se encontró la columna %s para normalizar fechas.", date_col)
        return d

    d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
    if ptypes.is_datetime64tz_dtype(d[date_col].dtype):
        d[date_col] = d[date_col].dt.tz_convert(None)
    try:
        d[date_col] = d[date_col].dt.tz_localize(None)
    except (TypeError, AttributeError):
        # La serie ya es naive o no soporta tz_localize
        pass

    return d


def build_player_to_date_in_memory(
    player_box: pd.DataFrame,
    mean_cols: List[str],
    roll_cols: List[str],
    group_keys: Sequence[str] = ('PLAYER_ID',),
    date_col: str = 'GAME_DATE',
) -> pd.DataFrame:
    """Compute expanding/rolling stats for players without leaking current game information."""

    if player_box is None or player_box.empty:
        logger.warning("build_player_to_date_in_memory recibió boxscores vacíos.")
        return pd.DataFrame()

    missing_required = [col for col in ('PLAYER_ID', 'TEAM_ID', 'GAME_ID') if col not in player_box.columns]
    if missing_required:
        raise ValueError(f"Boxscores incompletos, faltan columnas: {missing_required}")

    if date_col not in player_box.columns:
        raise ValueError(f"Boxscores sin columna de fecha '{date_col}'.")

    df = prep_dates(player_box, date_col)

    order_cols: List[str] = list(dict.fromkeys((*group_keys, 'TEAM_ID', date_col, 'GAME_ID')))
    order_cols = [c for c in order_cols if c in df.columns]
    df = df.sort_values(order_cols)

    available_mean = [c for c in mean_cols if c in df.columns]
    missing_mean = sorted(set(mean_cols) - set(available_mean))
    if missing_mean:
        logger.warning("Columnas mean no disponibles para historizar: %s", missing_mean)

    available_roll = [c for c in roll_cols if c in df.columns]
    missing_roll = sorted(set(roll_cols) - set(available_roll))
    if missing_roll:
        logger.warning("Columnas rolling no disponibles para historizar: %s", missing_roll)

    group_cols = list(group_keys)
    n_roll = int(DEFAULT_LINEUP_CONFIG.get('N_ROLL_MIN', 5))

    grouped = df.groupby(group_cols, dropna=False, sort=False)
    pieces: List[pd.DataFrame] = []
    for _, g in grouped:
        g = g.copy()
        for col in available_mean:
            g[f"{col}_to_date"] = g[col].expanding(min_periods=1).mean().shift(1)
        for col in available_roll:
            g[f"{col}_roll{n_roll}_prev"] = (
                g[col].rolling(window=n_roll, min_periods=1).mean().shift(1)
            )
        pieces.append(g)

    result = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=df.columns)
    return result


def compute_on_off_to_date_in_memory(
    on_df: Optional[pd.DataFrame],
    off_df: Optional[pd.DataFrame],
    date_col: str = 'GAME_DATE',
) -> pd.DataFrame:
    """Combine ON/OFF splits to obtain per-player historical net rating differentials."""

    expected_cols = [
        'PLAYER_ID',
        'TEAM_ID',
        'GAME_ID',
        date_col,
        'ON_NET_RATING_to_date',
        'OFF_NET_RATING_to_date',
        'NET_RATING_DIFF_to_date',
    ]

    if on_df is None or on_df.empty or off_df is None or off_df.empty:
        logger.info("Sin datos ON/OFF: se devuelve DataFrame vacío con columnas esperadas.")
        return pd.DataFrame(columns=expected_cols)

    on = prep_dates(on_df, date_col)
    off = prep_dates(off_df, date_col)

    def _rating_column(df: pd.DataFrame) -> str:
        for candidate in ('NET_RATING_to_date', 'NET_RATING'):
            if candidate in df.columns:
                return candidate
        raise ValueError("Falta columna NET_RATING en splits ON/OFF")

    on_rating_col = _rating_column(on)
    off_rating_col = _rating_column(off)

    on_hist = on[['PLAYER_ID', 'TEAM_ID', 'GAME_ID', date_col, on_rating_col]].rename(
        columns={on_rating_col: 'ON_NET_RATING_to_date'}
    )
    off_hist = off[['PLAYER_ID', 'TEAM_ID', 'GAME_ID', date_col, off_rating_col]].rename(
        columns={off_rating_col: 'OFF_NET_RATING_to_date'}
    )

    merged = pd.merge(
        on_hist,
        off_hist,
        on=['PLAYER_ID', 'TEAM_ID', 'GAME_ID', date_col],
        how='outer',
        suffixes=('_on', '_off'),
    )
    merged['NET_RATING_DIFF_to_date'] = merged['ON_NET_RATING_to_date'] - merged['OFF_NET_RATING_to_date']

    return merged[expected_cols]


def estimate_expected_minutes(player_hist: pd.DataFrame, n_roll: int = 5) -> pd.Series:
    """Estimate expected minutes for players using rolling history with sensible fallbacks."""

    if player_hist.empty:
        return pd.Series(dtype=float)

    col_roll = f"MIN_roll{n_roll}_prev"
    if col_roll in player_hist.columns:
        minutes = player_hist[col_roll].astype(float)
    elif 'MIN_to_date' in player_hist.columns:
        minutes = player_hist['MIN_to_date'].astype(float)
    elif 'MIN' in player_hist.columns:
        minutes = player_hist['MIN'].astype(float)
    else:
        minutes = pd.Series(np.nan, index=player_hist.index, dtype=float)
        logger.warning("No se encontraron columnas MIN históricas; se usará fallback global.")

    if minutes.isna().all():
        minutes = pd.Series(15.0, index=player_hist.index, dtype=float)
        return minutes

    team_median = minutes.median(skipna=True)
    if not np.isfinite(team_median):
        team_median = 15.0

    minutes = minutes.fillna(team_median).clip(lower=0)
    if (minutes == 0).all():
        minutes = pd.Series(15.0, index=player_hist.index, dtype=float)
    return minutes


def compute_lineup_metrics_for_row(
    row: pd.Series,
    player_hist: pd.DataFrame,
    onoff_hist: pd.DataFrame,
    lineups_df: Optional[pd.DataFrame],
    config: Dict[str, object],
) -> Dict[str, object]:
    """Compute lineup-centric features for a single team-game row."""

    defaults = {
        'LINEUP_NUM_PLAYERS': 0,
        'LINEUP_EXP_MIN_SUM': 0.0,
        'LINEUP_NET_RATING_AVG': np.nan,
        'LINEUP_IMPACT_AVG': np.nan,
        'LINEUP_TS_PCT_AVG': np.nan,
        'LINEUP_SCORE': 0.5,
        'LINEUP_STARTERS_OUT': 0,
        'LINEUP_B2B_FLAG': int(row.get('B2B_GAMES', 0) or 0),
    }

    team_id = row.get('TEAM_ID')
    game_id = row.get('GAME_ID')
    if pd.isna(team_id) or pd.isna(game_id):
        logger.warning("Fila sin TEAM_ID o GAME_ID, se devuelven valores por defecto.")
        return defaults

    mask = (player_hist['TEAM_ID'] == team_id) & (player_hist['GAME_ID'] == game_id)
    players = player_hist.loc[mask].copy()
    if players.empty:
        logger.warning(
            "Sin boxscores históricos para TEAM_ID=%s GAME_ID=%s; valores por defecto.",
            team_id,
            game_id,
        )
        return defaults

    n_roll = int(config.get('N_ROLL_MIN', DEFAULT_LINEUP_CONFIG['N_ROLL_MIN']))
    exp_minutes = estimate_expected_minutes(players, n_roll=n_roll)
    players['MIN_exp'] = exp_minutes

    def _get_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
        if col in df.columns:
            return df[col].astype(float)
        return pd.Series(default, index=df.index, dtype=float)

    pts = _get_series(players, 'PTS_to_date')
    reb = _get_series(players, 'REB_to_date')
    ast = _get_series(players, 'AST_to_date')
    stl = _get_series(players, 'STL_to_date')
    blk = _get_series(players, 'BLK_to_date')
    tov = _get_series(players, 'TOV_to_date')
    if tov.eq(0).all() and 'TO_to_date' in players.columns:
        tov = players['TO_to_date'].astype(float)

    min_to_date = _get_series(players, 'MIN_to_date', default=1.0).replace(0, 1.0)
    impact = (pts + reb + ast + stl + blk - tov) / np.maximum(min_to_date, 1.0)
    players['IMPACT_to_date'] = impact.replace([np.inf, -np.inf], np.nan)

    if not onoff_hist.empty and config.get('USE_ON_OFF', True):
        merge_keys = ['PLAYER_ID', 'TEAM_ID']
        if 'GAME_ID' in onoff_hist.columns:
            merge_keys.append('GAME_ID')
        players = players.merge(onoff_hist, on=merge_keys, how='left')

    net_rating = players.get('NET_RATING_to_date')
    if net_rating is None or net_rating.isna().all():
        net_rating = players['IMPACT_to_date']
        players['NET_RATING_to_date'] = net_rating

    weights = players['MIN_exp'].astype(float)
    weight_sum = float(weights.sum()) if not weights.empty else 0.0

    def _weighted_average(values: pd.Series) -> float:
        if values is None or values.empty:
            return np.nan
        if weight_sum > 0:
            return float(np.nansum(values * weights) / weight_sum)
        if values.notna().any():
            return float(values.mean())
        return np.nan

    lineup_net = _weighted_average(net_rating)
    lineup_impact = _weighted_average(players['IMPACT_to_date'])
    ts_series = players.get('TS_PCT_to_date')
    lineup_ts = _weighted_average(ts_series) if ts_series is not None else np.nan

    score = lineup_net
    if not np.isfinite(score):
        score = lineup_impact
    if not np.isfinite(score):
        score = 0.0
    lineup_score = float(np.clip((score + 20.0) / 40.0, 0.0, 1.0))

    starters_out = 0
    if (
        lineups_df is not None
        and config.get('USE_LINEUPS', True)
        and not lineups_df.empty
        and {'TEAM_ID', 'GAME_ID'}.issubset(lineups_df.columns)
    ):
        subset = lineups_df[
            (lineups_df['TEAM_ID'] == team_id) & (lineups_df['GAME_ID'] == game_id)
        ]
        if not subset.empty:
            starters: set = set()
            if 'PLAYER_ID' in subset.columns:
                if 'IS_STARTER' in subset.columns:
                    starters = set(
                        subset.loc[subset['IS_STARTER'].astype(bool), 'PLAYER_ID'].dropna().astype(int)
                    )
                elif 'STARTER' in subset.columns:
                    starters = set(
                        subset.loc[subset['STARTER'].astype(bool), 'PLAYER_ID'].dropna().astype(int)
                    )
                else:
                    starters = set(subset['PLAYER_ID'].dropna().astype(int))
            elif 'STARTERS' in subset.columns:
                starters_text = subset['STARTERS'].dropna().astype(str).tolist()
                starters = {
                    int(s.strip())
                    for text in starters_text
                    for s in re.split(r"[,;\s]+", text)
                    if s.strip().isdigit()
                }

            active_players = set(players['PLAYER_ID'].dropna().astype(int))
            starters_out = len(starters - active_players)

    defaults.update(
        {
            'LINEUP_NUM_PLAYERS': int(players['PLAYER_ID'].nunique()),
            'LINEUP_EXP_MIN_SUM': float(weight_sum),
            'LINEUP_NET_RATING_AVG': float(lineup_net) if np.isfinite(lineup_net) else np.nan,
            'LINEUP_IMPACT_AVG': float(lineup_impact) if np.isfinite(lineup_impact) else np.nan,
            'LINEUP_TS_PCT_AVG': float(lineup_ts) if np.isfinite(lineup_ts) else np.nan,
            'LINEUP_SCORE': lineup_score,
            'LINEUP_STARTERS_OUT': int(starters_out) if starters_out else 0,
        }
    )

    return defaults


def add_lineup_features_in_memory(
    df_teamgames: pd.DataFrame,
    df_player_box: pd.DataFrame,
    df_on: Optional[pd.DataFrame],
    df_off: Optional[pd.DataFrame],
    df_lineups: Optional[pd.DataFrame],
    config: Dict[str, object] = DEFAULT_LINEUP_CONFIG,
) -> pd.DataFrame:
    """Augment team game logs with lineup-derived features computed in-memory."""

    if df_teamgames is None or df_teamgames.empty:
        raise ValueError("Se requieren teamgamelogs con información de partidos.")

    if df_player_box is None or df_player_box.empty:
        raise ValueError("Se requieren boxscores de jugadores para construir historiales.")

    if not {'TEAM_ID', 'GAME_ID'}.issubset(df_teamgames.columns):
        raise ValueError("teamgamelogs sin columnas TEAM_ID/GAME_ID, no se pueden generar features.")

    if not {'TEAM_ID', 'GAME_ID', 'PLAYER_ID'}.issubset(df_player_box.columns):
        raise ValueError("boxscores sin columnas clave TEAM_ID/GAME_ID/PLAYER_ID.")

    effective_config = {**DEFAULT_LINEUP_CONFIG, **(config or {})}

    df_team = prep_dates(df_teamgames.copy(), 'GAME_DATE')

    if 'GAME_DATE' not in df_player_box.columns:
        date_map = (
            df_team[['TEAM_ID', 'GAME_ID', 'GAME_DATE']]
            .drop_duplicates()
            .rename(columns={'GAME_DATE': 'GAME_DATE'})
        )
        df_player_box = df_player_box.merge(date_map, on=['TEAM_ID', 'GAME_ID'], how='left')
        missing_dates = df_player_box['GAME_DATE'].isna().sum()
        if missing_dates > 0:
            logger.warning("%s boxscores sin GAME_DATE tras merge; se imputarán como NaT.", missing_dates)
    df_players = prep_dates(df_player_box, 'GAME_DATE')

    df_on = prep_dates(df_on, 'GAME_DATE') if df_on is not None else None
    df_off = prep_dates(df_off, 'GAME_DATE') if df_off is not None else None
    df_lineups = prep_dates(df_lineups, 'GAME_DATE') if df_lineups is not None else None

    mean_cols = [c for c in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN', 'NET_RATING', 'TS_PCT', 'USG_PCT'] if c in df_players.columns]
    roll_cols = [c for c in ['MIN', 'PTS', 'AST', 'REB'] if c in df_players.columns]

    player_hist = build_player_to_date_in_memory(
        df_players,
        mean_cols=mean_cols,
        roll_cols=roll_cols,
        group_keys=('PLAYER_ID',),
        date_col='GAME_DATE',
    )

    onoff_hist = compute_on_off_to_date_in_memory(df_on, df_off, date_col='GAME_DATE')

    results = []
    for _, row in df_team.iterrows():
        metrics = compute_lineup_metrics_for_row(row, player_hist, onoff_hist, df_lineups, effective_config)
        results.append(metrics)

    lineup_df = pd.DataFrame(results, index=df_team.index)
    df_aug = df_team.copy()
    for col in lineup_df.columns:
        df_aug[col] = lineup_df[col]

    lineup_cols = [c for c in df_aug.columns if c.startswith('LINEUP_')]
    lower_pct, upper_pct = effective_config.get('TEAM_PCTL', (5, 95))

    def _normalize_group(series: pd.Series) -> pd.Series:
        valid = series.dropna()
        if valid.empty:
            return series
        lower = np.nanpercentile(valid, lower_pct) if np.isfinite(lower_pct) else valid.min()
        upper = np.nanpercentile(valid, upper_pct) if np.isfinite(upper_pct) else valid.max()
        if not np.isfinite(upper) or not np.isfinite(lower) or upper == lower:
            return series
        scaled = (series - lower) / (upper - lower)
        return scaled.clip(0.0, 1.0)

    for col in lineup_cols:
        norm_col = f"{col}_PCTL"
        df_aug[norm_col] = df_aug.groupby('TEAM_ID')[col].transform(_normalize_group)

    return df_aug
