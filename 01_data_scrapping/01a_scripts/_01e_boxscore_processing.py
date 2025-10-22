#!/usr/bin/env python3
"""Agrega métricas de boxscore agregadas al gamelog de equipos."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

DEFAULT_GAMELOG_PATH = Path(
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00b_intermediate/2024-25/teamgamelogs_by_game.parquet"
)
DEFAULT_BOXSCORES_PATH = Path(
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00c_final/2024-25/boxscores.parquet"
)
DEFAULT_OUTPUT_PATH = Path(
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00c_final/2024-25/teamgamelogs_by_game.parquet"
)

KEY_COLUMNS = ["GAME_ID", "TEAM_ID"]

# Lista de todos los Team IDs de la NBA
NBA_TEAM_IDS = [
    "1610612737", "1610612738", "1610612739", "1610612740", "1610612741",
    "1610612742", "1610612743", "1610612744", "1610612745", "1610612746",
    "1610612747", "1610612748", "1610612749", "1610612750", "1610612751",
    "1610612752", "1610612753", "1610612754", "1610612755", "1610612756",
    "1610612757", "1610612758", "1610612759", "1610612760", "1610612761",
    "1610612762", "1610612763", "1610612764", "1610612765", "1610612766",
    "1610612767", "1610612768"
]

SUM_COLUMNS = [
    "AST",
    "BLK",
    "BLKA",
    "DREB",
    "FG3A",
    "FG3M",
    "FGA",
    "FGM",
    "FTA",
    "FTM",
    "OREB",
    "PF",
    "PFD",
    "PTS",
    "REB",
    "STL",
    "TOV",
]

MEAN_COLUMNS = [
    "AST_PCT",
    "AST_TOV",
    "AST_TO",
    "CFID",
    "CFPARAMS",
    "DEF_RATING",
    "DREB_PCT",
    "EFG_PCT",
    "E_OFF_RATING",
    "E_DEF_RATING",
    "E_NET_RATING",
    "E_PACE",
    "E_PACE",
    "FG3_PCT",
    "FG_PCT",
    "NET_RATING",
    "OFF_RATING",
    "OREB_PCT",
    "PACE",
    "PACE_PER40",
    "PIE",
    "POSS",
    "REB_PCT",
    "TM_TOV_PCT",
    "TS_PCT",
    "USG_PCT",
    "SECOND_CHANCE_PTS",
    "SECOND_CHANCE_PTS_2ND_CHANCE",
    "PTS_2ND_CHANCE",
    "PTS_FB",
    "PTS_PAINT",
    "PTS_OFF_TOV",
    "OPP_PTS_2ND_CHANCE",
    "OPP_PTS_FB",
    "OPP_PTS_PAINT",
    "OPP_PTS_OFF_TOV",
    "OPP_TOV",
    "OPP_AST",
    "OPP_BLK",
    "OPP_STL",
    "OPP_OREB",
    "OPP_DREB",
    "OPP_REB",
    "OPP_FGM",
    "OPP_FGA",
    "OPP_FG_PCT",
    "OPP_FG3M",
    "OPP_FG3A",
    "OPP_FG3_PCT",
    "OPP_FTM",
    "OPP_FTA",
    "OPP_FT_PCT",
    "OPP_PF",
    "PCT_AST_2PM",
    "PCT_AST_3PM",
    "PCT_AST_FGM",
    "PCT_AST_UAST",
    "PCT_BLK_2PA",
    "PCT_BLK_3PA",
    "PCT_BLK_FGA",
    "PCT_DREB",
    "PCT_FGA",
    "PCT_FGA_2PT",
    "PCT_FGA_3PT",
    "PCT_FGM",
    "PCT_FTA",
    "PCT_FTM",
    "PCT_OREB",
    "PCT_PF",
    "PCT_PFD",
    "PCT_PTS",
    "PCT_PTS_2PT",
    "PCT_PTS_2PT_MR",
    "PCT_PTS_3PT",
    "PCT_PTS_FB",
    "PCT_PTS_FT",
    "PCT_PTS_OFF_TOV",
    "PCT_PTS_PAINT",
    "PCT_REB",
    "PCT_STL",
    "PCT_TOV",
    "PCT_UAST_2PM",
    "PCT_UAST_3PM",
    "PCT_UAST_FGM",
    "PIE",
    "PLUS_MINUS",
    "POSS",
    "REB_PCT",
    "TM_TOV_PCT",
    "TS_PCT",
    "USG_PCT",
]

INSERTION_ORDER_GROUPS: List[List[str]] = [
    ["FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT"],
    ["OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PFD", "PTS"],
    ["PLUS_MINUS", "POSS", "PACE", "PACE_PER40", "E_PACE"],
    ["OFF_RATING", "DEF_RATING", "NET_RATING", "E_OFF_RATING", "E_DEF_RATING", "E_NET_RATING", "TS_PCT", "EFG_PCT"],
    ["USG_PCT", "REB_PCT", "OREB_PCT", "DREB_PCT", "TM_TOV_PCT", "PIE"],
    ["SECOND_CHANCE_PTS", "PTS_OFF_TOV", "PTS_FB", "PTS_PAINT"],
    ["OPP_AST", "OPP_STL", "OPP_BLK", "OPP_OREB", "OPP_DREB", "OPP_REB"],
    ["OPP_FGM", "OPP_FGA", "OPP_FG_PCT", "OPP_FG3M", "OPP_FG3A", "OPP_FG3_PCT", "OPP_FTM", "OPP_FTA", "OPP_FT_PCT"],
    ["OPP_PF", "OPP_TOV", "OPP_PTS_OFF_TOV", "OPP_PTS_FB", "OPP_PTS_PAINT", "OPP_PTS_2ND_CHANCE"],
]

TOTALS_PRIORITY: List[str] = [
    "PTS", "REB", "AST", "STL", "BLK", "OREB", "DREB", "TOV",
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "PF", "PFD",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agrega métricas de boxscore (equipo-partido) al gamelog de equipos."
    )
    parser.add_argument(
        "--gamelog",
        type=Path,
        default=DEFAULT_GAMELOG_PATH,
        help="Ruta del parquet intermedio de team gamelog",
    )
    parser.add_argument(
        "--boxscores",
        type=Path,
        default=DEFAULT_BOXSCORES_PATH,
        help="Ruta del parquet de boxscores (jugadores)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Ruta de salida para el gamelog final",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No escribir archivo de salida, solo logs",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Nivel de log (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()

def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")

def ensure_string_keys(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    for key in keys:
        if key in df.columns:
            df[key] = df[key].astype(str)
    return df

def filter_players_by_minutes(df: pd.DataFrame, min_minutes: float = 1.0) -> pd.DataFrame:
    if 'MIN' not in df.columns:
        logging.warning("La columna 'MIN' no existe en el DataFrame. No se filtrará por minutos.")
        return df.copy()

    if df['MIN'].dtype == object:
        def minutes_to_float(min_str):
            if pd.isna(min_str):
                return 0.0
            try:
                parts = str(min_str).split(':')
                if len(parts) == 2:
                    return int(parts[0]) + int(parts[1]) / 60.0
                else:
                    return float(min_str)
            except (ValueError, TypeError):
                return 0.0

        df_minutes = df['MIN'].apply(minutes_to_float)
    else:
        df_minutes = df['MIN'].astype(float)

    original_count = len(df)
    filtered_df = df[df_minutes >= min_minutes].copy()
    filtered_count = len(filtered_df)

    logging.info(
        "Filtrado por minutos: %d jugadores -> %d jugadores (>= %.1f minutos)",
        original_count, filtered_count, min_minutes
    )

    return filtered_df

def drop_ignored_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        "VIDEO_AVAILABLE",
        "COMMENT",
        "NICKNAME",
        "START_POSITION",
        "START_TIME",
        "PLUS_MINUS_RANK",
        "GP_RANK",
        "TEAM_ABBREVIATION",
        "TEAM_NAME",
        "PLAYER_ID",
        "PLAYER_NAME",
        "MIN_RANK",
    ]
    columns_to_drop = [c for c in columns_to_drop if c in df.columns]
    if columns_to_drop:
        logging.debug("Eliminando columnas no relevantes de boxscores: %s", ", ".join(columns_to_drop))
        df = df.drop(columns=columns_to_drop)
    return df

def aggregate_boxscores(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_ignored_columns(df)

    sumable = [c for c in SUM_COLUMNS if c in df.columns]
    meanable = [c for c in MEAN_COLUMNS if c in df.columns]

    agg_dict: Dict[str, str] = {c: "sum" for c in sumable}
    agg_dict.update({c: "mean" for c in meanable})

    grouped = df.groupby(KEY_COLUMNS, as_index=False).agg(agg_dict)

    numeric_columns = [c for c in grouped.columns if c not in KEY_COLUMNS]
    grouped[numeric_columns] = grouped[numeric_columns].round(5)

    return grouped

def drop_colliding_columns(
        aggregated: pd.DataFrame, existing_columns: Sequence[str]
) -> Tuple[pd.DataFrame, List[str]]:
    collisions = [
        column
        for column in aggregated.columns
        if column not in KEY_COLUMNS and column in existing_columns
    ]
    if collisions:
        collisions = sorted(collisions)
        aggregated = aggregated.drop(columns=collisions)
    return aggregated, collisions

def order_new_columns(new_columns: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    new_set = set(new_columns)
    for group in INSERTION_ORDER_GROUPS:
        ordered.extend([column for column in group if column in new_set])
    totals_to_add = [
        column
        for column in TOTALS_PRIORITY
        if column in new_set and column not in ordered
    ]
    ordered.extend(totals_to_add)
    remaining = [c for c in new_columns if c not in ordered]
    if remaining:
        logging.debug(
            "Columnas nuevas sin orden específico, se agregan al final: %s",
            ", ".join(remaining),
        )
        ordered.extend(sorted(remaining))
    return ordered

def merge_dataframes(
        gamelog: pd.DataFrame, aggregated_boxscores: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    original_columns = list(gamelog.columns)
    merged = gamelog.merge(aggregated_boxscores, on=KEY_COLUMNS, how="left")
    new_columns = [column for column in merged.columns if column not in original_columns]
    ordered_new_columns = order_new_columns(new_columns)
    if "GP_RANK" in original_columns:
        insert_position = original_columns.index("GP_RANK")
    else:
        insert_position = len(original_columns)
        logging.warning("La columna GP_RANK no está presente; las métricas se añaden al final")
    before = original_columns[:insert_position]
    after = original_columns[insert_position:]
    final_columns = before + ordered_new_columns + after
    merged = merged[final_columns]
    return merged, new_columns, ordered_new_columns

def validate_final_dataframe(df: pd.DataFrame) -> Tuple[bool, bool]:
    duplicated_pairs = df.duplicated(subset=KEY_COLUMNS, keep=False)
    has_duplicates = bool(duplicated_pairs.any())
    games_per_id = df.groupby("GAME_ID").size()
    two_rows_per_game = bool((games_per_id == 2).all())
    if has_duplicates:
        duplicate_examples = df.loc[duplicated_pairs, KEY_COLUMNS].drop_duplicates().head()
        logging.warning(
            "Se detectaron duplicados en las claves. Ejemplos:\n%s",
            duplicate_examples,
        )
    if not two_rows_per_game:
        logging.warning(
            "Existen GAME_ID con un número distinto de 2 filas. Resumen:\n%s",
            games_per_id.value_counts().to_string(),
        )
    else:
        logging.info("Validación GAME_ID: 2 filas por partido confirmadas")
    return has_duplicates, two_rows_per_game

def filter_nba_teams(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra el DataFrame para mantener solo equipos de la NBA.
    """
    if "TEAM_ID" not in df.columns:
        return df
    return df[df["TEAM_ID"].isin(NBA_TEAM_IDS)].copy()

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s: %(message)s")

    logging.info("Leyendo gamelog desde %s", args.gamelog)
    gamelog_df = ensure_string_keys(read_parquet(args.gamelog), KEY_COLUMNS)

    # Filtrar gamelog para mantener solo equipos NBA
    before_gamelog = len(gamelog_df)
    gamelog_df = gamelog_df[gamelog_df["TEAM_ID"].isin(NBA_TEAM_IDS)].copy()
    after_gamelog = len(gamelog_df)
    logging.info(
        "Gamelog: %d filas, %d columnas (filtrado de %d a %d filas)",
        len(gamelog_df),
        len(gamelog_df.columns),
        before_gamelog,
        after_gamelog
    )

    logging.info("Leyendo boxscores desde %s", args.boxscores)
    boxscores_df = ensure_string_keys(read_parquet(args.boxscores), KEY_COLUMNS)

    # Filtrar boxscores para mantener solo equipos NBA
    before_boxscores = len(boxscores_df)
    boxscores_df = boxscores_df[boxscores_df["TEAM_ID"].isin(NBA_TEAM_IDS)].copy()
    after_boxscores = len(boxscores_df)

    # FILTRO NUEVO: Filtrar jugadores por minutos jugados (al menos 1 minuto)
    boxscores_df = filter_players_by_minutes(boxscores_df, min_minutes=1.0)

    logging.info(
        "Boxscores: %d filas, %d columnas (filtrado de %d a %d filas + filtro por minutos)",
        len(boxscores_df),
        len(boxscores_df.columns),
        before_boxscores,
        after_boxscores
    )

    aggregated_boxscores = aggregate_boxscores(boxscores_df)
    logging.info(
        "Boxscores agregados: %d filas, %d columnas",
        len(aggregated_boxscores),
        len(aggregated_boxscores.columns),
    )

    aggregated_boxscores, collisions = drop_colliding_columns(
        aggregated_boxscores, gamelog_df.columns
    )
    if collisions:
        logging.info(
            "Columnas omitidas por colisión con el gamelog: %s",
            ", ".join(collisions),
        )
    else:
        logging.info("No hubo colisiones de nombres con el gamelog")

    merged_df, new_columns, ordered_new_columns = merge_dataframes(
        gamelog_df, aggregated_boxscores
    )

    # === (MOVIDO AQUÍ) Rolling averages sobre el DataFrame ENRIQUECIDO ===
    combined_columns = list(dict.fromkeys(SUM_COLUMNS + MEAN_COLUMNS))
    rolling_source_columns = [
        column for column in combined_columns if column in merged_df.columns
    ]
    missing_rolling_columns = [
        column for column in combined_columns if column not in merged_df.columns
    ]
    if missing_rolling_columns:
        logging.debug(
            "Columnas no disponibles para rolling averages: %s",
            ", ".join(sorted(missing_rolling_columns)),
        )

    if not rolling_source_columns:
        logging.warning(
            "No se calcularon promedios móviles: ninguna columna esperada está presente"
        )
    elif "GAME_DATE" not in merged_df.columns:
        logging.warning(
            "No se calcularon promedios móviles: la columna GAME_DATE no está presente"
        )
    elif "GAME_ID" not in merged_df.columns:
        logging.warning(
            "No se calcularon promedios móviles: la columna GAME_ID no está presente"
        )
    else:
        if not pd.api.types.is_datetime64_any_dtype(merged_df["GAME_DATE"]):
            merged_df["GAME_DATE"] = pd.to_datetime(
                merged_df["GAME_DATE"], errors="coerce"
            )
        if merged_df["GAME_DATE"].isna().any():
            logging.warning(
                "Algunos valores de GAME_DATE son inválidos; sus rolling averages serán NaN"
            )
        merged_sorted = merged_df.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"])
        rolling_means = (
            merged_sorted
            .groupby("TEAM_ID", as_index=False, sort=False)[rolling_source_columns]
            .apply(lambda g: g.shift(1).rolling(window=10, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        rolling_means = rolling_means.add_prefix("ROLL10_")
        rolling_features = pd.concat(
            [merged_sorted[["TEAM_ID", "GAME_ID"]], rolling_means], axis=1
        )
        merged_df = merged_df.merge(
            rolling_features,
            on=["TEAM_ID", "GAME_ID"],
            how="left",
        )

        logging.info(
            "Rolling averages agregadas: %d columnas (ventana=10, shift=1)",
            len(rolling_means.columns),
        )
    # === FIN BLOQUE MOVIDO ===

    # Filtrar el resultado final por si acaso (aunque ya debería estar filtrado)
    before_final = len(merged_df)
    merged_df = merged_df[merged_df["TEAM_ID"].isin(NBA_TEAM_IDS)].copy()
    after_final = len(merged_df)

    if before_final != after_final:
        logging.info(f"Filtrado final: {before_final} -> {after_final} filas")

    logging.info(
        "Total columnas nuevas agregadas: %d", len(new_columns)
    )
    logging.debug(
        "Orden de inserción aplicado: %s",
        ", ".join(ordered_new_columns),
    )
    logging.info(
        "DataFrame final: %d filas, %d columnas",
        len(merged_df),
        len(merged_df.columns),
    )

    validate_final_dataframe(merged_df)

    if args.dry_run:
        logging.info("Ejecución en modo dry-run: no se escribe salida")
        return

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(
        output_path,
        engine="pyarrow",
        index=False,
    )
    logging.info("Archivo final sobrescrito en %s", output_path)


if __name__ == "__main__":
    main()
