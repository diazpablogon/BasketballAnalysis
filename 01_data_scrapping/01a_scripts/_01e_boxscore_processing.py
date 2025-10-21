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
    "REB",
    "STL",
    "TO",
    "PTS",
    "PTS_2ND_CHANCE",
    "PTS_FB",
    "PTS_OFF_TOV",
    "PTS_PAINT",
]

MEAN_COLUMNS = [
    "AST_PCT",
    "AST_RATIO",
    "AST_TOV",
    "DEF_RATING",
    "DREB_PCT",
    "EFG_PCT",
    "E_DEF_RATING",
    "E_NET_RATING",
    "E_OFF_RATING",
    "E_PACE",
    "E_USG_PCT",
    "FG3_PCT",
    "FG_PCT",
    "FT_PCT",
    "FTA_RATE",
    "NET_RATING",
    "OFF_RATING",
    "OPP_EFG_PCT",
    "OPP_FTA_RATE",
    "OPP_OREB_PCT",
    "OPP_PTS_2ND_CHANCE",
    "OPP_PTS_FB",
    "OPP_PTS_OFF_TOV",
    "OPP_PTS_PAINT",
    "OPP_TOV_PCT",
    "OREB_PCT",
    "PACE",
    "PACE_PER40",
    "PCT_AST",
    "PCT_AST_2PM",
    "PCT_AST_3PM",
    "PCT_AST_FGM",
    "PCT_BLK",
    "PCT_BLKA",
    "PCT_DREB",
    "PCT_FG3A",
    "PCT_FG3M",
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

IGNORE_COLUMNS = {
    "PLAYER_ID",
    "PLAYER_NAME",
    "NICKNAME",
    "TEAM_ABBREVIATION",
    "TEAM_CITY",
    "START_POSITION",
    "COMMENT",
    "endpoint",
    "game_id",
    "season",
}
IGNORE_COLUMNS_LOWER = {column.lower() for column in IGNORE_COLUMNS}

INSERTION_ORDER_GROUPS: Tuple[Sequence[str], ...] = (
    (
        "OFF_RATING",
        "DEF_RATING",
        "NET_RATING",
        "E_OFF_RATING",
        "E_DEF_RATING",
        "E_NET_RATING",
    ),
    (
        "PACE",
        "PACE_PER40",
        "E_PACE",
        "POSS",
        "USG_PCT",
        "E_USG_PCT",
        "TM_TOV_PCT",
    ),
    (
        "EFG_PCT",
        "TS_PCT",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "FTA_RATE",
    ),
    (
        "OREB_PCT",
        "DREB_PCT",
        "REB_PCT",
    ),
    (
        "AST_PCT",
        "AST_RATIO",
        "AST_TOV",
        "PCT_AST",
        "PCT_AST_2PM",
        "PCT_AST_3PM",
        "PCT_AST_FGM",
        "PCT_UAST_2PM",
        "PCT_UAST_3PM",
        "PCT_UAST_FGM",
        "PCT_TOV",
        "PCT_STL",
        "PCT_BLK",
        "PCT_BLKA",
    ),
    (
        "PCT_FGA",
        "PCT_FGA_2PT",
        "PCT_FGA_3PT",
        "PCT_FGM",
        "PCT_FG3A",
        "PCT_FG3M",
        "PCT_FTA",
        "PCT_FTM",
        "PCT_PTS",
        "PCT_PTS_2PT",
        "PCT_PTS_2PT_MR",
        "PCT_PTS_3PT",
        "PCT_PTS_FB",
        "PCT_PTS_FT",
        "PCT_PTS_OFF_TOV",
        "PCT_PTS_PAINT",
    ),
    (
        "OPP_EFG_PCT",
        "OPP_FTA_RATE",
        "OPP_OREB_PCT",
        "OPP_TOV_PCT",
        "OPP_PTS_2ND_CHANCE",
        "OPP_PTS_FB",
        "OPP_PTS_OFF_TOV",
        "OPP_PTS_PAINT",
    ),
    ("PIE",),
)

TOTALS_PRIORITY = (
    "PTS_2ND_CHANCE",
    "PTS_FB",
    "PTS_OFF_TOV",
    "PTS_PAINT",
    "TO",
    "BLKA",
    "PFD",
)


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
        help="Solo mostrar estadísticas sin escribir archivo",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Nivel de logging (DEBUG, INFO, WARNING, ...)",
    )
    return parser.parse_args()


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo {path}")
    return pd.read_parquet(path)


def ensure_string_keys(df: pd.DataFrame, key_columns: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for column in key_columns:
        if column not in df.columns:
            raise KeyError(f"La columna clave {column} no está presente en el DataFrame")
        df[column] = df[column].astype("string")
    return df


def drop_ignored_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        column
        for column in df.columns
        if column not in KEY_COLUMNS
        and (column in IGNORE_COLUMNS or column.lower() in IGNORE_COLUMNS_LOWER)
    ]
    if columns_to_drop:
        logging.debug("Columnas ignoradas del boxscore: %s", ", ".join(columns_to_drop))
        df = df.drop(columns=columns_to_drop)
    return df


def build_aggregation_dict(df: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
    aggregation: Dict[str, str] = {}
    missing_columns: List[str] = []
    for column in SUM_COLUMNS:
        if column in df.columns:
            aggregation[column] = "sum"
        else:
            missing_columns.append(column)
    for column in MEAN_COLUMNS:
        if column in df.columns:
            aggregation[column] = "mean"
        else:
            missing_columns.append(column)
    if missing_columns:
        logging.debug(
            "Columnas del boxscore ausentes y no agregadas: %s",
            ", ".join(sorted(set(missing_columns))),
        )
    return aggregation, missing_columns


def aggregate_boxscores(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = drop_ignored_columns(df)
    aggregation, _ = build_aggregation_dict(cleaned_df)
    if not aggregation:
        logging.warning("No hay columnas numéricas seleccionadas para agregar")
        grouped = cleaned_df[KEY_COLUMNS].drop_duplicates()
        return grouped
    grouped = (
        cleaned_df.groupby(KEY_COLUMNS, as_index=False).agg(aggregation)
    )

    # Redondear todas las columnas numéricas a 5 decimales
    numeric_columns = grouped.select_dtypes(include=['number']).columns
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
    remaining_totals = [
        column
        for column in SUM_COLUMNS
        if column in new_set and column not in ordered
    ]
    ordered.extend(remaining_totals)
    remaining = [column for column in new_columns if column not in ordered]
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    logging.info("Leyendo gamelog desde %s", args.gamelog)
    gamelog_df = ensure_string_keys(read_parquet(args.gamelog), KEY_COLUMNS)
    logging.info(
        "Gamelog: %d filas, %d columnas",
        len(gamelog_df),
        len(gamelog_df.columns),
    )

    logging.info("Leyendo boxscores desde %s", args.boxscores)
    boxscores_df = ensure_string_keys(read_parquet(args.boxscores), KEY_COLUMNS)
    logging.info(
        "Boxscores: %d filas, %d columnas",
        len(boxscores_df),
        len(boxscores_df.columns),
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
