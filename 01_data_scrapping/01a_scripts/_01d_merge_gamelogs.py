#!/usr/bin/env python3
"""Mergea gamelogs de equipos apilando archivos por temporada."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:  # Optional dependency
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None  # type: ignore

INPUT_ROOT = Path(
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/teamgamelogs_by_game"
)
INTERMEDIATE_ROOT = Path(
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00b_intermediate"
)
FINAL_ROOT = Path(
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00c_final"
)
OUTPUT_FILENAME = "teamgamelogs_by_game.parquet"
KEY_COLUMNS = ["GAME_ID", "TEAM_ID"]


def detect_seasons(base_path: Path) -> List[str]:
    """Devuelve la lista de temporadas disponibles en el directorio base."""

    if not base_path.exists():
        logging.warning("El directorio de entrada %s no existe", base_path)
        return []
    seasons = [entry.name for entry in base_path.iterdir() if entry.is_dir()]
    return sorted(seasons)


def iter_parquet_files(base_path: Path) -> List[Path]:
    """Itera sobre los archivos parquet dentro de un directorio dado."""

    if not base_path.exists():
        logging.warning("El directorio %s no existe", base_path)
        return []
    return [path for path in sorted(base_path.glob("*.parquet")) if path.is_file()]


def read_parquet(path: Path, engine: str) -> Optional[pd.DataFrame]:
    """Lee un archivo parquet con el motor especificado."""

    try:
        if engine == "polars":
            if pl is None:
                raise ImportError("polars no está disponible")
            frame = pl.read_parquet(path)
            df = frame.to_pandas()
        else:
            df = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - robustez en ejecución
        logging.warning("No se pudo leer %s: %s", path, exc)
        return None
    return df


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura que las columnas clave tengan tipo string."""

    df = df.copy()
    for column in KEY_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("string")
    return df


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas completamente vacías, preservando claves."""

    non_key_columns = [col for col in df.columns if col not in KEY_COLUMNS]
    empty_columns = [col for col in non_key_columns if df[col].isna().all()]
    if empty_columns:
        logging.info("Columnas vacías eliminadas: %s", ", ".join(sorted(empty_columns)))
        df = df.drop(columns=empty_columns)
    return df


def process_season(season: str, engine: str, dry_run: bool) -> None:
    """Procesa una temporada: lee, concatena y escribe los gamelogs."""

    season_dir = INPUT_ROOT / season
    parquet_files = list(iter_parquet_files(season_dir))
    if not parquet_files:
        logging.warning("Temporada %s sin archivos parquet en %s", season, season_dir)
        return

    frames: List[pd.DataFrame] = []
    for file_path in parquet_files:
        df = read_parquet(file_path, engine)
        if df is None:
            continue
        frames.append(df)

    if not frames:
        logging.warning("Temporada %s sin datos válidos tras la lectura", season)
        return

    merged_df = pd.concat(frames, ignore_index=True, sort=False)
    merged_df = normalize_key_columns(merged_df)
    merged_df = drop_empty_columns(merged_df)

    logging.info(
        "Temporada %s - archivos leídos: %d, filas acumuladas: %d, columnas: %d",
        season,
        len(frames),
        len(merged_df),
        len(merged_df.columns),
    )

    if dry_run:
        logging.info("Temporada %s - ejecución en modo dry-run, sin escritura", season)
        return

    for base_dir in (INTERMEDIATE_ROOT, FINAL_ROOT):
        output_dir = base_dir / season
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / OUTPUT_FILENAME
        merged_df.to_parquet(
            output_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )
        logging.info("Temporada %s - archivo escrito en %s", season, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge vertical de gamelogs de equipos por temporada"
    )
    parser.add_argument(
        "--season",
        help="Temporada específica a procesar (ej. 2024-25)",
    )
    parser.add_argument(
        "--engine",
        choices=["pandas", "polars"],
        default="pandas",
        help="Motor de lectura de parquet",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecuta sin escribir archivos de salida",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if args.season:
        seasons = [args.season]
    else:
        seasons = detect_seasons(INPUT_ROOT)
        if not seasons:
            logging.error("No se detectaron temporadas en %s", INPUT_ROOT)
            return
        logging.info("Temporadas detectadas: %s", ", ".join(seasons))

    for season in seasons:
        if not (INPUT_ROOT / season).exists():
            logging.warning("La temporada %s no se encontró en %s", season, INPUT_ROOT)
            continue
        process_season(season, args.engine, args.dry_run)


if __name__ == "__main__":
    main()
