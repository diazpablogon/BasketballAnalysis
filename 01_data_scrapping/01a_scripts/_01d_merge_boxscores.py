#!/usr/bin/env python3
"""Merge player boxscores per season."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:  # Optional dependency
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None  # type: ignore

RAW_ROOT = Path("/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/boxscore")
OUTPUT_ROOT = Path("/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00c_final")
MERGE_ORDER = [
    "boxscore_traditional_v2",
    "boxscore_advanced_v2",
    "boxscore_fourfactors_v2",
    "boxscore_misc_v2",
    "boxscore_scoring_v2",
    "boxscore_usage_v2",
    "boxscore_matchups_v3",
]
KEY_COLUMNS = ["GAME_ID", "TEAM_ID", "PLAYER_ID"]


def detect_seasons(candidate_dirs: Iterable[str]) -> List[str]:
    seasons = set()
    for subdir in candidate_dirs:
        path = RAW_ROOT / subdir
        if "summary" in subdir.lower() or not path.exists() or not path.is_dir():
            continue
        for child in path.iterdir():
            if child.is_dir() and "summary" not in child.name.lower():
                seasons.add(child.name)
    return sorted(seasons)


def iter_parquet_files(base_path: Path) -> Iterable[Path]:
    for file_path in sorted(base_path.rglob("*.parquet")):
        if any("summary" in part.lower() for part in file_path.parts):
            continue
        yield file_path


def read_parquet(path: Path, engine: str) -> Optional[pd.DataFrame]:
    try:
        if engine == "polars":
            if pl is None:
                raise ImportError("polars no está disponible")
            frame = pl.read_parquet(path)
            df = frame.to_pandas()
        else:
            df = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - runtime robustness
        logging.warning("No se pudo leer %s: %s", path, exc)
        return None
    return df


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for key in KEY_COLUMNS:
        if key in df.columns:
            df[key] = df[key].astype("string")
    for column in df.columns:
        if column in KEY_COLUMNS:
            continue
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            df[column] = series.astype(float)
            continue
        converted = pd.to_numeric(series, errors="ignore")
        if pd.api.types.is_numeric_dtype(converted):
            df[column] = converted.astype(float)
    return df


def drop_duplicate_columns(df: pd.DataFrame, existing_columns: Iterable[str]) -> pd.DataFrame:
    existing_set = set(existing_columns)
    keep_columns = [col for col in df.columns if col in KEY_COLUMNS or col not in existing_set]
    if len(keep_columns) != len(df.columns):
        df = df[keep_columns]
    return df


def merge_sources_for_season(season: str, engine: str) -> Optional[Tuple[pd.DataFrame, Dict[str, int]]]:
    merged_df: Optional[pd.DataFrame] = None
    files_read: Dict[str, int] = {}
    for source in MERGE_ORDER:
        source_path = RAW_ROOT / source / season
        if not source_path.exists() or not source_path.is_dir():
            files_read[source] = 0
            continue
        valid_frames: List[pd.DataFrame] = []
        for parquet_file in iter_parquet_files(source_path):
            df = read_parquet(parquet_file, engine)
            if df is None:
                continue
            missing_keys = [col for col in KEY_COLUMNS if col not in df.columns]
            if missing_keys:
                logging.warning(
                    "Archivo %s omitido: columnas faltantes %s",
                    parquet_file,
                    missing_keys,
                )
                continue
            df = normalize_types(df)
            valid_frames.append(df)
        files_read[source] = len(valid_frames)
        if not valid_frames:
            continue
        source_df = pd.concat(valid_frames, ignore_index=True)
        if merged_df is None:
            merged_df = source_df
            continue
        source_df = drop_duplicate_columns(source_df, merged_df.columns)
        merged_df = merged_df.merge(source_df, on=KEY_COLUMNS, how="outer")
    if merged_df is None:
        return None
    for key in KEY_COLUMNS:
        if key in merged_df.columns:
            merged_df[key] = merged_df[key].astype("string")
    non_key_columns = [col for col in merged_df.columns if col not in KEY_COLUMNS]
    empty_columns = [col for col in non_key_columns if merged_df[col].isna().all()]
    if empty_columns:
        merged_df = merged_df.drop(columns=empty_columns)
    non_key_columns = [col for col in merged_df.columns if col not in KEY_COLUMNS]
    ordered_columns = KEY_COLUMNS + sorted(non_key_columns)
    merged_df = merged_df[ordered_columns]
    return merged_df, files_read


def process_season(season: str, engine: str, dry_run: bool) -> None:
    result = merge_sources_for_season(season, engine)
    if result is None:
        logging.warning("Temporada %s sin datos válidos", season)
        return
    merged_df, files_read = result
    logging.info(
        "Temporada %s - archivos válidos por fuente: %s",
        season,
        ", ".join(f"{src}: {count}" for src, count in files_read.items()),
    )
    logging.info(
        "Temporada %s - filas: %d, columnas: %d",
        season,
        len(merged_df),
        len(merged_df.columns),
    )
    if dry_run:
        logging.info("Temporada %s - ejecución en modo dry-run, sin escritura", season)
        return
    output_dir = OUTPUT_ROOT / season
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "boxscores.parquet"
    merged_df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    logging.info("Temporada %s - archivo escrito en %s", season, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge de boxscores por temporada")
    parser.add_argument("--season", help="Temporada específica a procesar (ej. 2024-25)")
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
    seasons = detect_seasons(MERGE_ORDER)
    if not seasons:
        logging.error("No se detectaron temporadas en %s", RAW_ROOT)
        return
    logging.info("Temporadas detectadas: %s", ", ".join(seasons))
    if args.season:
        if args.season not in seasons:
            logging.error("La temporada %s no se encontró en las fuentes", args.season)
            return
        seasons = [args.season]
    for season in seasons:
        process_season(season, args.engine, args.dry_run)


if __name__ == "__main__":
    main()
