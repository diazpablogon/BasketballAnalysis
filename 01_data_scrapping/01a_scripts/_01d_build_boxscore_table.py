#!/usr/bin/env python3
"""Combina los box scores de todos los endpoints en un único parquet por temporada.

Uso:
    python _01d_build_boxscore_table.py --season 2024-25 \
        --base-dir /Users/pablo/Documents/BigData/BasketballAnalysis
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import pandas as pd


@dataclass(frozen=True)
class EndpointConfig:
    slug: str
    prefix: str


ENDPOINTS: tuple[EndpointConfig, ...] = (
    EndpointConfig("boxscore_traditional_v2", "traditional"),
    EndpointConfig("boxscore_advanced_v2", "advanced"),
    EndpointConfig("boxscore_fourfactors_v2", "fourfactors"),
    EndpointConfig("boxscore_matchups_v3", "matchups"),
    EndpointConfig("boxscore_misc_v2", "misc"),
    EndpointConfig("boxscore_scoring_v2", "scoring"),
    EndpointConfig("boxscore_usage_v2", "usage"),
)

# Columnas candidatas a utilizar para el join (en orden de prioridad).
PREFERRED_JOIN_COLUMNS: list[str] = [
    "GAME_ID",
    "TEAM_ID",
    "PLAYER_ID",
    "TEAM_ABBREVIATION",
    "PLAYER_NAME",
    "TEAM_CITY",
    "TEAM_NAME",
]

REQUIRED_JOIN_COLUMNS = {"GAME_ID", "TEAM_ID", "PLAYER_ID"}
METADATA_COLUMNS = {"season", "game_id", "endpoint"}


class EmojiFormatter(logging.Formatter):
    """Añade un emoji acorde al nivel del log."""

    LEVEL_EMOJI = {
        logging.DEBUG: "🐞",
        logging.INFO: "✅",
        logging.WARNING: "⚠️",
        logging.ERROR: "❌",
        logging.CRITICAL: "💥",
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 (documentado arriba)
        record.emoji = self.LEVEL_EMOJI.get(record.levelno, "✅")
        return super().format(record)


logger = logging.getLogger("build_boxscore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fusiona todos los box scores de una temporada en un único parquet."
    )
    default_base = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=default_base,
        help="Directorio raíz del repositorio (por defecto, la raíz detectada).",
    )
    parser.add_argument(
        "--season",
        default="2024-25",
        help="Temporada a procesar (ejemplo: 2024-25).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Nivel de log deseado (por defecto, INFO).",
    )
    return parser.parse_args()


def discover_game_files(base_dir: Path, season: str) -> Dict[str, Dict[str, Path]]:
    """Devuelve un diccionario GAME_ID -> {slug: ruta parquet}."""
    root = base_dir / "00_data" / "00a_raw" / "boxscore"
    mapping: Dict[str, Dict[str, Path]] = defaultdict(dict)

    for endpoint in ENDPOINTS:
        season_dir = root / endpoint.slug / season
        if not season_dir.exists():
            logger.warning("No existe el directorio de %s: %s", endpoint.slug, season_dir)
            continue

        for file in season_dir.glob(f"{endpoint.slug}__*.parquet"):
            gid = file.stem.split("__")[-1]
            mapping[gid][endpoint.slug] = file

    return mapping


def _transform_matchups(df: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """Ajusta el boxscore de matchups a nivel jugador-ofensivo si es necesario."""

    rename_map = {
        "OFF_TEAM_ID": "TEAM_ID",
        "OFF_TEAM_ABBREVIATION": "TEAM_ABBREVIATION",
        "OFF_TEAM_NICKNAME": "TEAM_NAME",
        "OFF_TEAM_CITY": "TEAM_CITY",
        "OFF_PLAYER_ID": "PLAYER_ID",
        "OFF_PLAYER_NAME": "PLAYER_NAME",
    }

    missing = [col for col in rename_map if col not in df.columns]
    if not missing:
        df = df.rename(columns=rename_map)
    else:
        # Si ya está en formato jugador-equipo solo homogenizamos duplicados.
        if not REQUIRED_JOIN_COLUMNS.issubset(df.columns):
            raise ValueError(
                "No se pueden armonizar columnas de matchups para "
                f"GAME_ID {game_id}: faltan {missing}"
            )

    group_cols = list(REQUIRED_JOIN_COLUMNS)

    # Agregamos únicamente cuando hay filas duplicadas para las claves.
    if df.duplicated(subset=group_cols).any():
        numeric_cols = df.select_dtypes(include="number").columns.difference(group_cols)
        agg_spec: dict[str, str] = {col: "sum" for col in numeric_cols}

        non_numeric = [
            col for col in df.columns if col not in group_cols and col not in numeric_cols
        ]
        for col in non_numeric:
            agg_spec[col] = "first"

        df = df.groupby(group_cols, dropna=False).agg(agg_spec).reset_index()

    return df


SPECIAL_TRANSFORMS = {
    "boxscore_matchups_v3": _transform_matchups,
}


def apply_endpoint_transform(slug: str, df: pd.DataFrame, game_id: str) -> pd.DataFrame:
    transform = SPECIAL_TRANSFORMS.get(slug)
    if transform is None:
        return df
    try:
        return transform(df, game_id)
    except ValueError as exc:  # pragma: no cover - dependerá de datos externos
        logger.warning("%s", exc)
        return pd.DataFrame()


def load_game_frames(
    game_id: str, game_files: Mapping[str, Path]
) -> Dict[str, pd.DataFrame]:
    """Carga las tablas de un partido y descarta las que no cuadran."""
    frames: Dict[str, pd.DataFrame] = {}
    expected_rows: int | None = None

    for endpoint in ENDPOINTS:
        path = game_files.get(endpoint.slug)
        if path is None:
            continue

        df = pd.read_parquet(path)
        if df.empty:
            logger.warning("Dataset vacío %s para %s", endpoint.slug, game_id)
            continue

        df = df.loc[:, ~df.columns.duplicated()]
        df = df.drop(columns=[c for c in METADATA_COLUMNS if c in df.columns], errors="ignore")
        df = apply_endpoint_transform(endpoint.slug, df, game_id)
        if df.empty:
            continue

        if not REQUIRED_JOIN_COLUMNS.issubset(df.columns):
            missing = REQUIRED_JOIN_COLUMNS - set(df.columns)
            logger.warning(
                "Saltando %s para %s: faltan columnas clave %s",
                endpoint.slug,
                game_id,
                sorted(missing),
            )
            continue

        df = df.drop_duplicates(subset=list(REQUIRED_JOIN_COLUMNS)).reset_index(drop=True)

        if expected_rows is None:
            expected_rows = len(df)
        elif len(df) != expected_rows:
            logger.warning(
                "Saltando %s para %s: %d filas esperadas, %d encontradas",
                endpoint.slug,
                game_id,
                expected_rows,
                len(df),
            )
            continue

        frames[endpoint.slug] = df

    return frames


def infer_join_columns(frames: Mapping[str, pd.DataFrame]) -> list[str]:
    """Selecciona las columnas comunes que servirán como clave de unión."""
    join_cols: list[str] = [
        col
        for col in PREFERRED_JOIN_COLUMNS
        if all(col in df.columns for df in frames.values())
    ]
    if not REQUIRED_JOIN_COLUMNS.issubset(join_cols):
        raise ValueError(
            f"No hay suficientes columnas comunes para unir: se necesitan {REQUIRED_JOIN_COLUMNS}"
        )
    return join_cols


def merge_game_frames(frames: Mapping[str, pd.DataFrame], join_cols: Iterable[str]) -> pd.DataFrame:
    """Une horizontalmente todas las tablas de un partido."""
    join_cols = list(join_cols)
    slug_order = [endpoint.slug for endpoint in ENDPOINTS if endpoint.slug in frames]
    base_slug = slug_order[0]
    merged = frames[base_slug].copy()

    for slug in slug_order[1:]:
        df = frames[slug]
        extra_cols = [col for col in df.columns if col not in join_cols]
        if not extra_cols:
            continue

        prefix = next((cfg.prefix for cfg in ENDPOINTS if cfg.slug == slug), slug)
        rename_map = {col: f"{prefix}__{col}" for col in extra_cols}
        piece = df[join_cols + extra_cols].rename(columns=rename_map)
        merged = merged.merge(piece, on=join_cols, how="left", validate="one_to_one")

    return merged


def build_season_boxscore(base_dir: Path, season: str) -> Path:
    """Procesa la temporada completa y devuelve la ruta del parquet final."""
    game_mapping = discover_game_files(base_dir, season)
    if not game_mapping:
        raise FileNotFoundError(f"No se encontraron archivos parquet para la temporada {season}")

    merged_games: list[pd.DataFrame] = []
    skipped_games = 0
    global_join_cols: list[str] | None = None

    for game_id, files in sorted(game_mapping.items()):
        frames = load_game_frames(game_id, files)
        if len(frames) < 2:
            logger.warning(
                "GAME_ID %s omitido: solo %d datasets alineados",
                game_id,
                len(frames),
            )
            skipped_games += 1
            continue

        try:
            join_cols = infer_join_columns(frames)
        except ValueError as exc:
            logger.warning("GAME_ID %s omitido: %s", game_id, exc)
            skipped_games += 1
            continue

        merged = merge_game_frames(frames, join_cols)
        merged_games.append(merged)

        global_join_cols = (
            join_cols
            if global_join_cols is None
            else [col for col in global_join_cols if col in join_cols]
        )

    if not merged_games:
        raise RuntimeError("No se pudo fusionar ningún partido con columnas comunes.")

    season_df = pd.concat(merged_games, ignore_index=True)
    output_path = base_dir / "00_data" / "00c_final" / season / "boxscore.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    season_df.to_parquet(output_path, index=False)

    logger.info("Columnas de unión finales: %s", global_join_cols)
    logger.info("Partidos fusionados: %d | Partidos omitidos: %d", len(merged_games), skipped_games)
    logger.info("Shape final: %s", season_df.shape)
    logger.info("Parquet guardado en: %s", output_path)

    return output_path


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(EmojiFormatter("%(emoji)s %(message)s"))
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    logger.addHandler(handler)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    base_dir = args.base_dir.expanduser().resolve()
    try:
        build_season_boxscore(base_dir=base_dir, season=args.season)
    except Exception as exc:  # pragma: no cover - comunicación al usuario final
        logger.error("%s", exc)
        raise


if __name__ == "__main__":
    main()
