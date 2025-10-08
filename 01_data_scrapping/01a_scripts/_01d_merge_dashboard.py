#!/usr/bin/env python3
"""Mergea dashboards de equipos siguiendo la estructura de boxscores."""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

DEFAULT_BASE_DIR = Path("/Users/pablo/Documents/BigData/BasketballAnalysis/00_data")
RAW_SUBDIR = Path("00a_raw/team_dashboard")
OUTPUT_SUBDIR = Path("00c_final")

FILE_PATTERN = re.compile(
    r"__?(?P<team_id>\d{6,})?__?dataset_(?P<dataset>\d+)\.parquet$",
    re.IGNORECASE,
)

SORT_KEY_PRIORITY = [
    "TEAM_ID",
    "PLAYER_ID",
    "GROUP_ID",
    "GROUP_SET",
    "GROUP_VALUE",
    "LINEUP_ID",
    "MATCHUP_ID",
    "SHOT_TYPE",
    "SHOT_ZONE_BASIC",
    "SHOT_ZONE_AREA",
    "SHOT_ZONE_RANGE",
    "MEASURE_TYPE",
    "SEASON",
    "Season",
]


@dataclass
class DashboardFile:
    path: Path
    team_id: Optional[int]
    dataset: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge de dashboards de equipo")
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help="Directorio base de datos (por defecto %(default)s)",
    )
    parser.add_argument(
        "--season",
        default="2024-25",
        help="Temporada a procesar (por defecto %(default)s)",
    )
    parser.add_argument(
        "--season-type",
        default="Regular Season",
        help="Tipo de temporada (por defecto %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de workers para futuras paralelizaciones",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecuta sin escribir archivos de salida",
    )
    return parser.parse_args()


def discover_dashboard_files(directory: Path) -> List[DashboardFile]:
    files: List[DashboardFile] = []
    for path in sorted(directory.rglob("*.parquet")):
        match = FILE_PATTERN.search(path.name)
        if not match:
            logging.warning("Archivo ignorado por nombre inesperado: %s", path)
            continue
        team_id_str = match.group("team_id")
        team_id = int(team_id_str) if team_id_str else None
        dataset = int(match.group("dataset"))
        files.append(DashboardFile(path=path, team_id=team_id, dataset=dataset))
    return files


def read_dashboard_frame(
    dashboard_file: DashboardFile,
    season: str,
    season_type: str,
) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_parquet(dashboard_file.path)
    except Exception as exc:  # pragma: no cover - robustez en runtime
        logging.warning("No se pudo leer %s: %s", dashboard_file.path, exc)
        return None
    df = df.copy()
    if "TEAM_ID" not in df.columns and dashboard_file.team_id is not None:
        df["TEAM_ID"] = dashboard_file.team_id
    if "TEAM_ID" in df.columns:
        df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce").astype("Int64")
    if "PLAYER_ID" in df.columns:
        df["PLAYER_ID"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce").astype("Int64")
    for column in ("Season", "SEASON"):
        if column in df.columns:
            df[column] = season
    for column in ("SeasonType", "Season_Type", "SEASON_TYPE"):
        if column in df.columns:
            df[column] = season_type
    return df


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    keys = [col for col in SORT_KEY_PRIORITY if col in df.columns]
    remaining = [col for col in df.columns if col not in keys]
    ordered = keys + sorted(remaining)
    return df[ordered]


def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    sort_keys = [col for col in SORT_KEY_PRIORITY if col in df.columns]
    if not sort_keys:
        return df
    return df.sort_values(sort_keys).reset_index(drop=True)


def concat_vertical(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = order_columns(combined)
    combined = sort_dataframe(combined)
    return combined


def remove_identical_columns(base: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    overlap = [col for col in candidate.columns if col in base.columns]
    drop_cols: List[str] = []
    for col in overlap:
        if base[col].equals(candidate[col]):
            drop_cols.append(col)
    if drop_cols:
        candidate = candidate.drop(columns=drop_cols)
    return candidate


def rename_collisions(
    base: pd.DataFrame,
    candidate: pd.DataFrame,
    keys: Sequence[str],
    left_suffix: str,
    right_suffix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    collisions = [
        col for col in candidate.columns if col in base.columns and col not in keys
    ]
    if not collisions:
        return base, candidate
    base = base.copy()
    candidate = candidate.copy()
    for col in collisions:
        if base[col].equals(candidate[col]):
            candidate = candidate.drop(columns=[col])
            continue
        base.rename(columns={col: f"{col}{left_suffix}"}, inplace=True)
        candidate.rename(columns={col: f"{col}{right_suffix}"}, inplace=True)
    return base, candidate


def infer_join_keys(frames: Sequence[pd.DataFrame]) -> List[str]:
    if not frames:
        return []
    common_columns = set(frames[0].columns)
    for df in frames[1:]:
        common_columns &= set(df.columns)
    if not common_columns:
        return []
    candidate_keys = [
        col
        for col in common_columns
        if not pd.api.types.is_float_dtype(frames[0][col])
    ]
    if "TEAM_ID" in frames[0].columns and "TEAM_ID" not in candidate_keys:
        candidate_keys.append("TEAM_ID")
    ordered = [
        col for col in SORT_KEY_PRIORITY if col in candidate_keys
    ]
    for col in sorted(candidate_keys):
        if col not in ordered:
            ordered.append(col)
    valid_keys = [col for col in ordered if all(col in df.columns for df in frames)]
    return valid_keys


def merge_horizontal(
    frames: Sequence[pd.DataFrame],
    suffixes: Sequence[str],
    description: str,
) -> Optional[pd.DataFrame]:
    if not frames:
        return None
    keys = infer_join_keys(frames)
    if keys:
        base = frames[0]
        for idx, df in enumerate(frames[1:], start=1):
            left_suffix = suffixes[0]
            right_suffix = suffixes[idx]
            merged_left, adjusted = rename_collisions(base, df, keys, left_suffix, right_suffix)
            try:
                base = merged_left.merge(
                    adjusted,
                    on=keys,
                    how="outer",
                    validate="one_to_one",
                )
            except Exception as exc:
                logging.warning(
                    "%s - merge fallido con llaves %s: %s. Se intentará concat", description, keys, exc
                )
                keys = []
                break
        if keys:
            return base
    lengths = {len(df) for df in frames}
    if len(lengths) != 1:
        logging.warning(
            "%s - no fue posible alinear por índice debido a longitudes distintas: %s",
            description,
            lengths,
        )
        return None
    aligned = [df.reset_index(drop=True) for df in frames]
    base = aligned[0]
    for idx, df in enumerate(aligned[1:], start=1):
        left_suffix = suffixes[0]
        right_suffix = suffixes[idx]
        df = remove_identical_columns(base, df)
        base, df = rename_collisions(base, df, [], left_suffix, right_suffix)
        overlap = [col for col in df.columns if col in base.columns]
        if overlap:
            df = df.drop(columns=overlap)
        base = pd.concat([base, df], axis=1)
    return base


def process_team_dash_lineups(
    files: List[DashboardFile],
    season: str,
    season_type: str,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for dashboard_file in files:
        if dashboard_file.dataset != 1:
            continue
        df = read_dashboard_frame(dashboard_file, season, season_type)
        if df is not None:
            frames.append(df)
    return concat_vertical(frames)


def process_team_dash_pt_pass(
    files: List[DashboardFile],
    season: str,
    season_type: str,
) -> pd.DataFrame:
    files_by_team: Dict[int, Dict[int, DashboardFile]] = {}
    for dashboard_file in files:
        if dashboard_file.team_id is None:
            continue
        files_by_team.setdefault(dashboard_file.team_id, {})[dashboard_file.dataset] = dashboard_file
    merged_frames: List[pd.DataFrame] = []
    for team_id, datasets in sorted(files_by_team.items()):
        available_frames: List[pd.DataFrame] = []
        suffixes: List[str] = []
        for dataset in (0, 1):
            dashboard_file = datasets.get(dataset)
            if dashboard_file is None:
                continue
            df = read_dashboard_frame(dashboard_file, season, season_type)
            if df is None:
                continue
            df = df.copy()
            df["TEAM_ID"] = pd.Series([team_id] * len(df), dtype="Int64")
            available_frames.append(df)
            suffixes.append("__d0" if dataset == 0 else "__d1")
        if not available_frames:
            logging.warning("team_dash_pt_pass - TEAM_ID %s sin datasets 0/1", team_id)
            continue
        description = f"team_dash_pt_pass TEAM_ID {team_id}"
        merged = merge_horizontal(available_frames, suffixes, description)
        if merged is None:
            logging.warning("%s - no se obtuvo dataframe combinado", description)
            continue
        merged_frames.append(merged)
    return concat_vertical(merged_frames)


def process_team_dash_pt_reb(
    files: List[DashboardFile],
    season: str,
    season_type: str,
) -> Dict[int, pd.DataFrame]:
    frames_by_dataset: Dict[int, List[pd.DataFrame]] = {i: [] for i in range(1, 5)}
    for dashboard_file in files:
        if dashboard_file.dataset not in frames_by_dataset:
            continue
        df = read_dashboard_frame(dashboard_file, season, season_type)
        if df is not None:
            frames_by_dataset[dashboard_file.dataset].append(df)
    return {dataset: concat_vertical(frames) for dataset, frames in frames_by_dataset.items()}


def process_team_dash_pt_shots(
    files: List[DashboardFile],
    season: str,
    season_type: str,
) -> Dict[int, pd.DataFrame]:
    frames_by_dataset: Dict[int, List[pd.DataFrame]] = {i: [] for i in range(6)}
    for dashboard_file in files:
        if dashboard_file.dataset not in frames_by_dataset:
            continue
        df = read_dashboard_frame(dashboard_file, season, season_type)
        if df is not None:
            frames_by_dataset[dashboard_file.dataset].append(df)
    return {dataset: concat_vertical(frames) for dataset, frames in frames_by_dataset.items()}


def process_simple_stack(
    files: List[DashboardFile],
    allowed_datasets: Iterable[int],
    season: str,
    season_type: str,
) -> Dict[int, pd.DataFrame]:
    allowed = set(allowed_datasets)
    frames_by_dataset: Dict[int, List[pd.DataFrame]] = {dataset: [] for dataset in allowed}
    for dashboard_file in files:
        if dashboard_file.dataset not in allowed:
            continue
        df = read_dashboard_frame(dashboard_file, season, season_type)
        if df is not None:
            frames_by_dataset[dashboard_file.dataset].append(df)
    return {dataset: concat_vertical(frames) for dataset, frames in frames_by_dataset.items()}


def process_team_player_on_off(
    details_files: List[DashboardFile],
    summary_files: List[DashboardFile],
    season: str,
    season_type: str,
) -> Dict[int, pd.DataFrame]:
    details_by_team: Dict[Tuple[int, int], DashboardFile] = {}
    summary_by_team: Dict[Tuple[int, int], DashboardFile] = {}
    for dashboard_file in details_files:
        if dashboard_file.team_id is None:
            continue
        details_by_team[(dashboard_file.team_id, dashboard_file.dataset)] = dashboard_file
    for dashboard_file in summary_files:
        if dashboard_file.team_id is None:
            continue
        summary_by_team[(dashboard_file.team_id, dashboard_file.dataset)] = dashboard_file
    datasets = {1, 2}
    frames_by_dataset: Dict[int, List[pd.DataFrame]] = {dataset: [] for dataset in datasets}
    keys = sorted(set(details_by_team) | set(summary_by_team))
    for (team_id, dataset) in keys:
        if dataset not in datasets:
            continue
        details_file = details_by_team.get((team_id, dataset))
        summary_file = summary_by_team.get((team_id, dataset))
        frames: List[pd.DataFrame] = []
        suffixes: List[str] = []
        if details_file is not None:
            df_det = read_dashboard_frame(details_file, season, season_type)
            if df_det is not None:
                frames.append(df_det)
                suffixes.append("__det")
        if summary_file is not None:
            df_sum = read_dashboard_frame(summary_file, season, season_type)
            if df_sum is not None:
                frames.append(df_sum)
                suffixes.append("__sum")
        if not frames:
            continue
        description = f"team_player_on_off TEAM_ID {team_id} dataset {dataset}"
        merged = merge_horizontal(frames, suffixes, description)
        if merged is None:
            logging.warning("%s - no se pudo combinar details+summary", description)
            continue
        frames_by_dataset.setdefault(dataset, []).append(merged)
    return {dataset: concat_vertical(frames) for dataset, frames in frames_by_dataset.items()}


def ensure_output_dir(base_dir: Path, season: str) -> Path:
    output_dir = base_dir / OUTPUT_SUBDIR / season / "dashboards"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_dataframe(
    df: pd.DataFrame,
    output_path: Path,
    dry_run: bool,
) -> Tuple[int, int]:
    rows, cols = len(df), len(df.columns)
    if dry_run:
        logging.info("Dry-run: se omite escritura de %s", output_path)
        return rows, cols
    df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    logging.info("Archivo escrito: %s (filas: %d, columnas: %d)", output_path, rows, cols)
    return rows, cols


def process_dashboards(args: argparse.Namespace) -> Dict[str, Tuple[int, int]]:
    base_dir = Path(args.base_dir)
    input_root = base_dir / RAW_SUBDIR
    if not input_root.exists():
        raise FileNotFoundError(f"No existe el directorio de entrada: {input_root}")
    logging.info(
        "Procesando dashboards para la temporada %s (%s)",
        args.season,
        args.season_type,
    )
    output_dir = ensure_output_dir(base_dir, args.season)
    summary: Dict[str, Tuple[int, int]] = {}

    def get_family_dir(name: str) -> Path:
        return input_root / name / args.season / args.season_type

    families = [
        "team_dash_lineups",
        "team_dash_pt_pass",
        "team_dash_pt_reb",
        "team_dash_pt_shots",
        "team_dashboard_by_general_splits",
        "team_dashboard_by_shooting_splits",
        "team_player_dashboard",
        "team_player_on_off_details",
        "team_player_on_off_summary",
    ]

    discovered: Dict[str, List[DashboardFile]] = {}
    for family in families:
        family_dir = get_family_dir(family)
        if not family_dir.exists():
            logging.warning("Directorio %s inexistente", family_dir)
            discovered[family] = []
            continue
        discovered[family] = discover_dashboard_files(family_dir)

    # Lineups dataset 1
    lineups_df = process_team_dash_lineups(
        discovered.get("team_dash_lineups", []), args.season, args.season_type
    )
    if not lineups_df.empty:
        output = output_dir / "team_dash_lineups__dataset_1.parquet"
        summary[output.name] = write_dataframe(lineups_df, output, args.dry_run)
    else:
        logging.warning("team_dash_lineups - sin datos para escribir")

    # PT Pass datasets 0 y 1
    pt_pass_df = process_team_dash_pt_pass(
        discovered.get("team_dash_pt_pass", []), args.season, args.season_type
    )
    if not pt_pass_df.empty:
        output = output_dir / "team_dash_pt_pass__dataset_0_1__WIDE_byTEAM.parquet"
        summary[output.name] = write_dataframe(pt_pass_df, output, args.dry_run)
    else:
        logging.warning("team_dash_pt_pass - sin datos combinados")

    # PT Reb datasets 1..4
    pt_reb = process_team_dash_pt_reb(
        discovered.get("team_dash_pt_reb", []), args.season, args.season_type
    )
    for dataset in range(1, 5):
        df = pt_reb.get(dataset, pd.DataFrame())
        if df.empty:
            logging.warning("team_dash_pt_reb dataset %d sin datos", dataset)
            continue
        output = output_dir / f"team_dash_pt_reb__dataset_{dataset}.parquet"
        summary[output.name] = write_dataframe(df, output, args.dry_run)

    # PT Shots datasets 0..5
    pt_shots = process_team_dash_pt_shots(
        discovered.get("team_dash_pt_shots", []), args.season, args.season_type
    )
    for dataset in range(6):
        df = pt_shots.get(dataset, pd.DataFrame())
        if df.empty:
            logging.warning("team_dash_pt_shots dataset %d sin datos", dataset)
            continue
        output = output_dir / f"team_dash_pt_shots__dataset_{dataset}.parquet"
        summary[output.name] = write_dataframe(df, output, args.dry_run)

    # General splits dataset 0
    general_splits = process_simple_stack(
        discovered.get("team_dashboard_by_general_splits", []),
        [0],
        args.season,
        args.season_type,
    )
    df_general = general_splits.get(0, pd.DataFrame())
    if not df_general.empty:
        output = output_dir / "team_dashboard_by_general_splits__dataset_0.parquet"
        summary[output.name] = write_dataframe(df_general, output, args.dry_run)
    else:
        logging.warning("team_dashboard_by_general_splits dataset 0 sin datos")

    # Shooting splits datasets 1..6
    shooting_splits = process_simple_stack(
        discovered.get("team_dashboard_by_shooting_splits", []),
        range(1, 7),
        args.season,
        args.season_type,
    )
    for dataset in range(1, 7):
        df = shooting_splits.get(dataset, pd.DataFrame())
        if df.empty:
            logging.warning("team_dashboard_by_shooting_splits dataset %d sin datos", dataset)
            continue
        output = output_dir / f"team_dashboard_by_shooting_splits__dataset_{dataset}.parquet"
        summary[output.name] = write_dataframe(df, output, args.dry_run)

    # Team player dashboard dataset 1
    player_dashboard = process_simple_stack(
        discovered.get("team_player_dashboard", []),
        [1],
        args.season,
        args.season_type,
    )
    df_player_dash = player_dashboard.get(1, pd.DataFrame())
    if not df_player_dash.empty:
        output = output_dir / "team_player_dashboard__dataset_1.parquet"
        summary[output.name] = write_dataframe(df_player_dash, output, args.dry_run)
    else:
        logging.warning("team_player_dashboard dataset 1 sin datos")

    # Player on/off details + summary datasets 1 y 2
    on_off = process_team_player_on_off(
        discovered.get("team_player_on_off_details", []),
        discovered.get("team_player_on_off_summary", []),
        args.season,
        args.season_type,
    )
    for dataset in (1, 2):
        df = on_off.get(dataset, pd.DataFrame())
        if df.empty:
            logging.warning("team_player_on_off dataset %d sin datos combinados", dataset)
            continue
        output = output_dir / (
            f"team_player_on_off__dataset_{dataset}__DETAILS_PLUS_SUMMARY.parquet"
        )
        summary[output.name] = write_dataframe(df, output, args.dry_run)

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    logging.info(
        "Parámetros - base_dir: %s, season: %s, season_type: %s, workers: %s, dry_run: %s",
        args.base_dir,
        args.season,
        args.season_type,
        args.workers,
        args.dry_run,
    )
    try:
        summary = process_dashboards(args)
    except Exception as exc:  # pragma: no cover - robustez
        logging.exception("Error procesando dashboards: %s", exc)
        return
    if not summary:
        logging.warning("No se generaron archivos de salida")
        return
    logging.info("Resumen de archivos generados:")
    for filename, (rows, cols) in summary.items():
        logging.info(" - %s -> filas: %d, columnas: %d", filename, rows, cols)


if __name__ == "__main__":
    main()

