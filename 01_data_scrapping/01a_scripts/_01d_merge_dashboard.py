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

TEAM_NAME_MAP: Dict[int, str] = {
    1610612737: "Atlanta Hawks",
    1610612738: "Boston Celtics",
    1610612739: "Cleveland Cavaliers",
    1610612740: "New Orleans Pelicans",
    1610612741: "Chicago Bulls",
    1610612742: "Dallas Mavericks",
    1610612743: "Denver Nuggets",
    1610612744: "Golden State Warriors",
    1610612745: "Houston Rockets",
    1610612746: "LA Clippers",
    1610612747: "Los Angeles Lakers",
    1610612748: "Miami Heat",
    1610612749: "Milwaukee Bucks",
    1610612750: "Minnesota Timberwolves",
    1610612751: "Brooklyn Nets",
    1610612752: "New York Knicks",
    1610612753: "Orlando Magic",
    1610612754: "Indiana Pacers",
    1610612755: "Philadelphia 76ers",
    1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers",
    1610612758: "Sacramento Kings",
    1610612759: "San Antonio Spurs",
    1610612760: "Oklahoma City Thunder",
    1610612761: "Toronto Raptors",
    1610612762: "Utah Jazz",
    1610612763: "Memphis Grizzlies",
    1610612764: "Washington Wizards",
    1610612765: "Detroit Pistons",
    1610612766: "Charlotte Hornets",
}

KEY_COLUMN_ORDER = [
    "TEAM_ID",
    "TEAM_NAME",
    "PLAYER_ID",
    "GROUP_SET",
    "GROUP_VALUE",
    "GROUP_NAME",
]

SORT_KEY_PRIORITY = [
    "TEAM_ID",
    "TEAM_NAME",
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

PT_PASS_KEY_SETS: Sequence[Sequence[str]] = (
    ("TEAM_ID", "PLAYER_ID", "GROUP_SET", "GROUP_VALUE"),
    ("TEAM_ID", "PLAYER_ID", "GROUP_SET", "GROUP_NAME"),
    ("TEAM_ID", "PLAYER_ID", "GROUP_SET"),
    ("TEAM_ID", "PLAYER_ID", "GROUP_VALUE"),
    ("TEAM_ID", "PLAYER_ID"),
    ("TEAM_ID", "GROUP_SET", "GROUP_VALUE", "GROUP_NAME"),
    ("TEAM_ID", "GROUP_VALUE", "GROUP_NAME"),
    ("TEAM_ID",),
)

ON_OFF_KEY_SETS: Sequence[Sequence[str]] = (
    ("TEAM_ID", "PLAYER_ID", "GROUP_SET", "GROUP_VALUE", "GROUP_NAME"),
    ("TEAM_ID", "PLAYER_ID", "GROUP_SET", "GROUP_VALUE"),
    ("TEAM_ID", "PLAYER_ID", "GROUP_NAME"),
    ("TEAM_ID", "PLAYER_ID", "GROUP_SET"),
    ("TEAM_ID", "PLAYER_ID"),
    ("TEAM_ID", "GROUP_SET", "GROUP_VALUE", "GROUP_NAME"),
    ("TEAM_ID", "GROUP_VALUE", "GROUP_NAME"),
    ("TEAM_ID",),
)


@dataclass
class DashboardFile:
    path: Path
    team_id: Optional[int]
    dataset: int


@dataclass
class FrameBundle:
    data: pd.DataFrame
    template: List[str]


@dataclass
class MergeOutcome:
    frame: pd.DataFrame
    rename_map: Dict[str, str]
    new_columns: List[str]
    deduplicated: List[str]
    keys: Optional[List[str]]
    fallback_used: bool


class ColumnTracker:
    def __init__(self) -> None:
        self.template: List[str] = []
        self.additional: List[str] = []

    def set_template(self, columns: Sequence[str]) -> None:
        if not self.template and columns:
            self.template = list(columns)

    def apply_renames(self, rename_map: Dict[str, str]) -> None:
        if not rename_map:
            return
        self.template = [rename_map.get(col, col) for col in self.template]
        self.additional = [rename_map.get(col, col) for col in self.additional]

    def register_additional(self, columns: Sequence[str]) -> None:
        for col in columns:
            if col not in self.template and col not in self.additional:
                self.additional.append(col)

    def ordered_columns(self, df: pd.DataFrame) -> List[str]:
        keys = [col for col in KEY_COLUMN_ORDER if col in df.columns]
        template_cols = [
            col for col in self.template if col in df.columns and col not in keys
        ]
        additional_cols = [
            col
            for col in self.additional
            if col in df.columns and col not in keys and col not in template_cols
        ]
        remaining = [
            col
            for col in df.columns
            if col not in keys and col not in template_cols and col not in additional_cols
        ]
        return keys + template_cols + additional_cols + remaining




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
) -> Optional[FrameBundle]:
    try:
        df = pd.read_parquet(dashboard_file.path)
    except Exception as exc:  # pragma: no cover - robustez en runtime
        logging.warning("No se pudo leer %s: %s", dashboard_file.path, exc)
        return None
    template = list(df.columns)
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
    return FrameBundle(data=df, template=template)

def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    sort_keys = [col for col in SORT_KEY_PRIORITY if col in df.columns]
    if not sort_keys:
        return df
    return df.sort_values(sort_keys).reset_index(drop=True)


def normalize_identifier_columns(df: pd.DataFrame, context: str) -> pd.DataFrame:
    df = df.copy()
    if "TEAM_ID" in df.columns:
        team_id_series = pd.to_numeric(df["TEAM_ID"], errors="coerce").astype("Int64")
        if team_id_series.isna().any():
            missing = sorted(team_id_series[team_id_series.isna()].index.tolist())
            logging.warning("%s - TEAM_ID con valores nulos en filas %s", context, missing)
        else:
            team_id_series = team_id_series.astype("int64")
        df["TEAM_ID"] = team_id_series
    if "PLAYER_ID" in df.columns:
        df["PLAYER_ID"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce").astype("Int64")
    for column in ("GROUP_SET", "GROUP_VALUE", "GROUP_NAME"):
        if column in df.columns:
            series = df[column]
            if pd.api.types.is_string_dtype(series) or series.dtype == object:
                df[column] = series.astype("string").str.strip()
    return df


def ensure_team_name(df: pd.DataFrame, context: str) -> pd.DataFrame:
    if "TEAM_ID" not in df.columns:
        logging.warning("%s - no se encontró TEAM_ID para mapear TEAM_NAME", context)
        return df
    team_ids = pd.to_numeric(df["TEAM_ID"], errors="coerce")
    mapped = team_ids.map(TEAM_NAME_MAP)
    missing_ids = sorted(set(team_ids[mapped.isna()].dropna().astype(int).tolist()))
    if missing_ids:
        logging.warning("%s - TEAM_ID sin mapeo de nombre: %s", context, missing_ids)
    team_name_series = pd.Series(mapped, index=df.index, dtype="string")
    df = df.copy()
    df["TEAM_NAME"] = team_name_series
    if "TEAM_ID" in df.columns:
        position = list(df.columns).index("TEAM_ID") + 1
        team_name = df.pop("TEAM_NAME")
        df.insert(position, "TEAM_NAME", team_name)
    return df


def finalize_dataframe(
    df: pd.DataFrame,
    tracker: ColumnTracker,
    context: str,
) -> pd.DataFrame:
    df = normalize_identifier_columns(df, context)
    df = ensure_team_name(df, context)
    ordered_cols = tracker.ordered_columns(df)
    df = df.reindex(columns=ordered_cols)
    df = sort_dataframe(df)
    return df


def concat_bundles(
    bundles: Sequence[FrameBundle],
    tracker: ColumnTracker,
    context: str,
) -> pd.DataFrame:
    if not bundles:
        return pd.DataFrame()
    for bundle in bundles:
        tracker.set_template(bundle.template)
    frames = [bundle.data for bundle in bundles]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return finalize_dataframe(combined, tracker, context)


def rename_frames(frames: Sequence[pd.DataFrame], rename_map: Dict[str, str]) -> None:
    if not rename_map:
        return
    for frame in frames:
        frame.rename(columns=rename_map, inplace=True)


def determine_join_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    preferred_sets: Sequence[Sequence[str]],
) -> Optional[List[str]]:
    for candidate in preferred_sets:
        if not candidate:
            continue
        if not all(col in left.columns and col in right.columns for col in candidate):
            continue
        if left.duplicated(candidate).any() or right.duplicated(candidate).any():
            continue
        left_keys = left[candidate].copy()
        right_keys = right[candidate].copy()
        if left_keys.isnull().any(axis=None) or right_keys.isnull().any(axis=None):
            continue
        left_unique = left_keys.drop_duplicates().sort_values(candidate).reset_index(drop=True)
        right_unique = right_keys.drop_duplicates().sort_values(candidate).reset_index(drop=True)
        if len(left_unique) != len(left) or len(right_unique) != len(right):
            continue
        if left_unique.equals(right_unique):
            return list(candidate)
    return None


def prepare_for_horizontal_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: Sequence[str],
    left_suffix: str,
    right_suffix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str], List[str], List[str]]:
    left = left.copy()
    right = right.copy()
    rename_map: Dict[str, str] = {}
    deduplicated: List[str] = []
    new_columns: List[str] = []
    for column in list(right.columns):
        if column in keys:
            continue
        if column in left.columns:
            if left[column].equals(right[column]):
                right.drop(columns=[column], inplace=True)
                deduplicated.append(column)
            else:
                new_left = f"{column}{left_suffix}"
                new_right = f"{column}{right_suffix}"
                left.rename(columns={column: new_left}, inplace=True)
                right.rename(columns={column: new_right}, inplace=True)
                rename_map[column] = new_left
                new_columns.append(new_right)
        else:
            new_columns.append(column)
    return left, right, rename_map, deduplicated, new_columns


def merge_bundles_horizontal(
    left: FrameBundle,
    right: FrameBundle,
    left_suffix: str,
    right_suffix: str,
    preferred_keys: Sequence[Sequence[str]],
    description: str,
) -> Optional[MergeOutcome]:
    keys = determine_join_keys(left.data, right.data, preferred_keys)
    if keys is not None:
        left_prepared, right_prepared, rename_map, deduplicated, new_columns = prepare_for_horizontal_merge(
            left.data, right.data, keys, left_suffix, right_suffix
        )
        left_indexed = left_prepared.set_index(keys)
        right_indexed = right_prepared.set_index(keys)
        merged = left_indexed.join(right_indexed, how="inner")
        merged.reset_index(inplace=True)
        return MergeOutcome(
            frame=merged,
            rename_map=rename_map,
            new_columns=new_columns,
            deduplicated=deduplicated,
            keys=list(keys),
            fallback_used=False,
        )
    if len(left.data) != len(right.data):
        logging.warning(
            "%s - longitudes distintas sin llaves fiables: %s",
            description,
            {len(left.data), len(right.data)},
        )
        return None
    left_prepared, right_prepared, rename_map, deduplicated, new_columns = prepare_for_horizontal_merge(
        left.data, right.data, [], left_suffix, right_suffix
    )
    merged = pd.concat(
        [left_prepared.reset_index(drop=True), right_prepared.reset_index(drop=True)],
        axis=1,
    )
    logging.warning("%s - sin llaves fiables, se usó alineamiento por índice", description)
    return MergeOutcome(
        frame=merged,
        rename_map=rename_map,
        new_columns=new_columns,
        deduplicated=deduplicated,
        keys=None,
        fallback_used=True,
    )




def process_team_dash_lineups(
    files: List[DashboardFile],
    season: str,
    season_type: str,
) -> pd.DataFrame:
    bundles: List[FrameBundle] = []
    tracker = ColumnTracker()
    for dashboard_file in files:
        if dashboard_file.dataset != 1:
            continue
        bundle = read_dashboard_frame(dashboard_file, season, season_type)
        if bundle is None:
            continue
        bundles.append(bundle)
    return concat_bundles(bundles, tracker, "team_dash_lineups__dataset_1")


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
    tracker = ColumnTracker()
    for team_id, datasets in sorted(files_by_team.items()):
        dataset_bundles: Dict[int, FrameBundle] = {}
        for dataset in (0, 1):
            dashboard_file = datasets.get(dataset)
            if dashboard_file is None:
                continue
            bundle = read_dashboard_frame(dashboard_file, season, season_type)
            if bundle is None:
                continue
            dataset_bundles[dataset] = bundle
        if not dataset_bundles:
            logging.warning("team_dash_pt_pass - TEAM_ID %s sin datasets 0/1", team_id)
            continue
        base_dataset = 0 if 0 in dataset_bundles else min(dataset_bundles.keys())
        base_bundle = dataset_bundles[base_dataset]
        current_df = base_bundle.data.copy()
        current_template = list(base_bundle.template)
        tracker.set_template(current_template)
        description = f"team_dash_pt_pass TEAM_ID {team_id}"
        for dataset, suffix in ((0, "__d0"), (1, "__d1")):
            if dataset == base_dataset or dataset not in dataset_bundles:
                continue
            outcome = merge_bundles_horizontal(
                FrameBundle(data=current_df, template=current_template),
                dataset_bundles[dataset],
                "__d0" if base_dataset == 0 else "__d1",
                suffix,
                PT_PASS_KEY_SETS,
                f"{description} dataset {dataset}",
            )
            if outcome is None:
                logging.warning("%s dataset %d - no se pudo combinar", description, dataset)
                continue
            if outcome.rename_map:
                tracker.apply_renames(outcome.rename_map)
                rename_frames(merged_frames, outcome.rename_map)
                current_template = [outcome.rename_map.get(col, col) for col in current_template]
                logging.info(
                    "%s dataset %d - columnas renombradas por colisión: %s",
                    description,
                    dataset,
                    [f"{old}->{new}" for old, new in outcome.rename_map.items()],
                )
            if outcome.deduplicated:
                logging.info(
                    "%s dataset %d - columnas duplicadas eliminadas: %s",
                    description,
                    dataset,
                    outcome.deduplicated,
                )
            if outcome.new_columns:
                tracker.register_additional(outcome.new_columns)
                logging.info(
                    "%s dataset %d - columnas añadidas: %s",
                    description,
                    dataset,
                    outcome.new_columns,
                )
            if outcome.keys:
                logging.info("%s dataset %d - llaves de unión: %s", description, dataset, outcome.keys)
            current_df = outcome.frame
        current_df["TEAM_ID"] = pd.Series([team_id] * len(current_df), dtype="int64")
        merged_frames.append(current_df)
    if not merged_frames:
        return pd.DataFrame()
    combined = pd.concat(merged_frames, ignore_index=True, sort=False)
    return finalize_dataframe(combined, tracker, "team_dash_pt_pass")


def process_team_dash_pt_reb(
    files: List[DashboardFile],
    season: str,
    season_type: str,
) -> Dict[int, pd.DataFrame]:
    bundles_by_dataset: Dict[int, List[FrameBundle]] = {i: [] for i in range(1, 5)}
    trackers: Dict[int, ColumnTracker] = {i: ColumnTracker() for i in range(1, 5)}
    for dashboard_file in files:
        if dashboard_file.dataset not in bundles_by_dataset:
            continue
        bundle = read_dashboard_frame(dashboard_file, season, season_type)
        if bundle is not None:
            bundles_by_dataset[dashboard_file.dataset].append(bundle)
    output: Dict[int, pd.DataFrame] = {}
    for dataset, bundles in bundles_by_dataset.items():
        tracker = trackers[dataset]
        df = concat_bundles(bundles, tracker, f"team_dash_pt_reb__dataset_{dataset}")
        output[dataset] = df
    return output


def process_team_dash_pt_shots(
    files: List[DashboardFile],
    season: str,
    season_type: str,
) -> Dict[int, pd.DataFrame]:
    bundles_by_dataset: Dict[int, List[FrameBundle]] = {i: [] for i in range(6)}
    trackers: Dict[int, ColumnTracker] = {i: ColumnTracker() for i in range(6)}
    for dashboard_file in files:
        if dashboard_file.dataset not in bundles_by_dataset:
            continue
        bundle = read_dashboard_frame(dashboard_file, season, season_type)
        if bundle is not None:
            bundles_by_dataset[dashboard_file.dataset].append(bundle)
    output: Dict[int, pd.DataFrame] = {}
    for dataset, bundles in bundles_by_dataset.items():
        tracker = trackers[dataset]
        df = concat_bundles(bundles, tracker, f"team_dash_pt_shots__dataset_{dataset}")
        output[dataset] = df
    return output


def process_simple_stack(
    files: List[DashboardFile],
    allowed_datasets: Iterable[int],
    season: str,
    season_type: str,
) -> Dict[int, pd.DataFrame]:
    allowed = set(allowed_datasets)
    bundles_by_dataset: Dict[int, List[FrameBundle]] = {dataset: [] for dataset in allowed}
    trackers: Dict[int, ColumnTracker] = {dataset: ColumnTracker() for dataset in allowed}
    for dashboard_file in files:
        if dashboard_file.dataset not in allowed:
            continue
        bundle = read_dashboard_frame(dashboard_file, season, season_type)
        if bundle is not None:
            bundles_by_dataset[dashboard_file.dataset].append(bundle)
    output: Dict[int, pd.DataFrame] = {}
    for dataset, bundles in bundles_by_dataset.items():
        tracker = trackers[dataset]
        df = concat_bundles(bundles, tracker, f"simple_stack_dataset_{dataset}")
        output[dataset] = df
    return output


def process_team_player_on_off(
    details_files: List[DashboardFile],
    summary_files: List[DashboardFile],
    season: str,
    season_type: str,
) -> Dict[int, pd.DataFrame]:
    details_by_team: Dict[Tuple[int, int], FrameBundle] = {}
    summary_by_team: Dict[Tuple[int, int], FrameBundle] = {}
    for dashboard_file in details_files:
        if dashboard_file.team_id is None:
            continue
        bundle = read_dashboard_frame(dashboard_file, season, season_type)
        if bundle is not None:
            details_by_team[(dashboard_file.team_id, dashboard_file.dataset)] = bundle
    for dashboard_file in summary_files:
        if dashboard_file.team_id is None:
            continue
        bundle = read_dashboard_frame(dashboard_file, season, season_type)
        if bundle is not None:
            summary_by_team[(dashboard_file.team_id, dashboard_file.dataset)] = bundle
    datasets = {1, 2}
    trackers: Dict[int, ColumnTracker] = {dataset: ColumnTracker() for dataset in datasets}
    frames_by_dataset: Dict[int, List[pd.DataFrame]] = {dataset: [] for dataset in datasets}
    keys = sorted(set(details_by_team) | set(summary_by_team))
    for (team_id, dataset) in keys:
        if dataset not in datasets:
            continue
        tracker = trackers[dataset]
        description = f"team_player_on_off TEAM_ID {team_id} dataset {dataset}"
        base_bundle = details_by_team.get((team_id, dataset))
        other_bundle = summary_by_team.get((team_id, dataset))
        base_suffix = "__det"
        other_suffix = "__sum"
        if base_bundle is None and other_bundle is None:
            continue
        if base_bundle is None:
            base_bundle, other_bundle = other_bundle, None
            base_suffix, other_suffix = "__sum", "__det"
        current_df = base_bundle.data.copy()
        current_template = list(base_bundle.template)
        tracker.set_template(current_template)
        if other_bundle is not None:
            outcome = merge_bundles_horizontal(
                FrameBundle(data=current_df, template=current_template),
                other_bundle,
                base_suffix,
                other_suffix,
                ON_OFF_KEY_SETS,
                description,
            )
            if outcome is None:
                logging.warning("%s - no se pudo combinar details+summary", description)
            else:
                if outcome.rename_map:
                    tracker.apply_renames(outcome.rename_map)
                    rename_frames(frames_by_dataset[dataset], outcome.rename_map)
                    current_template = [outcome.rename_map.get(col, col) for col in current_template]
                    logging.info(
                        "%s - columnas renombradas por colisión: %s",
                        description,
                        [f"{old}->{new}" for old, new in outcome.rename_map.items()],
                    )
                if outcome.deduplicated:
                    logging.info(
                        "%s - columnas duplicadas eliminadas: %s",
                        description,
                        outcome.deduplicated,
                    )
                if outcome.new_columns:
                    tracker.register_additional(outcome.new_columns)
                    logging.info(
                        "%s - columnas añadidas: %s",
                        description,
                        outcome.new_columns,
                    )
                if outcome.keys:
                    logging.info("%s - llaves de unión: %s", description, outcome.keys)
                current_df = outcome.frame
        current_df["TEAM_ID"] = pd.Series([team_id] * len(current_df), dtype="int64")
        frames_by_dataset[dataset].append(current_df)
    output: Dict[int, pd.DataFrame] = {}
    for dataset, frames in frames_by_dataset.items():
        tracker = trackers[dataset]
        if not frames:
            output[dataset] = pd.DataFrame()
            continue
        combined = pd.concat(frames, ignore_index=True, sort=False)
        output[dataset] = finalize_dataframe(
            combined,
            tracker,
            f"team_player_on_off__dataset_{dataset}",
        )
    return output


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
        output = output_dir / "team_dash_pt_pass.parquet"
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
        output = output_dir / f"team_player_on_off__dataset_{dataset}.parquet"
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

