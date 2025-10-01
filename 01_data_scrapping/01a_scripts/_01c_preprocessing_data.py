"""Preprocess team dashboard general splits into a single league-wide parquet file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

SPLIT_CONFIG = {
    0: {"split_type": "overall", "split_value": ("literal", "overall")},
    1: {"split_type": "team_game_location", "split_value": ("column", "TEAM_GAME_LOCATION")},
    2: {"split_type": "game_result", "split_value": ("column", "GAME_RESULT")},
    3: {"split_type": "season_month_name", "split_value": ("column", "SEASON_MONTH_NAME")},
    4: {"split_type": "season_segment", "split_value": ("column", "SEASON_SEGMENT")},
    5: {"split_type": "team_days_rest_range", "split_value": ("column", "TEAM_DAYS_REST_RANGE")},
}


def discover_dataset_files(input_dir: Path) -> Dict[int, List[Path]]:
    """Discover parquet files for each dataset index within ``input_dir``."""
    dataset_files: Dict[int, List[Path]] = {}
    for dataset_id in SPLIT_CONFIG:
        pattern = f"team_dashboard_by_general_splits__*__dataset_{dataset_id}.parquet"
        files = sorted(input_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No files found for dataset_{dataset_id} in {input_dir}. Expected pattern: {pattern}"
            )
        dataset_files[dataset_id] = files
    return dataset_files


def load_and_annotate_dataset(
    dataset_id: int,
    files: Iterable[Path],
    team_name_map: Dict[int, str] | None = None,
) -> pd.DataFrame:
    """Load parquet files for ``dataset_id`` and add dataset metadata columns."""
    frames: List[pd.DataFrame] = []
    config = SPLIT_CONFIG[dataset_id]
    split_type = config["split_type"]
    split_value_def = config["split_value"]

    for path in files:
        df = pd.read_parquet(path)
        df = df.copy()

        team_id_column = next((col for col in df.columns if col.lower() == "team_id"), None)
        team_id_value: int | None = None
        if team_id_column is not None:
            if team_id_column != "TEAM_ID":
                df = df.rename(columns={team_id_column: "TEAM_ID"})
            team_id_series = df["TEAM_ID"]
            if team_id_series.nunique() == 1:
                try:
                    team_id_value = int(team_id_series.iloc[0])
                except (TypeError, ValueError):
                    team_id_value = None
        else:
            parts = path.stem.split("__")
            if len(parts) < 3:
                raise ValueError(
                    "Unable to extract TEAM_ID from filename '"
                    f"{path.name}'. Expected pattern "
                    "'team_dashboard_by_general_splits__<TEAMID>__dataset_<id>.parquet'"
                )
            try:
                team_id_value = int(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"Extracted TEAM_ID '{parts[1]}' from filename '{path.name}' is not an integer"
                ) from exc
            df["TEAM_ID"] = team_id_value

        if team_id_value is None:
            team_id_series = df["TEAM_ID"]
            if team_id_series.nunique() == 1:
                try:
                    team_id_value = int(team_id_series.iloc[0])
                except (TypeError, ValueError):
                    team_id_value = None

        team_name_column = next((col for col in df.columns if col.lower() == "team_name"), None)
        if team_name_column is not None:
            if team_name_column != "TEAM_NAME":
                df = df.rename(columns={team_name_column: "TEAM_NAME"})
        elif "TEAM_NAME" not in df.columns:
            df["TEAM_NAME"] = pd.NA

        if team_id_value is not None and team_name_map is not None:
            if "TEAM_NAME" in df.columns:
                non_null_names = df["TEAM_NAME"].dropna().unique()
                if len(non_null_names) == 1:
                    team_name_map.setdefault(team_id_value, str(non_null_names[0]))
                elif len(non_null_names) > 1:
                    # Prefer the first non-null value when multiple names appear unexpectedly.
                    team_name_map.setdefault(team_id_value, str(non_null_names[0]))
            mapped_name = team_name_map.get(team_id_value)
            if mapped_name is not None:
                df["TEAM_NAME"] = mapped_name

        df["source_dataset"] = dataset_id
        df["split_type"] = split_type

        kind, value = split_value_def
        if kind == "literal":
            df["split_value"] = value
        elif kind == "column":
            if value not in df.columns:
                raise KeyError(
                    f"Column '{value}' required for dataset_{dataset_id} not found in {path}"
                )
            df["split_value"] = df[value]
        else:
            raise ValueError(f"Unknown split value definition: {split_value_def}")

        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def preprocess(base_dir: Path, season: str, season_type: str) -> Path:
    """Run preprocessing and return the output parquet path."""
    input_dir = (
        base_dir
        / "00_data"
        / "00a_raw"
        / "team_dashboard"
        / "team_dashboard_by_general_splits"
        / season
        / season_type
    )
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    dataset_files = discover_dataset_files(input_dir)

    team_name_map: Dict[int, str] = {}
    dataframes = []
    for dataset_id, files in sorted(dataset_files.items()):
        df = load_and_annotate_dataset(dataset_id, files, team_name_map=team_name_map)
        dataframes.append(df)

    combined = pd.concat(dataframes, ignore_index=True, sort=False)

    if "TEAM_ID" not in combined.columns:
        raise KeyError("TEAM_ID column not found in combined dataframe")

    if "TEAM_NAME" not in combined.columns:
        combined["TEAM_NAME"] = pd.NA

    if combined["TEAM_NAME"].isna().any():
        if team_name_map:
            combined["TEAM_NAME"] = combined["TEAM_ID"].map(team_name_map).fillna(combined["TEAM_NAME"])
        else:
            name_lookup = (
                combined.loc[combined["TEAM_NAME"].notna(), ["TEAM_ID", "TEAM_NAME"]]
                .drop_duplicates(subset="TEAM_ID")
                .set_index("TEAM_ID")["TEAM_NAME"]
            )
            combined["TEAM_NAME"] = combined["TEAM_ID"].map(name_lookup).fillna(combined["TEAM_NAME"])

    output_path = (
        base_dir
        / "00_data"
        / "00b_intermediate"
        / "team_dashboard"
        / "general_splits"
        / season
        / season_type
        / "team_dashboard__general_splits.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    print(f"Total rows: {len(combined)}")
    unique_teams = combined["TEAM_ID"].nunique()
    print(f"Unique TEAM_ID count: {unique_teams}")

    split_counts = combined["split_type"].value_counts()
    print("Rows per split_type:")
    for dataset_id in SPLIT_CONFIG:
        stype = SPLIT_CONFIG[dataset_id]["split_type"]
        count = int(split_counts.get(stype, 0))
        print(f"  {stype}: {count}")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess team dashboard general splits parquet files into a single league-wide file."
        )
    )
    default_base = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--base_dir",
        default=default_base,
        type=Path,
        help="Base directory of the repository (default: repository root)",
    )
    parser.add_argument(
        "--season",
        default="2024-25",
        help="Season identifier (default: 2024-25)",
    )
    parser.add_argument(
        "--season_type",
        default="Regular Season",
        help="Season type (default: Regular Season)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.expanduser().resolve()
    preprocess(base_dir=base_dir, season=args.season, season_type=args.season_type)


if __name__ == "__main__":
    main()
