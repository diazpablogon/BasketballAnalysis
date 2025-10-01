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


def load_and_annotate_dataset(dataset_id: int, files: Iterable[Path]) -> pd.DataFrame:
    """Load parquet files for ``dataset_id`` and add dataset metadata columns."""
    frames: List[pd.DataFrame] = []
    config = SPLIT_CONFIG[dataset_id]
    split_type = config["split_type"]
    split_value_def = config["split_value"]

    for path in files:
        df = pd.read_parquet(path)
        df = df.copy()
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

    dataframes = [
        load_and_annotate_dataset(dataset_id, files)
        for dataset_id, files in sorted(dataset_files.items())
    ]

    combined = pd.concat(dataframes, ignore_index=True, sort=False)

    if "TEAM_ID" not in combined.columns:
        raise KeyError("TEAM_ID column not found in combined dataframe")

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
