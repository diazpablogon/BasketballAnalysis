"""Combine boxscore parquet endpoints into a single merged table."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd


def discover_parquet_files(boxscore_dir: Path, season: str) -> Dict[str, List[Path]]:
    """Return parquet files grouped by endpoint for the given season."""
    files_by_endpoint: Dict[str, List[Path]] = defaultdict(list)
    for endpoint_dir in sorted(p for p in boxscore_dir.iterdir() if p.is_dir()):
        season_dir = endpoint_dir / season
        if season_dir.exists():
            files_by_endpoint[endpoint_dir.name].extend(sorted(season_dir.glob("*.parquet")))
    return files_by_endpoint


def merge_dataframe(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Merge ``other`` into ``base`` on the shared key columns."""
    keys = [k for k in ("GAME_ID", "TEAM_ID", "PLAYER_ID") if k in base.columns and k in other.columns]
    if not keys:
        return base
    other = other.drop_duplicates(subset=keys)
    columns = list(dict.fromkeys(keys + [c for c in other.columns if c not in base.columns]))
    join_type = "outer" if "PLAYER_ID" in keys else "left"
    merged = base.merge(other[columns], on=keys, how=join_type)
    key_subset = [k for k in ("GAME_ID", "TEAM_ID", "PLAYER_ID") if k in merged.columns]
    return merged.drop_duplicates(subset=key_subset) if key_subset else merged


def load_parquet(path: Path) -> pd.DataFrame:
    """Load parquet file and drop duplicates over the available keys."""
    df = pd.read_parquet(path)
    if "GAME_ID" not in df.columns:
        return pd.DataFrame()
    keys = [k for k in ("GAME_ID", "TEAM_ID", "PLAYER_ID") if k in df.columns]
    return df.drop_duplicates(subset=keys) if keys else df


def process_boxscores(base_dir: str, season: str) -> Path:
    """Process boxscore parquet endpoints and return the merged parquet path."""
    repo_dir = Path(base_dir).expanduser().resolve()
    boxscore_dir = repo_dir / "00_data" / "00a_raw" / "boxscore"
    if not boxscore_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {boxscore_dir}")

    files_by_endpoint = discover_parquet_files(boxscore_dir, season)
    player_frames: List[pd.DataFrame] = []
    team_frames: List[pd.DataFrame] = []
    game_frames: List[pd.DataFrame] = []
    for paths in files_by_endpoint.values():
        for path in paths:
            df = load_parquet(path)
            if df.empty:
                continue
            cols = set(df.columns)
            if "PLAYER_ID" in cols:
                player_frames.append(df)
            elif "TEAM_ID" in cols:
                team_frames.append(df)
            else:
                game_frames.append(df)

    if not player_frames:
        raise ValueError("No player-level boxscore data found; cannot build merged dataset")

    base = player_frames[0]
    for frame in player_frames[1:] + team_frames + game_frames:
        base = merge_dataframe(base, frame)
    if "PLAYER_ID" in base.columns:
        base = base[base["PLAYER_ID"].notna()]

    output_path = repo_dir / "00_data" / "00c_final" / "boxscore_merged.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(output_path, index=False)

    print(f"💾 Guardado {output_path}")
    print(f"✅ Combinado {len(base):,} filas y {len(base.columns):,} columnas → {output_path.name}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine boxscore parquet endpoints into a merged file.")
    default_base = Path(__file__).resolve().parents[2]
    parser.add_argument("--base_dir", default=default_base, type=Path)
    parser.add_argument("--season", default="2024-25")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_boxscores(base_dir=str(args.base_dir), season=args.season)


if __name__ == "__main__":
    main()
