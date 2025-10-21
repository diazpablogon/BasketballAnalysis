#!/usr/bin/env python3
"""Generate player summary statistics from boxscore level data."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

DEFAULT_INPUT = Path(
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00c_final/2024-25/boxscores.parquet"
)
DEFAULT_OUTPUT = Path(
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00c_final/2024-25/player_summaries.parquet"
)

TOTAL_VOLUME_STATS = [
    "PTS",
    "REB",
    "OREB",
    "DREB",
    "AST",
    "STL",
    "BLK",
    "TO",
    "PF",
    "PFD",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
]
MEAN_VOLUME_STATS = ["MIN"] + TOTAL_VOLUME_STATS
EXPLICIT_RATIO_STATS = {
    "FG_PCT",
    "FG3_PCT",
    "FT_PCT",
    "EFG_PCT",
    "TS_PCT",
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "PACE",
    "PACE_PER40",
    "E_PACE",
    "USG_PCT",
    "E_USG_PCT",
    "TM_TOV_PCT",
    "REB_PCT",
    "OREB_PCT",
    "DREB_PCT",
    "PIE",
    "POSS",
    "PLUS_MINUS",
}
DISPERSION_METRICS = [
    "PTS",
    "REB",
    "AST",
    "TO",
    "MIN",
    "FG_PCT",
    "FG3_PCT",
    "FT_PCT",
    "TS_PCT",
    "USG_PCT",
    "PLUS_MINUS",
]
RANK_METRICS_HIGHER_BETTER = [
    "PTS_G",
    "REB_G",
    "AST_G",
    "STL_G",
    "BLK_G",
    "FG_PCT_G",
    "FG3_PCT_G",
    "FT_PCT_G",
    "EFG_PCT_G",
    "TS_PCT_G",
    "USG_PCT_G",
    "OFF_RATING_G",
    "PIE_G",
    "PLUS_MINUS_G",
]
RANK_METRICS_LOWER_BETTER = ["TO_G", "TM_TOV_PCT_G", "PF_G"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumen estadístico de jugadores por temporada")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT,
        help="Ruta del parquet con boxscores (por defecto se usa el merge final)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Ruta de salida para el parquet con los resúmenes",
    )
    parser.add_argument("--season", help="Temporada a filtrar (ej. 2024-25)")
    parser.add_argument(
        "--by-team-splits",
        action="store_true",
        help="Agrupa las estadísticas por jugador y equipo dentro de la temporada",
    )
    parser.add_argument(
        "--sort-by-rank",
        help="Ordenar jugadores dentro del grupo por la métrica de ranking indicada (ej. PTS_G_RANK)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribe el archivo de salida si ya existe",
    )
    return parser.parse_args()


def ensure_required_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Columnas obligatorias ausentes: {', '.join(missing)}")


def most_frequent(series: pd.Series) -> Optional[object]:
    non_null = series.dropna()
    if non_null.empty:
        return None
    counts = non_null.value_counts()
    return counts.sort_values(ascending=False).index[0]


def select_columns(
    df: pd.DataFrame,
    candidates: Iterable[str],
    *,
    context: str,
    require_numeric: bool = True,
) -> List[str]:
    selected: List[str] = []
    missing: List[str] = []
    non_numeric: List[str] = []
    for column in candidates:
        if column not in df.columns:
            missing.append(column)
            continue
        if require_numeric and not pd.api.types.is_numeric_dtype(df[column]):
            non_numeric.append(column)
            continue
        selected.append(column)
    if missing:
        logging.warning(
            "Columnas ausentes omitidas para %s: %s",
            context,
            ", ".join(sorted(missing)),
        )
    if non_numeric:
        logging.warning(
            "Columnas no numéricas omitidas para %s: %s",
            context,
            ", ".join(sorted(non_numeric)),
        )
    return selected


def discover_additional_ratios(df: pd.DataFrame, excluded: Iterable[str]) -> List[str]:
    excluded_set = set(excluded)
    discovered: List[str] = []
    for column in df.columns:
        if column in excluded_set:
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        if column.startswith("OPP_") or column.startswith("PCT_") or column.endswith("_PCT"):
            discovered.append(column)
    return sorted(discovered)


def compute_primary_team(df: pd.DataFrame) -> pd.DataFrame:
    if "TEAM_ID" not in df.columns:
        return pd.DataFrame()
    grouping_fields = ["SEASON", "PLAYER_ID", "TEAM_ID"]
    use_abbreviation = "TEAM_ABBREVIATION" in df.columns
    if use_abbreviation:
        grouping_fields.append("TEAM_ABBREVIATION")
    counts = (
        df.groupby(grouping_fields)
        .size()
        .reset_index(name="games")
        .sort_values(
            ["SEASON", "PLAYER_ID", "games"]
            + (["TEAM_ABBREVIATION"] if use_abbreviation else ["TEAM_ID"]),
            ascending=[True, True, False, True],
        )
    )
    primary = counts.drop_duplicates(subset=["SEASON", "PLAYER_ID"])
    primary = primary.set_index(["SEASON", "PLAYER_ID"])
    columns: List[pd.Series] = []
    if "TEAM_ID" in primary.columns:
        columns.append(primary["TEAM_ID"].rename("_PRIMARY_TEAM_ID"))
    if use_abbreviation and "TEAM_ABBREVIATION" in primary.columns:
        columns.append(primary["TEAM_ABBREVIATION"].rename("TEAM_ABBREVIATION"))
    if not columns:
        return pd.DataFrame(index=primary.index)
    return pd.concat(columns, axis=1)


def compute_dispersion(
    grouped: pd.core.groupby.generic.DataFrameGroupBy,
    metrics: List[str],
) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame(index=grouped.size().index)
    dispersion = grouped[metrics].agg(["std", "var", "min", "max"])
    renamed = {
        (metric, "std"): f"{metric}_STD"
        for metric in metrics
    }
    renamed.update({(metric, "var"): f"{metric}_VAR" for metric in metrics})
    renamed.update({(metric, "min"): f"{metric}_MIN" for metric in metrics})
    renamed.update({(metric, "max"): f"{metric}_MAX" for metric in metrics})
    dispersion.columns = [renamed[col] for col in dispersion.columns]  # type: ignore[index]
    return dispersion


def compute_summary(df: pd.DataFrame, by_team_splits: bool) -> pd.DataFrame:
    group_keys = ["SEASON", "PLAYER_ID"]
    if by_team_splits:
        group_keys.append("TEAM_ID")
    ensure_required_columns(df, group_keys)

    grouped = df.groupby(group_keys, dropna=False)
    gp = grouped.size().rename("GP")
    result = gp.to_frame()

    if "PLAYER_NAME" in df.columns:
        names = grouped["PLAYER_NAME"].agg(most_frequent).rename("PLAYER_NAME")
        result = result.join(names)

    if by_team_splits:
        if "TEAM_ABBREVIATION" in df.columns:
            team_abbr = grouped["TEAM_ABBREVIATION"].agg(most_frequent).rename("TEAM_ABBREVIATION")
            result = result.join(team_abbr)
        if "TEAM_ID" in df.columns:
            multi_team = (
                df.groupby(["SEASON", "PLAYER_ID"])["TEAM_ID"].nunique(dropna=True).gt(1).rename("MULTI_TEAM_SEASON")
            )
            result = result.join(multi_team, on=["SEASON", "PLAYER_ID"])
    else:
        primary_team = compute_primary_team(df)
        if not primary_team.empty:
            result = result.join(primary_team)

    volume_candidates = MEAN_VOLUME_STATS
    volume_numeric = select_columns(
        df,
        volume_candidates,
        context="totales y promedios de volumen",
        require_numeric=True,
    )
    totals_base = [col for col in volume_numeric if col in TOTAL_VOLUME_STATS]
    if totals_base:
        totals_raw = grouped[totals_base].sum(min_count=1)
        totals = totals_raw.rename(columns=lambda column: f"{column}_total")
        result = result.join(totals)
        per_game_volume = totals_raw.div(gp, axis=0)
        per_game_volume = per_game_volume.rename(columns=lambda column: f"{column}_G")
        result = result.join(per_game_volume)
    if "MIN" in volume_numeric:
        min_mean = grouped["MIN"].mean()
        result = result.join(min_mean.rename("MIN_G"))

    ratio_candidates = select_columns(
        df,
        sorted(EXPLICIT_RATIO_STATS),
        context="ratios por partido",
        require_numeric=True,
    )
    ratio_candidates.extend(discover_additional_ratios(df, volume_numeric + ratio_candidates))
    ratio_candidates = sorted(dict.fromkeys(ratio_candidates))
    if ratio_candidates:
        ratio_means = grouped[ratio_candidates].mean()
        ratio_means = ratio_means.rename(columns=lambda column: f"{column}_G")
        result = result.join(ratio_means)

    dispersion_metrics = select_columns(
        df,
        DISPERSION_METRICS,
        context="dispersión",
        require_numeric=True,
    )
    dispersion = compute_dispersion(grouped, dispersion_metrics)
    if not dispersion.empty:
        result = result.join(dispersion)

    result = result.reset_index()
    result["GP"] = result["GP"].astype("int64")
    return result


def add_rankings(df: pd.DataFrame, by_team_splits: bool) -> Tuple[pd.DataFrame, List[str]]:
    ranking_columns: List[str] = []
    grouping_fields = ["SEASON"]
    if by_team_splits and "TEAM_ID" in df.columns:
        grouping_fields.append("TEAM_ID")

    for metric in RANK_METRICS_HIGHER_BETTER:
        if metric not in df.columns:
            continue
        rank_col = f"{metric}_RANK"
        df[rank_col] = (
            df.groupby(grouping_fields)[metric]
            .rank(method="dense", ascending=False, na_option="keep")
        )
        ranking_columns.append(rank_col)

    for metric in RANK_METRICS_LOWER_BETTER:
        if metric not in df.columns:
            continue
        rank_col = f"{metric}_RANK"
        df[rank_col] = (
            df.groupby(grouping_fields)[metric]
            .rank(method="dense", ascending=True, na_option="keep")
        )
        ranking_columns.append(rank_col)

    std_columns = [column for column in df.columns if column.endswith("_STD")]
    for column in std_columns:
        rank_col = f"{column}_RANK"
        df[rank_col] = (
            df.groupby(grouping_fields)[column]
            .rank(method="dense", ascending=True, na_option="keep")
        )
        ranking_columns.append(rank_col)
    var_columns = [column for column in df.columns if column.endswith("_VAR")]
    for column in var_columns:
        rank_col = f"{column}_RANK"
        df[rank_col] = (
            df.groupby(grouping_fields)[column]
            .rank(method="dense", ascending=True, na_option="keep")
        )
        ranking_columns.append(rank_col)

    return df, ranking_columns


def apply_sorting(
    df: pd.DataFrame,
    by_team_splits: bool,
    sort_by_rank: Optional[str],
) -> pd.DataFrame:
    df = df.copy()
    sort_columns: List[str] = ["SEASON"]
    ascending: List[bool] = [True]

    if by_team_splits:
        if "TEAM_ABBREVIATION" in df.columns:
            sort_columns.append("TEAM_ABBREVIATION")
            ascending.append(True)
        elif "TEAM_ID" in df.columns:
            sort_columns.append("TEAM_ID")
            ascending.append(True)
    else:
        if "TEAM_ABBREVIATION" in df.columns:
            sort_columns.append("TEAM_ABBREVIATION")
            ascending.append(True)
        if "_PRIMARY_TEAM_ID" in df.columns:
            sort_columns.append("_PRIMARY_TEAM_ID")
            ascending.append(True)

    if sort_by_rank and sort_by_rank in df.columns:
        sort_columns.append(sort_by_rank)
        ascending.append(True)
    elif sort_by_rank:
        logging.warning(
            "Columna de ranking %s no encontrada, se usa el orden por defecto",
            sort_by_rank,
        )

    if "PLAYER_NAME" in df.columns:
        sort_columns.append("PLAYER_NAME")
    else:
        sort_columns.append("PLAYER_ID")
    ascending.append(True)

    df = df.sort_values(sort_columns, ascending=ascending, kind="mergesort")
    if "_PRIMARY_TEAM_ID" in df.columns:
        df = df.drop(columns="_PRIMARY_TEAM_ID")
    return df


def order_columns(df: pd.DataFrame) -> List[str]:
    ordered: List[str] = []
    for column in [
        "SEASON",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "MULTI_TEAM_SEASON",
        "GP",
    ]:
        if column in df.columns and column not in ordered:
            ordered.append(column)

    for stat in TOTAL_VOLUME_STATS:
        column = f"{stat}_total"
        if column in df.columns:
            ordered.append(column)

    for stat in MEAN_VOLUME_STATS:
        column = f"{stat}_G"
        if column in df.columns and column not in ordered:
            ordered.append(column)

    efficiency_order = [
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "EFG_PCT",
        "TS_PCT",
        "USG_PCT",
        "E_USG_PCT",
        "OFF_RATING",
        "DEF_RATING",
        "NET_RATING",
        "PACE",
        "PACE_PER40",
        "E_PACE",
        "TM_TOV_PCT",
        "REB_PCT",
        "OREB_PCT",
        "DREB_PCT",
        "PIE",
        "POSS",
        "PLUS_MINUS",
    ]
    for stat in efficiency_order:
        column = f"{stat}_G"
        if column in df.columns and column not in ordered:
            ordered.append(column)

    remaining_ratio = [
        column
        for column in df.columns
        if column.endswith("_G") and column not in ordered
    ]
    ordered.extend(sorted(remaining_ratio))

    dispersion_metrics_order = DISPERSION_METRICS
    for metric in dispersion_metrics_order:
        for suffix in ["STD", "VAR", "MIN", "MAX"]:
            column = f"{metric}_{suffix}"
            if column in df.columns and column not in ordered:
                ordered.append(column)

    for column in df.columns:
        if column.endswith("_RANK") and column not in ordered:
            ordered.append(column)

    remaining = [column for column in df.columns if column not in ordered]
    ordered.extend(remaining)
    return ordered


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if not args.input_path.exists():
        logging.error("El archivo de entrada no existe: %s", args.input_path)
        return

    df = pd.read_parquet(args.input_path)
    if args.season:
        if "SEASON" not in df.columns:
            logging.error("La columna SEASON no está presente en los datos")
            return
        df = df[df["SEASON"] == args.season]
        if df.empty:
            logging.error("No hay registros para la temporada %s", args.season)
            return

    summary = compute_summary(df, args.by_team_splits)
    summary, ranking_columns = add_rankings(summary, args.by_team_splits)
    summary = apply_sorting(summary, args.by_team_splits, args.sort_by_rank)

    ordered_columns = order_columns(summary)
    summary = summary[ordered_columns]

    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        logging.error("El archivo de salida ya existe. Use --overwrite para sobrescribirlo: %s", output_path)
        return
    summary.to_parquet(output_path, engine="pyarrow", index=False)

    num_players = summary["PLAYER_ID"].nunique() if "PLAYER_ID" in summary.columns else 0
    if "TEAM_ID" in summary.columns:
        num_teams = summary["TEAM_ID"].nunique()
    elif "TEAM_ABBREVIATION" in summary.columns:
        num_teams = summary["TEAM_ABBREVIATION"].nunique()
    else:
        num_teams = 0

    logging.info("Jugadores procesados: %d", num_players)
    logging.info("Equipos detectados: %d", num_teams)
    if ranking_columns:
        logging.info("Columnas de ranking generadas: %s", ", ".join(ranking_columns))
    else:
        logging.info("Columnas de ranking generadas: ninguna")
    preview = summary.head(5)
    logging.info("Vista previa (5 filas):\n%s", preview.to_string(index=False))


if __name__ == "__main__":
    main()
