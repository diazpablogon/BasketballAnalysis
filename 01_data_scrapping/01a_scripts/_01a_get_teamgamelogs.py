import os
import argparse
import logging
import time
from collections import Counter

import pandas as pd
from nba_api.stats.endpoints import TeamGameLog
from nba_api.stats.library.parameters import SeasonTypeAllStar

from _01a_function_tools import (
    _call_ep,
    list_team_ids,
    list_team_ids_for_season,
    save_parquet,
)

OUTPUT_ROOT = "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw"
ENDPOINT_SLUG = "teamgamelog"

logger = logging.getLogger(__name__)

STATUS_META = {
    "saved": ("💾", "Guardado"),
    "exists": ("⏭", "Ya existía"),
    "empty": ("⚠️", "Sin datos"),
    "error": ("❌", "Error"),
}


def _print_team_summary(
    *,
    season: str,
    team_index: int,
    total_teams: int,
    team_id: int,
    statuses: list,
    teams_with_new_data: int,
):
    """Muestra un bloque resumen visual para cada equipo procesado."""

    total_items = len(statuses)
    counter = Counter(status["status"] for status in statuses)
    new_files = counter.get("saved", 0) + counter.get("empty", 0)
    existing_files = counter.get("exists", 0)
    warnings = counter.get("empty", 0)
    errors = counter.get("error", 0)

    header = (
        f"Temporada {season} · Equipo {team_index}/{total_teams} "
        f"· TEAM_ID {team_id}"
    )
    line_length = max(len(header), 72)
    border = "─" * line_length
    progress_pct = (team_index / total_teams * 100) if total_teams else 100

    print("\n" + border)
    print(header)
    print(border)

    for status in statuses:
        icon, label = STATUS_META.get(status["status"], ("•", status["status"]))
        rel_path = status.get("rel_path", status.get("path", ""))
        if status["status"] in {"saved", "empty"}:
            rows = status.get("rows", 0)
            detail = f"{status['slug']} ({rows} filas) → {rel_path}"
        elif status["status"] == "exists":
            detail = f"{status['slug']} (ya estaba)"
        elif status["status"] == "error":
            detail = f"{status['slug']} · {status.get('message', 'falló')}"
        else:
            detail = f"{status['slug']}"
        print(f"  {icon} {label:<11} {detail}")

    summary_parts = [
        f"Descargas realizadas: {total_items}/{total_items}",
        f"Nuevos archivos: {new_files}",
        f"Ya existentes: {existing_files}",
        f"Equipos con nuevos datos: {teams_with_new_data}",
        f"Progreso temporada: {progress_pct:.1f}%",
    ]
    if warnings:
        summary_parts.insert(2, f"Sin datos: {warnings}")
    if errors:
        summary_parts.insert(2, f"Errores: {errors}")

    print("Resumen → " + " | ".join(summary_parts))
    print(border)


def _download_team_segment(
    *,
    team_id: int,
    season: str,
    season_type: str,
    season_label: str,
    max_retries: int,
    sleep: float,
) -> pd.DataFrame:
    """Descarga un segmento (Regular/Playoffs) del TeamGameLog con reintentos."""

    for attempt in range(1, max_retries + 1):
        try:
            response = _call_ep(
                TeamGameLog,
                team_id=team_id,
                season=season,
                season_type=season_type,
            )
            data_frames = response.get_data_frames()
            df = data_frames[0] if data_frames else pd.DataFrame()
            if not df.empty:
                df = df.copy()
                df["SEASON_SEGMENT"] = season_label
            return df
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                (
                    "Fallo en TeamGameLog para team_id=%s, season=%s, "
                    "season_type=%s (intento %s/%s): %s"
                ),
                team_id,
                season,
                season_type,
                attempt,
                max_retries,
                exc,
            )
            time.sleep(sleep * attempt)

    logger.error(
        "No se pudo descargar TeamGameLog para team_id=%s en %s (%s)",
        team_id,
        season,
        season_type,
    )
    return pd.DataFrame()


def _download_team_gamelog(
    *,
    team_id: int,
    season: str,
    include_playoffs: bool,
    max_retries: int,
    sleep: float,
) -> pd.DataFrame:
    """Descarga los datos completos (RS/PO) del TeamGameLog para un equipo."""

    segments = [
        (SeasonTypeAllStar.regular, "Regular Season"),
    ]
    if include_playoffs:
        segments.append((SeasonTypeAllStar.playoffs, "Playoffs"))

    frames: list[pd.DataFrame] = []
    for season_type, label in segments:
        df = _download_team_segment(
            team_id=team_id,
            season=season,
            season_type=season_type,
            season_label=label,
            max_retries=max_retries,
            sleep=sleep,
        )
        if not df.empty:
            frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        sort_cols = [col for col in ("GAME_DATE", "GAME_ID") if col in combined.columns]
        if sort_cols:
            combined.sort_values(by=sort_cols, inplace=True, ignore_index=True)
        else:
            combined.reset_index(drop=True, inplace=True)
        return combined

    return pd.DataFrame()


def process_season(
    season: str,
    *,
    include_playoffs: bool = False,
    sleep: float = 0.8,
    max_retries: int = 3,
):
    team_ids = list_team_ids_for_season(
        season,
        include_playoffs=include_playoffs,
    )
    if not team_ids:
        logger.warning(
            "%s: no se obtuvieron equipos desde LeagueGameLog; se usará lista estática",
            season,
        )
        team_ids = list_team_ids()

    team_ids = sorted(team_ids)
    total_teams = len(team_ids)
    if total_teams == 0:
        logger.warning(f"{season}: no se encontraron equipos para descargar")
        return

    logger.info(
        "%s: procesando %s equipos (include_playoffs=%s)",
        season,
        total_teams,
        include_playoffs,
    )

    teams_with_new_data = 0

    for index, team_id in enumerate(team_ids, start=1):
        statuses = []
        out_path = os.path.join(
            OUTPUT_ROOT,
            ENDPOINT_SLUG,
            season,
            f"{ENDPOINT_SLUG}__{team_id}.parquet",
        )
        rel_path = os.path.relpath(out_path, OUTPUT_ROOT)

        if os.path.exists(out_path):
            statuses.append(
                {
                    "slug": ENDPOINT_SLUG,
                    "status": "exists",
                    "path": out_path,
                    "rel_path": rel_path,
                }
            )
        else:
            df = _download_team_gamelog(
                team_id=team_id,
                season=season,
                include_playoffs=include_playoffs,
                max_retries=max_retries,
                sleep=sleep,
            )
            if df is None:
                df = pd.DataFrame()

            try:
                result = save_parquet(
                    df,
                    out_path,
                    season=season,
                    endpoint_slug=ENDPOINT_SLUG,
                    team_id=team_id,
                )
                statuses.append(
                    {
                        "slug": ENDPOINT_SLUG,
                        "status": "empty" if result["was_empty"] else "saved",
                        "rows": result["rows"],
                        "path": result["path"],
                        "rel_path": rel_path,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "No se pudo guardar TeamGameLog para team_id=%s en %s: %s",
                    team_id,
                    season,
                    exc,
                )
                statuses.append(
                    {
                        "slug": ENDPOINT_SLUG,
                        "status": "error",
                        "message": str(exc),
                        "path": out_path,
                        "rel_path": rel_path,
                    }
                )

        if any(status["status"] in {"saved", "empty"} for status in statuses):
            teams_with_new_data += 1

        _print_team_summary(
            season=season,
            team_index=index,
            total_teams=total_teams,
            team_id=team_id,
            statuses=statuses,
            teams_with_new_data=teams_with_new_data,
        )

    logger.info(
        "%s: temporada completada. Equipos procesados: %s. Con nuevos datos en %s equipos.",
        season,
        total_teams,
        teams_with_new_data,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Descarga y guarda Team Game Logs por temporada(s).",
    )
    parser.add_argument("--seasons", required=True, help="Ej: '2023-24,2024-25'")
    parser.add_argument("--include-playoffs", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.8)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    for season in (s.strip() for s in args.seasons.split(",")):
        if not season:
            continue
        process_season(
            season=season,
            include_playoffs=args.include_playoffs,
            sleep=args.sleep,
            max_retries=args.max_retries,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
