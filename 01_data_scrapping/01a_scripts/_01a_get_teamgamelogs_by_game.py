import os
import time
import argparse
import logging
from collections import Counter

import pandas as pd
from nba_api.stats.endpoints import TeamGameLogs
from nba_api.stats.library.parameters import SeasonTypeAllStar

from _01a_function_tools import _call_ep, save_parquet

OUTPUT_ROOT = "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw"
ENDPOINT_SLUG = "teamgamelogs_by_game"

logger = logging.getLogger(__name__)

STATUS_META = {
    "saved": ("💾", "Guardado"),
    "exists": ("⏭", "Ya existía"),
    "empty": ("⚠️", "Sin datos"),
    "error": ("❌", "Error"),
}


def _season_type_label(season_type: str) -> str:
    return {
        SeasonTypeAllStar.regular: "Regular Season",
        SeasonTypeAllStar.playoffs: "Playoffs",
    }.get(season_type, season_type)


def _fetch_team_game_logs(
    *, season: str, season_type: str, max_retries: int, sleep: float
) -> pd.DataFrame:
    label = _season_type_label(season_type)

    for attempt in range(max_retries):
        try:
            response = _call_ep(
                TeamGameLogs,
                season=season,
                season_type=season_type,
            )
            data_frames = response.get_data_frames() or []
            if not data_frames:
                logger.warning(
                    f"{season}: TeamGameLogs ({label}) no devolvió tablas de datos"
                )
                return pd.DataFrame()
            df = data_frames[0]
            logger.info(f"{season}: TeamGameLogs ({label}) → {len(df)} filas")
            return df
        except Exception as exc:  # noqa: BLE001
            attempt_no = attempt + 1
            logger.warning(
                (
                    f"Fallo en TeamGameLogs para {season} ({label}) "
                    f"(intento {attempt_no}/{max_retries}): {exc}"
                )
            )
            time.sleep(sleep * attempt_no)

    logger.error(
        f"No se pudo descargar TeamGameLogs para {season} ({label}) tras {max_retries} intentos"
    )
    return pd.DataFrame()


def _print_game_summary(
    *,
    season: str,
    game_index: int,
    total_games: int,
    game_id: str,
    statuses: list,
    games_with_new_data: int,
):
    total_items = len(statuses)
    counter = Counter(status["status"] for status in statuses)
    new_files = counter.get("saved", 0) + counter.get("empty", 0)
    existing_files = counter.get("exists", 0)
    empty_files = counter.get("empty", 0)
    warnings_in_game = sum(1 for status in statuses if status.get("has_warning"))

    header = f"Temporada {season} · Partido {game_index}/{total_games} · GAME_ID {game_id}"
    line_length = max(len(header), 72)
    border = "─" * line_length
    progress_pct = (game_index / total_games * 100) if total_games else 100

    print("\n" + border)
    print(header)
    print(border)

    for status in statuses:
        icon, label = STATUS_META.get(status["status"], ("•", status["status"]))
        rel_path = status.get("rel_path", status.get("path", ""))
        detail = status.get("slug", "")

        if status["status"] in {"saved", "empty"}:
            rows = status.get("rows", 0)
            detail = f"{detail} ({rows} filas) → {rel_path}"
        elif status["status"] == "exists":
            detail = f"{detail} (ya estaba)"
        elif status["status"] == "error":
            detail = f"{detail} (error)"

        note = status.get("note")
        if note:
            detail += f" · {note}"

        print(f"  {icon} {label:<11} {detail}")

    summary_parts = [
        f"Archivos listos: {total_items}/{total_items}",
        f"Nuevos archivos: {new_files}",
        f"Ya existentes: {existing_files}",
        f"Partidos con nuevos datos: {games_with_new_data}",
        f"Progreso temporada: {progress_pct:.1f}%",
    ]

    if empty_files:
        summary_parts.insert(2, f"Sin datos: {empty_files}")
    if warnings_in_game:
        summary_parts.append(f"Avisos: {warnings_in_game}")

    print("Resumen → " + " | ".join(summary_parts))
    print(border)


def process_season(
    *,
    season: str,
    include_playoffs: bool,
    sleep: float,
    max_retries: int,
):
    logger.info(
        f"{season}: iniciando descarga de TeamGameLogs (include_playoffs={include_playoffs})"
    )

    season_types = [SeasonTypeAllStar.regular]
    if include_playoffs:
        season_types.append(SeasonTypeAllStar.playoffs)

    frames = []
    for season_type in season_types:
        df_segment = _fetch_team_game_logs(
            season=season,
            season_type=season_type,
            max_retries=max_retries,
            sleep=sleep,
        )
        if df_segment.empty:
            logger.warning(
                f"{season}: sin filas en TeamGameLogs ({_season_type_label(season_type)})"
            )
            continue
        frames.append(df_segment)

    if not frames:
        logger.warning(f"{season}: TeamGameLogs no devolvió datos")
        return

    df_all = pd.concat(frames, ignore_index=True)

    if "GAME_ID" not in df_all.columns:
        logger.error(f"{season}: 'GAME_ID' no está presente en TeamGameLogs")
        return

    df_all = df_all[df_all["GAME_ID"].notna()].copy()
    if df_all.empty:
        logger.warning(f"{season}: todas las filas carecen de GAME_ID válido")
        return

    if "GAME_DATE" in df_all.columns:
        df_all = df_all.sort_values(
            by=["GAME_DATE", "GAME_ID", "TEAM_ID"],
            kind="stable",
        )
    else:
        df_all = df_all.sort_values(by=["GAME_ID", "TEAM_ID"], kind="stable")

    df_all = df_all.reset_index(drop=True)
    grouped = df_all.groupby("GAME_ID", sort=False)
    total_games = grouped.ngroups

    if total_games == 0:
        logger.warning(f"{season}: no hay GAME_IDs tras agrupar TeamGameLogs")
        return

    logger.info(f"{season}: encontrados {total_games} partidos en TeamGameLogs")

    saved_games = 0
    empty_games = 0
    existing_games = 0
    warning_games = 0
    games_with_new_data = 0

    for index, (game_id, df_game) in enumerate(grouped, start=1):
        df_game = df_game.reset_index(drop=True)
        game_id_str = str(game_id)

        out_path = os.path.join(
            OUTPUT_ROOT,
            "teamgamelogs_by_game",
            season,
            f"{ENDPOINT_SLUG}__{game_id_str}.parquet",
        )
        rel_path = os.path.relpath(out_path, OUTPUT_ROOT)

        statuses = []
        game_has_warning = False

        if os.path.exists(out_path):
            existing_games += 1
            statuses.append(
                {
                    "slug": ENDPOINT_SLUG,
                    "status": "exists",
                    "path": out_path,
                    "rel_path": rel_path,
                }
            )
        else:
            rows_obtained = len(df_game)
            note = None
            if rows_obtained != 2:
                game_has_warning = True
                warning_games += 1
                note = f"esperadas 2 filas, encontradas {rows_obtained}"
                logger.warning(
                    (
                        f"{season}: GAME_ID {game_id_str} tiene {rows_obtained} fila(s) "
                        "en TeamGameLogs"
                    )
                )

            result = save_parquet(
                df_game,
                out_path,
                season=season,
                game_id=game_id_str,
                endpoint_slug=ENDPOINT_SLUG,
            )

            status_label = "empty" if result["was_empty"] else "saved"
            statuses.append(
                {
                    "slug": ENDPOINT_SLUG,
                    "status": status_label,
                    "rows": result["rows"],
                    "path": result["path"],
                    "rel_path": rel_path,
                    "note": note,
                    "has_warning": game_has_warning,
                }
            )

            if status_label == "saved":
                saved_games += 1
            else:
                empty_games += 1

            games_with_new_data += 1

        _print_game_summary(
            season=season,
            game_index=index,
            total_games=total_games,
            game_id=game_id_str,
            statuses=statuses,
            games_with_new_data=games_with_new_data,
        )

    logger.info(
        (
            f"{season}: totales → partidos={total_games}, nuevos={saved_games}, "
            f"vacíos={empty_games}, existentes={existing_games}, avisos={warning_games}, "
            f"con nuevos datos={games_with_new_data}"
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Descarga Team Game Logs y guarda un parquet por partido",
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
