import os
import argparse
import logging
from collections import Counter

from _01a_function_tools import (
    BOX_ENDPOINTS,
    list_game_ids,
    fetch_endpoint_df,
    save_parquet,
)

# Ruta base donde guardar (ajustada a lo que pediste)
OUTPUT_ROOT = "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw"

logger = logging.getLogger(__name__)

STATUS_META = {
    "saved": ("💾", "Guardado"),
    "exists": ("⏭", "Ya existía"),
    "empty": ("⚠️", "Sin datos"),
}


def _print_game_summary(
    *,
    season: str,
    game_index: int,
    total_games: int,
    game_id: str,
    statuses: list,
    games_with_new_data: int,
):
    """Muestra un bloque resumen visual para cada partido procesado."""

    total_endpoints = len(statuses)
    counter = Counter(status["status"] for status in statuses)
    new_files = counter.get("saved", 0) + counter.get("empty", 0)
    existing_files = counter.get("exists", 0)
    warnings = counter.get("empty", 0)

    header = (
        f"Temporada {season} · Partido {game_index}/{total_games} "
        f"· GAME_ID {game_id}"
    )
    line_length = max(len(header), 72)
    border = "─" * line_length
    progress_pct = (game_index / total_games * 100) if total_games else 100

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
        else:
            detail = f"{status['slug']}"
        print(f"  {icon} {label:<11} {detail}")

    summary_parts = [
        f"Endpoints listos: {total_endpoints}/{total_endpoints}",
        f"Nuevos archivos: {new_files}",
        f"Ya existentes: {existing_files}",
        f"Partidos con nuevos datos: {games_with_new_data}",
        f"Progreso temporada: {progress_pct:.1f}%",
    ]
    if warnings:
        summary_parts.insert(2, f"Sin datos: {warnings}")

    print("Resumen → " + " | ".join(summary_parts))
    print(border)


def process_season(
    season: str,
    include_playoffs: bool = False,
    sleep: float = 0.8,
    max_retries: int = 3,
):
    game_ids = list_game_ids(season, include_playoffs=include_playoffs)
    total_games = len(game_ids)
    if total_games == 0:
        logger.warning(f"{season}: no se encontraron partidos para descargar")
        return

    logger.info(f"{season}: encontrados {total_games} partidos")

    games_with_new_data = 0

    for index, gid in enumerate(game_ids, start=1):
        statuses = []

        for slug, class_name in BOX_ENDPOINTS:
            out_path = os.path.join(
                OUTPUT_ROOT, "boxscore", slug, season, f"{slug}__{gid}.parquet"
            )
            rel_path = os.path.relpath(out_path, OUTPUT_ROOT)

            if os.path.exists(out_path):
                statuses.append(
                    {
                        "slug": slug,
                        "status": "exists",
                        "path": out_path,
                        "rel_path": rel_path,
                    }
                )
                continue

            df = fetch_endpoint_df(
                class_name, gid, max_retries=max_retries, sleep=sleep
            )
            result = save_parquet(df, out_path, season, gid, slug)

            statuses.append(
                {
                    "slug": slug,
                    "status": "empty" if result["was_empty"] else "saved",
                    "rows": result["rows"],
                    "path": result["path"],
                    "rel_path": rel_path,
                }
            )

        if any(status["status"] in {"saved", "empty"} for status in statuses):
            games_with_new_data += 1

        _print_game_summary(
            season=season,
            game_index=index,
            total_games=total_games,
            game_id=gid,
            statuses=statuses,
            games_with_new_data=games_with_new_data,
        )

    logger.info(
        f"{season}: temporada completada. Partidos procesados: {total_games}. "
        f"Con nuevos datos en {games_with_new_data} partidos."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Descarga y guarda box scores por temporada(s)."
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
