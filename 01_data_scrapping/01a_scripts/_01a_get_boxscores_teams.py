import argparse
import logging
from typing import Dict, List

import pandas as pd

from _01a_function_tools import (
    fetch_boxscore_team_data,
    list_team_game_ids_by_season_type,
    save_boxscore_team_parquet,
)

logger = logging.getLogger(__name__)


def _format_progress(index: int, total: int) -> str:
    if total == 0:
        return "0/0 (0.0%)"
    pct = (index / total) * 100
    return f"{index}/{total} ({pct:.1f}%)"


def _split_csv_argument(raw_value: str) -> List[str]:
    return [value.strip() for value in raw_value.split(",") if value.strip()]


def _resolve_seasons(args: argparse.Namespace, parser: argparse.ArgumentParser) -> List[str]:
    if args.seasons and args.season:
        parser.error("No combines --season con --seasons")

    if args.seasons:
        seasons = _split_csv_argument(args.seasons)
    elif args.season:
        seasons = [args.season.strip()]
    else:
        parser.error("Debes indicar --season o --seasons")

    if not seasons:
        parser.error("La lista de temporadas está vacía")

    return list(dict.fromkeys(seasons))


def _resolve_season_types(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> List[str]:
    if args.season_types and args.season_type:
        parser.error("No combines --season-type con --season-types")

    if args.playoffs_only and args.regular_only:
        parser.error("No puedes combinar --playoffs-only y --regular-only")

    if args.season_types:
        season_types = _split_csv_argument(args.season_types)
    elif args.season_type:
        season_types = [args.season_type.strip()]
    else:
        season_types = []
        if not args.playoffs_only:
            season_types.append("Regular Season")
        if args.include_playoffs or args.playoffs_only:
            season_types.append("Playoffs")
        if args.regular_only:
            season_types = ["Regular Season"]
        if args.playoffs_only:
            season_types = ["Playoffs"]

    if not season_types:
        parser.error("No hay season_types válidos para procesar")

    return list(dict.fromkeys(season_types))


def process_game(
    game_id: str,
    *,
    season: str,
    season_type: str,
    sleep: float,
    max_retries: int,
) -> Dict[str, Dict[str, object]]:
    datasets = fetch_boxscore_team_data(
        game_id,
        max_retries=max_retries,
        sleep=sleep,
    )

    saved = {}

    for endpoint_name, df in datasets.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.warning(
                f"GAME_ID {game_id} · {endpoint_name}: DataFrame vacío, se omite guardado"
            )
            continue

        try:
            result = save_boxscore_team_parquet(
                df,
                season=season,
                season_type=season_type,
                game_id=game_id,
                endpoint_name=endpoint_name,
            )
            saved[endpoint_name] = result
        except Exception as exc:
            logger.error(
                f"GAME_ID {game_id} · {endpoint_name}: error guardando parquet → {exc}"
            )

    return saved


def process_season_type(
    *,
    season: str,
    season_type: str,
    sleep: float,
    max_retries: int,
) -> Dict[str, object]:
    game_ids = list_team_game_ids_by_season_type(season, season_type)
    total_games = len(game_ids)

    if total_games == 0:
        logger.warning(
            f"{season} · {season_type}: no se encontraron GAME_ID para procesar"
        )
        return {"total_games": 0, "failed_games": 0}

    logger.info(
        f"{season} · {season_type}: se procesarán {total_games} partidos"
    )

    failed_games = 0

    for index, game_id in enumerate(game_ids, start=1):
        try:
            saved = process_game(
                game_id,
                season=season,
                season_type=season_type,
                sleep=sleep,
                max_retries=max_retries,
            )
        except Exception as exc:
            failed_games += 1
            logger.error(f"GAME_ID {game_id}: error inesperado → {exc}")
            continue

        if not saved:
            logger.warning(
                f"GAME_ID {game_id}: no se guardaron endpoints de equipos"
            )
        else:
            logger.info(
                (
                    f"GAME_ID {game_id}: guardados {len(saved)} endpoints"
                    f" ({', '.join(sorted(saved))}) · "
                    f"Progreso { _format_progress(index, total_games) }"
                )
            )

    logger.info(
        (
            f"{season} · {season_type}: procesamiento finalizado. "
            f"Partidos con error: {failed_games}/{total_games}"
        )
    )

    return {"total_games": total_games, "failed_games": failed_games}


def main():
    parser = argparse.ArgumentParser(
        description="Descarga boxscores de equipo (home/away) por partido.",
    )
    parser.add_argument("--season", help="Formato '2024-25'")
    parser.add_argument("--seasons", help="Formato '2024-25,2025-26'")
    parser.add_argument(
        "--season_type",
        help="Regular Season, Playoffs o PlayIn",
    )
    parser.add_argument(
        "--season-types",
        help="Lista separada por comas de season types a procesar",
    )
    parser.add_argument("--include-playoffs", action="store_true")
    parser.add_argument("--regular-only", action="store_true")
    parser.add_argument("--playoffs-only", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.8, help="Pausa entre llamadas")
    parser.add_argument("--max-retries", type=int, default=3, help="Reintentos por endpoint")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    seasons = _resolve_seasons(args, parser)
    season_types = _resolve_season_types(args, parser)

    exit_code = 0

    for season in seasons:
        for season_type in season_types:
            result = process_season_type(
                season=season,
                season_type=season_type,
                sleep=args.sleep,
                max_retries=args.max_retries,
            )
            if result.get("failed_games"):
                exit_code = 1

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
