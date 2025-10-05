import argparse
import logging
from typing import Dict

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


def main():
    parser = argparse.ArgumentParser(
        description="Descarga boxscores de equipo (home/away) por partido.",
    )
    parser.add_argument("--season", required=True, help="Formato '2024-25'")
    parser.add_argument(
        "--season_type",
        required=True,
        help="Regular Season, Playoffs o PlayIn",
    )
    parser.add_argument("--sleep", type=float, default=0.8, help="Pausa entre llamadas")
    parser.add_argument("--max-retries", type=int, default=3, help="Reintentos por endpoint")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    game_ids = list_team_game_ids_by_season_type(args.season, args.season_type)
    total_games = len(game_ids)

    if total_games == 0:
        logger.warning(
            f"{args.season} · {args.season_type}: no se encontraron GAME_ID para procesar"
        )
        return

    logger.info(
        f"{args.season} · {args.season_type}: se procesarán {total_games} partidos"
    )

    for index, game_id in enumerate(game_ids, start=1):
        try:
            saved = process_game(
                game_id,
                season=args.season,
                season_type=args.season_type,
                sleep=args.sleep,
                max_retries=args.max_retries,
            )
        except Exception as exc:
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


if __name__ == "__main__":
    main()
