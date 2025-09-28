import os
import argparse
import logging
from 01a_function_tools import BOX_ENDPOINTS, list_game_ids, fetch_endpoint_df, save_parquet

OUTPUT_ROOT = "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw"


def process_season(season: str, include_playoffs=False, sleep=0.8, max_retries=3):
    game_ids = list_game_ids(season, include_playoffs=include_playoffs)
    logging.info(f"{season}: encontrados {len(game_ids)} partidos")

    for gid in game_ids:
        for slug, endpoint_cls in BOX_ENDPOINTS.items():
            out_path = os.path.join(
                OUTPUT_ROOT, "boxscore", slug, season, f"{slug}__{gid}.parquet"
            )
            if os.path.exists(out_path):
                continue  # skip
            df = fetch_endpoint_df(endpoint_cls, gid, max_retries=max_retries, sleep=sleep)
            save_parquet(df, out_path, season, gid, slug)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", required=True, help="Ej: '2023-24,2024-25'")
    parser.add_argument("--include-playoffs", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.8)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    for season in args.seasons.split(","):
        season = season.strip()
        process_season(
            season,
            include_playoffs=args.include_playoffs,
            sleep=args.sleep,
            max_retries=args.max_retries,
        )


if __name__ == "__main__":
    main()
