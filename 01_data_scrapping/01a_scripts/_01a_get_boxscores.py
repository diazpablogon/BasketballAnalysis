import os
import argparse
import logging

from _01a_function_tools import BOX_ENDPOINTS, list_game_ids, fetch_endpoint_df, save_parquet

# Ruta base donde guardar (ajustada a lo que pediste)
OUTPUT_ROOT = "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw"

def process_season(season: str, include_playoffs: bool = False, sleep: float = 0.8, max_retries: int = 3):
    game_ids = list_game_ids(season, include_playoffs=include_playoffs)
    logging.info(f"{season}: encontrados {len(game_ids)} partidos")

    for gid in game_ids:
        for slug, class_name in BOX_ENDPOINTS:
            out_path = os.path.join(OUTPUT_ROOT, "boxscore", slug, season, f"{slug}__{gid}.parquet")
            if os.path.exists(out_path):
                # Ya existe, saltamos
                continue
            df = fetch_endpoint_df(class_name, gid, max_retries=max_retries, sleep=sleep)
            save_parquet(df, out_path, season, gid, slug)

def main():
    parser = argparse.ArgumentParser(description="Descarga y guarda box scores por temporada(s).")
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
