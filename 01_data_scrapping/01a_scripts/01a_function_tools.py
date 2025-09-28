import os
import time
import logging
import pandas as pd
from nba_api.stats.endpoints import (
    BoxScoreTraditionalV2, BoxScoreAdvancedV2, BoxScoreFourFactorsV2,
    BoxScoreMiscV2, BoxScoreUsageV2, BoxScoreScoringV2,
    BoxScorePlayerTrackV2, BoxScoreMatchupsV3, BoxScoreSummaryV2,
    BoxScoreSimilarityScore, LeagueGameLog
)
from nba_api.stats.library.parameters import SeasonTypeAllStar

# --- Config logger ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Diccionario de endpoints ---
BOX_ENDPOINTS = {
    "boxscore_traditional_v2": BoxScoreTraditionalV2,
    "boxscore_advanced_v2": BoxScoreAdvancedV2,
    "boxscore_fourfactors_v2": BoxScoreFourFactorsV2,
    "boxscore_misc_v2": BoxScoreMiscV2,
    "boxscore_usage_v2": BoxScoreUsageV2,
    "boxscore_scoring_v2": BoxScoreScoringV2,
    "boxscore_playertrack_v2": BoxScorePlayerTrackV2,
    "boxscore_matchups_v3": BoxScoreMatchupsV3,
    "boxscore_summary_v2": BoxScoreSummaryV2,
    "boxscore_similarityscore": BoxScoreSimilarityScore,
}


def list_game_ids(season: str, include_playoffs=False):
    """Devuelve todos los GAME_ID de una temporada."""
    game_ids = []

    # Regular Season
    reg = LeagueGameLog(season=season, season_type_all_star=SeasonTypeAllStar.regular)
    df_reg = reg.get_data_frames()[0]
    game_ids.extend(df_reg["GAME_ID"].unique())

    if include_playoffs:
        po = LeagueGameLog(season=season, season_type_all_star=SeasonTypeAllStar.playoffs)
        df_po = po.get_data_frames()[0]
        game_ids.extend(df_po["GAME_ID"].unique())

    return list(set(game_ids))


def fetch_endpoint_df(endpoint_cls, game_id: str, max_retries=3, sleep=0.8):
    """Descarga un DataFrame de un endpoint concreto para un GAME_ID."""
    for attempt in range(max_retries):
        try:
            resp = endpoint_cls(game_id=game_id)
            dfs = resp.get_data_frames()
            # Normal: quedarse con la primera tabla (jugadores/equipos depende del endpoint)
            return dfs[0]
        except Exception as e:
            logging.warning(f"Fallo en {endpoint_cls.__name__} para {game_id}, intento {attempt+1}: {e}")
            time.sleep(sleep * (attempt + 1))
    logging.error(f"No se pudo descargar {endpoint_cls.__name__} para {game_id}")
    return pd.DataFrame()


def save_parquet(df: pd.DataFrame, path: str, season: str, game_id: str, endpoint: str):
    """Guarda el DataFrame en parquet, creando carpetas si hace falta."""
    if df.empty:
        logging.warning(f"DF vacío en {endpoint} {game_id}, guardando placeholder")
    df["season"] = season
    df["game_id"] = game_id
    df["endpoint"] = endpoint
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logging.info(f"Guardado {path}")
