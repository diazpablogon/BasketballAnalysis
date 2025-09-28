import os
import time
import logging
import pandas as pd

# Import dinámico de endpoints para evitar fallos si alguno no existe
from nba_api.stats import endpoints as ep
from nba_api.stats.endpoints import LeagueGameLog
from nba_api.stats.library.parameters import SeasonTypeAllStar

# --- Logger ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Lista de endpoints a descargar (slug -> nombre de clase en nba_api) ---
BOX_ENDPOINTS = [
    ("boxscore_traditional_v2", "BoxScoreTraditionalV2"),
    ("boxscore_advanced_v2", "BoxScoreAdvancedV2"),
    ("boxscore_fourfactors_v2", "BoxScoreFourFactorsV2"),
    ("boxscore_misc_v2", "BoxScoreMiscV2"),
    ("boxscore_usage_v2", "BoxScoreUsageV2"),
    ("boxscore_scoring_v2", "BoxScoreScoringV2"),
    ("boxscore_playertrack_v2", "BoxScorePlayerTrackV2"),
    ("boxscore_matchups_v3", "BoxScoreMatchupsV3"),
    ("boxscore_summary_v2", "BoxScoreSummaryV2"),
    # OJO: NO incluimos "BoxScoreSimilarityScore"
]

def _get_endpoint_class(class_name: str):
    """Devuelve la clase del endpoint si existe en nba_api, o None si no está disponible."""
    return getattr(ep, class_name, None)

def list_game_ids(season: str, include_playoffs: bool = False):
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

    # Únicos
    return list(dict.fromkeys(game_ids))

def fetch_endpoint_df(endpoint_class_name: str, game_id: str, max_retries: int = 3, sleep: float = 0.8) -> pd.DataFrame:
    """Descarga un DataFrame de un endpoint concreto para un GAME_ID."""
    cls = _get_endpoint_class(endpoint_class_name)
    if cls is None:
        logging.warning(f"[skip] Endpoint no disponible en esta versión de nba_api: {endpoint_class_name}")
        return pd.DataFrame()

    for attempt in range(max_retries):
        try:
            resp = cls(game_id=game_id)
            dfs = resp.get_data_frames()
            return dfs[0] if dfs else pd.DataFrame()
        except Exception as e:
            logging.warning(f"Fallo en {endpoint_class_name} para {game_id} (intento {attempt+1}/{max_retries}): {e}")
            time.sleep(sleep * (attempt + 1))

    logging.error(f"No se pudo descargar {endpoint_class_name} para {game_id}")
    return pd.DataFrame()

def save_parquet(df: pd.DataFrame, path: str, season: str, game_id: str, endpoint_slug: str):
    """Guarda el DataFrame en parquet, creando carpetas si hace falta."""
    if df.empty:
        logging.warning(f"DF vacío en {endpoint_slug} {game_id}, guardando placeholder igualmente")
    df = df.copy()
    df["season"] = season
    df["game_id"] = game_id
    df["endpoint"] = endpoint_slug
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logging.info(f"[OK] Guardado {path}")
