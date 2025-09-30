import os
import time
import logging
import pandas as pd

# Import dinámico de endpoints para evitar fallos si alguno no existe
from nba_api.stats import endpoints as ep
from nba_api.stats.endpoints import LeagueGameLog
from nba_api.stats.library.parameters import SeasonTypeAllStar
from nba_api.stats.static import teams as static_teams

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
    ("boxscore_matchups_v3", "BoxScoreMatchupsV3"),
    ("boxscore_summary_v2", "BoxScoreSummaryV2"),
    # OJO: NO incluimos "BoxScoreSimilarityScore"
]


TEAM_DASH_ENDPOINTS = [
    ("team_player_dashboard", "TeamPlayerDashboard"),
    ("team_player_on_off_details", "TeamPlayerOnOffDetails"),
    ("team_player_on_off_summary", "TeamPlayerOnOffSummary"),
    ("team_vs_player", "TeamVsPlayer"),
    ("team_dash_lineups", "TeamDashLineups"),
    ("team_dash_pt_pass", "TeamDashPtPass"),
    ("team_dash_pt_reb", "TeamDashPtReb"),
    ("team_dash_pt_shots", "TeamDashPtShots"),
    ("team_dashboard_by_general_splits", "TeamDashboardByGeneralSplits"),
    ("team_dashboard_by_shooting_splits", "TeamDashboardByShootingSplits"),
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


def list_team_ids() -> list[int]:
    """Devuelve los IDs de los equipos NBA (solo franquicias activas)."""

    teams = static_teams.get_teams()
    return [team["id"] for team in teams if team.get("is_nba_team")]

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


def fetch_team_endpoint_tables(
    endpoint_class_name: str,
    *,
    team_id: int,
    season: str,
    season_type: str,
    max_retries: int = 3,
    sleep: float = 0.8,
    **extra_params,
):
    """Descarga todas las tablas disponibles para un endpoint de dashboards de equipo."""

    cls = _get_endpoint_class(endpoint_class_name)
    if cls is None:
        logging.warning(
            f"[skip] Endpoint no disponible en esta versión de nba_api: {endpoint_class_name}"
        )
        return []

    base_kwargs = {"team_id": team_id, "season": season}
    base_kwargs.update(extra_params)

    for attempt in range(max_retries):
        try:
            try:
                resp = cls(
                    season_type_all_star=season_type,
                    **base_kwargs,
                )
            except TypeError:
                resp = cls(
                    season_type=season_type,
                    **base_kwargs,
                )

            tables = []
            for index, dataset in enumerate(getattr(resp, "data_sets", []) or []):
                dataset_name = getattr(dataset, "name", f"dataset_{index}")
                if hasattr(dataset, "get_data_frame"):
                    df = dataset.get_data_frame()
                else:
                    df = pd.DataFrame(dataset)
                tables.append((dataset_name, df))
            return tables
        except Exception as e:
            logging.warning(
                (
                    f"Fallo en {endpoint_class_name} para team_id={team_id}, "
                    f"season={season}, season_type={season_type} "
                    f"(intento {attempt + 1}/{max_retries}): {e}"
                )
            )
            time.sleep(sleep * (attempt + 1))

    logging.error(
        f"No se pudo descargar {endpoint_class_name} para team_id={team_id} en {season} ({season_type})"
    )
    return []

def save_parquet(
    df: pd.DataFrame,
    path: str,
    season: str | None = None,
    game_id: str | None = None,
    endpoint_slug: str | None = None,
    **identifiers,
):
    """Guarda el DataFrame en parquet y devuelve metadatos útiles sobre la operación."""

    was_empty = df.empty
    metadata = {}
    if season is not None:
        metadata["season"] = season
    if game_id is not None:
        metadata["game_id"] = game_id
    if endpoint_slug is not None:
        metadata["endpoint"] = endpoint_slug

    metadata.update(identifiers)

    if was_empty:
        context = ", ".join(f"{k}={v}" for k, v in metadata.items())
        endpoint_label = endpoint_slug or "endpoint"
        context_str = f" ({context})" if context else ""
        logging.warning(
            f"DF vacío en {endpoint_label}{context_str}, guardando placeholder igualmente"
        )

    df_to_save = df.copy()
    for column, value in metadata.items():
        df_to_save[column] = value

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_to_save.to_parquet(path, index=False)

    return {
        "path": path,
        "rows": len(df_to_save),
        "was_empty": was_empty,
    }
