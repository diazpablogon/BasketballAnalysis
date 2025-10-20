import os
import time
import inspect
import logging
import pandas as pd

# Import dinámico de endpoints para evitar fallos si alguno no existe
from nba_api.stats import endpoints as ep
from nba_api.stats.endpoints import LeagueGameLog
from nba_api.stats.library.parameters import SeasonTypeAllStar
from nba_api.stats.static import teams as static_teams

# --- Logger ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

logger = logging.getLogger(__name__)

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


def _call_ep(ep_cls, **kwargs):
    """Invoca un endpoint de ``nba_api`` adaptando parámetros con sufijo ``_nullable``.

    Distintas versiones de la librería aceptan variaciones en los nombres de los
    parámetros. Este helper revisa la signatura de ``ep_cls`` y reubica las
    claves ``season``, ``season_type`` y ``team_id`` en su equivalente aceptado
    por la clase (``*_nullable`` o ``season_type_all_star``) antes de realizar
    la llamada.
    """

    signature = inspect.signature(ep_cls.__init__)
    parameters = signature.parameters
    accepts_var_kw = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())

    aliases = {
        "season": ("season", "season_nullable"),
        "season_type": ("season_type", "season_type_nullable", "season_type_all_star"),
        "team_id": ("team_id", "team_id_nullable"),
    }

    final_kwargs = {}

    for key, value in kwargs.items():
        if key in parameters:
            final_kwargs[key] = value
            continue

        matched = False
        for canonical, options in aliases.items():
            if key == canonical or key in options:
                for candidate in options:
                    if candidate in parameters:
                        final_kwargs[candidate] = value
                        matched = True
                        break
                if matched:
                    break

        if matched:
            continue

        if accepts_var_kw:
            final_kwargs[key] = value

    return ep_cls(**final_kwargs)

def list_game_ids(season: str, include_playoffs: bool = False):
    """Devuelve todos los GAME_ID de una temporada."""
    game_ids = []

    # Regular Season
    reg = _call_ep(LeagueGameLog, season=season, season_type=SeasonTypeAllStar.regular)
    df_reg = reg.get_data_frames()[0]
    game_ids.extend(df_reg["GAME_ID"].unique())

    if include_playoffs:
        po = _call_ep(LeagueGameLog, season=season, season_type=SeasonTypeAllStar.playoffs)
        df_po = po.get_data_frames()[0]
        game_ids.extend(df_po["GAME_ID"].unique())

    # Únicos
    return list(dict.fromkeys(game_ids))


def list_team_ids_for_season(season: str, *, include_playoffs: bool = False) -> list[int]:
    """Devuelve los ``TEAM_ID`` presentes en los game logs de la temporada."""

    team_ids: list[int] = []

    def _collect_ids(season_type: str) -> list[int]:
        try:
            response = _call_ep(
                LeagueGameLog,
                season=season,
                season_type=season_type,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "%s: no se pudo obtener LeagueGameLog (%s): %s",
                season,
                season_type,
                exc,
            )
            return []

        frames = response.get_data_frames()
        if not frames:
            logging.warning(
                "%s: LeagueGameLog (%s) no devolvió tablas",
                season,
                season_type,
            )
            return []

        df = frames[0]
        if "TEAM_ID" not in df.columns:
            logging.warning(
                "%s: LeagueGameLog (%s) sin columna TEAM_ID",
                season,
                season_type,
            )
            return []

        team_series = pd.to_numeric(df["TEAM_ID"], errors="coerce").dropna().astype(int)
        return team_series.tolist()

    team_ids.extend(_collect_ids(SeasonTypeAllStar.regular))

    if include_playoffs:
        team_ids.extend(_collect_ids(SeasonTypeAllStar.playoffs))

    ordered: list[int] = []
    seen: set[int] = set()
    for team_id in team_ids:
        if team_id not in seen:
            seen.add(team_id)
            ordered.append(team_id)

    return ordered


def list_team_ids() -> list[int]:
    """Devuelve los IDs de los equipos NBA (solo franquicias activas)."""

    team_ids: list[int] = []

    get_active = getattr(static_teams, "get_active_teams", None)
    if callable(get_active):
        try:
            teams = get_active() or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("No se pudieron obtener equipos activos: %s", exc)
            teams = []

        team_ids.extend(int(team["id"]) for team in teams if team.get("id"))

    if not team_ids:
        try:
            teams = static_teams.get_teams()
        except Exception as exc:  # noqa: BLE001
            logger.error("No se pudieron obtener equipos: %s", exc)
            return []

        for team in teams:
            flag = team.get("is_nba_team")
            if flag is None:
                flag = team.get("is_nba_franchise")

            if isinstance(flag, str):
                flag = flag.strip().lower() in {"true", "t", "1", "y", "yes"}
            elif isinstance(flag, (int, float)):
                flag = flag != 0

            if flag:
                team_id = team.get("id")
                if team_id is not None:
                    team_ids.append(int(team_id))

    return sorted(set(team_ids))

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
            resp = _call_ep(
                cls,
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
