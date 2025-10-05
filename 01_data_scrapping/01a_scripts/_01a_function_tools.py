import os
import time
import inspect
import logging
import pandas as pd
from typing import Dict, Iterable, Optional

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


BOX_SCORE_TEAM_ENDPOINT_CLASSES = [
    "BoxScoreTraditionalV2",
    "BoxScoreAdvancedV2",
    "BoxScoreFourFactorsV2",
    "BoxScoreMiscV2",
    "BoxScoreScoringV2",
    "BoxScoreSummaryV2",
]


BOX_SCORE_TEAMS_OUTPUT_ROOT = (
    "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw/boxscore_teams"
)


def _normalize_season_type_value(season_type: str) -> str:
    """Normaliza etiquetas de ``season_type`` según los valores esperados por la API."""

    if not season_type:
        return season_type

    lookup = {
        SeasonTypeAllStar.regular.lower(): SeasonTypeAllStar.regular,
        SeasonTypeAllStar.playoffs.lower(): SeasonTypeAllStar.playoffs,
        SeasonTypeAllStar.pre_season.lower(): SeasonTypeAllStar.pre_season,
        SeasonTypeAllStar.all_star.lower(): SeasonTypeAllStar.all_star,
        "playin": "PlayIn",
    }

    normalized = season_type.strip()
    key = normalized.lower()
    return lookup.get(key, normalized)


def list_team_game_ids_by_season_type(season: str, season_type: str) -> list[str]:
    """Obtiene los ``GAME_ID`` de una temporada para equipos (``player_or_team='T'``)."""

    normalized_type = _normalize_season_type_value(season_type)

    try:
        response = _call_ep(
            LeagueGameLog,
            season=season,
            season_type=normalized_type,
            player_or_team_abbreviation="T",
        )
    except Exception as exc:
        logging.error(
            f"Error listando GAME_ID para season={season}, season_type={season_type}: {exc}"
        )
        return []

    frames = response.get_data_frames() if response else []
    if not frames:
        return []

    df = frames[0]
    if "GAME_ID" not in df.columns:
        return []

    return [str(gid) for gid in df["GAME_ID"].astype(str).unique().tolist()]


def _iter_endpoint_data_sets(resp) -> Iterable[tuple[str, pd.DataFrame]]:
    """Itera los datasets de una respuesta de ``nba_api`` devolviendo (nombre, DF)."""

    for index, dataset in enumerate(getattr(resp, "data_sets", []) or []):
        name = getattr(dataset, "name", f"dataset_{index}")
        if hasattr(dataset, "get_data_frame"):
            df = dataset.get_data_frame()
        else:
            df = pd.DataFrame(dataset)
        yield name, df


def _normalize_boxscore_team_columns(df: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """Asegura las columnas clave (GAME_ID, TEAM_ID, TEAM_ABBREVIATION, TEAM_NAME)."""

    normalized_df = df.copy()

    normalized_df["GAME_ID"] = str(game_id)

    if "TEAM_ID" in normalized_df.columns:
        normalized_df["TEAM_ID"] = normalized_df["TEAM_ID"].astype(str)

    if "TEAM_ABBREVIATION" not in normalized_df.columns:
        for candidate in [
            "TEAM_ABBREV",
            "TEAM_ABBREVIATION_x",
            "TEAM_ABBREVIATION_y",
        ]:
            if candidate in normalized_df.columns:
                normalized_df["TEAM_ABBREVIATION"] = normalized_df[candidate]
                break
    if "TEAM_ABBREVIATION" in normalized_df.columns:
        normalized_df["TEAM_ABBREVIATION"] = (
            normalized_df["TEAM_ABBREVIATION"].astype(str).str.strip()
        )

    if "TEAM_NAME" not in normalized_df.columns:
        if {"TEAM_CITY_NAME", "TEAM_NICKNAME"}.issubset(normalized_df.columns):
            normalized_df["TEAM_NAME"] = (
                normalized_df["TEAM_CITY_NAME"].fillna("").str.strip()
                + " "
                + normalized_df["TEAM_NICKNAME"].fillna("").str.strip()
            ).str.strip()
        elif {"TEAM_CITY", "TEAM_NICKNAME"}.issubset(normalized_df.columns):
            normalized_df["TEAM_NAME"] = (
                normalized_df["TEAM_CITY"].fillna("").str.strip()
                + " "
                + normalized_df["TEAM_NICKNAME"].fillna("").str.strip()
            ).str.strip()
        elif "TEAM_NICKNAME" in normalized_df.columns:
            normalized_df["TEAM_NAME"] = normalized_df["TEAM_NICKNAME"].fillna("").str.strip()

    if "TEAM_NAME" in normalized_df.columns:
        normalized_df["TEAM_NAME"] = normalized_df["TEAM_NAME"].astype(str).str.strip()

    return normalized_df


def _combine_summary_tables(resp, game_id: str) -> pd.DataFrame:
    """Combina LineScore y OtherStats (si aplica) para ``BoxScoreSummaryV2``."""

    data_sets = dict(_iter_endpoint_data_sets(resp))
    line_score = data_sets.get("LineScore", pd.DataFrame())

    if line_score.empty:
        return _normalize_boxscore_team_columns(line_score, game_id)

    other_stats = data_sets.get("OtherStats")
    if other_stats is not None and not other_stats.empty and "TEAM_ID" in other_stats.columns:
        other_stats = other_stats.drop_duplicates(subset=["TEAM_ID"])
        merged = line_score.merge(other_stats, on="TEAM_ID", how="left", suffixes=("", "_other"))
    else:
        merged = line_score

    return _normalize_boxscore_team_columns(merged, game_id)


def _fetch_single_boxscore_team_df(
    endpoint_class_name: str,
    game_id: str,
    *,
    max_retries: int = 3,
    sleep: float = 0.8,
) -> Optional[pd.DataFrame]:
    """Descarga y prepara el DataFrame de equipos para un endpoint concreto."""

    cls = _get_endpoint_class(endpoint_class_name)
    if cls is None:
        logging.warning(
            f"[skip] Endpoint no disponible en esta versión de nba_api: {endpoint_class_name}"
        )
        return None

    for attempt in range(max_retries):
        try:
            response = _call_ep(cls, game_id=game_id)

            if endpoint_class_name == "BoxScoreSummaryV2":
                df = _combine_summary_tables(response, game_id)
            else:
                df = pd.DataFrame()
                for name, table in _iter_endpoint_data_sets(response):
                    if name.lower() == "teamstats":
                        df = table
                        break
                df = _normalize_boxscore_team_columns(df, game_id)

            if df.empty:
                logging.warning(
                    f"{endpoint_class_name}: DF vacío para GAME_ID {game_id}, se omite guardado"
                )
                return df

            if len(df) != 2:
                logging.error(
                    f"{endpoint_class_name}: se esperaban 2 filas y se obtuvieron {len(df)}"
                    f" para GAME_ID {game_id}"
                )
                return None

            required_columns = {"GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME"}
            missing = required_columns.difference(df.columns)
            if missing:
                logging.error(
                    f"{endpoint_class_name}: faltan columnas {sorted(missing)} en GAME_ID {game_id}"
                )
                return None

            return df
        except Exception as exc:
            logging.warning(
                f"Fallo en {endpoint_class_name} para GAME_ID {game_id} (intento {attempt + 1}/{max_retries}): {exc}"
            )
            time.sleep(sleep * (attempt + 1))

    logging.error(
        f"No se pudo descargar {endpoint_class_name} para GAME_ID {game_id} tras {max_retries} intentos"
    )
    return None


def fetch_boxscore_team_data(
    game_id: str,
    *,
    max_retries: int = 3,
    sleep: float = 0.8,
) -> Dict[str, pd.DataFrame]:
    """Descarga todos los endpoints de boxscore de equipos para un ``GAME_ID``."""

    results: Dict[str, pd.DataFrame] = {}

    for endpoint_class_name in BOX_SCORE_TEAM_ENDPOINT_CLASSES:
        df = _fetch_single_boxscore_team_df(
            endpoint_class_name,
            game_id,
            max_retries=max_retries,
            sleep=sleep,
        )

        if df is not None and not df.empty:
            results[endpoint_class_name] = df

        time.sleep(sleep)

    return results


def save_boxscore_team_parquet(
    df: pd.DataFrame,
    *,
    season: str,
    season_type: str,
    game_id: str,
    endpoint_name: str,
    base_output_path: Optional[str] = None,
) -> Dict[str, object]:
    """Guarda un DataFrame de boxscore de equipos respetando la convención de nombres."""

    if len(df) != 2:
        raise ValueError(
            f"El DataFrame para {endpoint_name} y GAME_ID {game_id} debe tener 2 filas"
        )

    required_columns = {"GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas obligatorias {sorted(missing)} en {endpoint_name} (GAME_ID {game_id})"
        )

    output_root = base_output_path or BOX_SCORE_TEAMS_OUTPUT_ROOT
    dir_path = os.path.join(output_root, season, season_type)
    os.makedirs(dir_path, exist_ok=True)

    filename = f"boxscore_teams__{game_id}__{endpoint_name}.parquet"
    full_path = os.path.join(dir_path, filename)

    df.to_parquet(full_path, index=False)

    return {
        "path": full_path,
        "rows": len(df),
        "endpoint": endpoint_name,
        "game_id": game_id,
    }
