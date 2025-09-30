import os
import re
import argparse
import logging

from nba_api.stats.library.parameters import SeasonTypeAllStar

from _01a_function_tools import (
    TEAM_DASH_ENDPOINTS,
    list_team_ids,
    fetch_team_endpoint_tables,
    save_parquet,
)

# Ruta base donde guardar (misma que otros scripts)
OUTPUT_ROOT = "/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00a_raw"

logger = logging.getLogger(__name__)

STATUS_META = {
    "saved": ("💾", "Guardado"),
    "exists": ("⏭", "Ya existía"),
    "empty": ("⚠️", "Sin datos"),
}


def _slugify_dataset(name: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return slug or "dataset"


def _print_team_summary(
    *,
    season: str,
    season_type: str,
    team_index: int,
    total_teams: int,
    team_id: int,
    statuses: list,
    teams_with_new_data: int,
):
    """Muestra un bloque resumen visual para cada equipo procesado."""

    total_entries = len(statuses)
    new_files = sum(
        1
        for status in statuses
        if status["status"] in {"saved", "empty"} and status.get("path")
    )
    existing_files = sum(1 for status in statuses if status["status"] == "exists")
    warnings = sum(1 for status in statuses if status["status"] == "empty")

    header = (
        f"Temporada {season} · {season_type} · Equipo {team_index}/{total_teams} "
        f"· TEAM_ID {team_id}"
    )
    line_length = max(len(header), 80)
    border = "─" * line_length
    progress_pct = (team_index / total_teams * 100) if total_teams else 100

    print("\n" + border)
    print(header)
    print(border)

    for status in statuses:
        icon, label = STATUS_META.get(status["status"], ("•", status["status"]))
        rel_path = status.get("rel_path", status.get("path", ""))
        dataset_label = status.get("dataset")
        detail_slug = status["slug"] if dataset_label is None else f"{status['slug']} · {dataset_label}"
        if status["status"] in {"saved", "empty"}:
            rows = status.get("rows", 0)
            detail = f"{detail_slug} ({rows} filas) → {rel_path}"
        elif status["status"] == "exists":
            detail = f"{detail_slug} (ya estaba)"
        else:
            detail = detail_slug
        print(f"  {icon} {label:<11} {detail}")

    summary_parts = [
        f"Tablas listas: {total_entries}/{total_entries}",
        f"Nuevos archivos: {new_files}",
        f"Ya existentes: {existing_files}",
        f"Equipos con nuevos datos: {teams_with_new_data}",
        f"Progreso etapa: {progress_pct:.1f}%",
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
    team_ids = sorted(list_team_ids())
    if not team_ids:
        logger.warning("No se encontraron equipos NBA para procesar")
        return

    season_types = [SeasonTypeAllStar.regular]
    if include_playoffs:
        season_types.append(SeasonTypeAllStar.playoffs)

    total_teams = len(team_ids)

    for season_type in season_types:
        logger.info(
            f"{season} ({season_type}): procesando {total_teams} equipos · {len(TEAM_DASH_ENDPOINTS)} endpoints"
        )

        teams_with_new_data = 0

        for index, team_id in enumerate(team_ids, start=1):
            statuses = []

            for slug, class_name in TEAM_DASH_ENDPOINTS:
                tables = fetch_team_endpoint_tables(
                    class_name,
                    team_id=team_id,
                    season=season,
                    season_type=season_type,
                    max_retries=max_retries,
                    sleep=sleep,
                )

                if not tables:
                    statuses.append(
                        {
                            "slug": slug,
                            "dataset": None,
                            "status": "empty",
                            "rows": 0,
                            "path": "",
                            "rel_path": "",
                        }
                    )
                    continue

                for dataset_name, df in tables:
                    dataset_slug = _slugify_dataset(dataset_name)
                    out_path = os.path.join(
                        OUTPUT_ROOT,
                        "team_dashboard",
                        slug,
                        season,
                        season_type,
                        f"{slug}__{team_id}__{dataset_slug}.parquet",
                    )
                    rel_path = os.path.relpath(out_path, OUTPUT_ROOT)

                    if os.path.exists(out_path):
                        statuses.append(
                            {
                                "slug": slug,
                                "dataset": dataset_name,
                                "status": "exists",
                                "path": out_path,
                                "rel_path": rel_path,
                            }
                        )
                        continue

                    result = save_parquet(
                        df,
                        out_path,
                        season=season,
                        endpoint_slug=slug,
                        team_id=team_id,
                        season_type=season_type,
                        dataset=dataset_name,
                    )

                    statuses.append(
                        {
                            "slug": slug,
                            "dataset": dataset_name,
                            "status": "empty" if result["was_empty"] else "saved",
                            "rows": result["rows"],
                            "path": result["path"],
                            "rel_path": rel_path,
                        }
                    )

            has_new_data = any(
                status["status"] in {"saved", "empty"} and status.get("path")
                for status in statuses
            )
            if has_new_data:
                teams_with_new_data += 1

            _print_team_summary(
                season=season,
                season_type=season_type,
                team_index=index,
                total_teams=total_teams,
                team_id=team_id,
                statuses=statuses,
                teams_with_new_data=teams_with_new_data,
            )

        logger.info(
            f"{season} ({season_type}): etapa completada. Equipos procesados: {total_teams}. "
            f"Con nuevos datos en {teams_with_new_data} equipos."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Descarga y guarda los dashboards de equipos por temporada(s)."
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

