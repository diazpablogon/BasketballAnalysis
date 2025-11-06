"""Pipeline para construir el parquet featurizado de partidos."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _print_feature_delta(
    phase: str, cols_before: set[str], cols_after: set[str], max_show: int = 10
) -> None:
    new_cols = list(cols_after - cols_before)
    n = len(new_cols)
    if n == 0:
        print(f"✅ {phase} aplicado (0 nuevas)")
        return
    sample = ", ".join(new_cols[:max_show])
    tail = "" if n <= max_show else f" …(+{n - max_show} más)"
    print(f"✅ {phase} aplicado (+{n}) -> {sample}{tail}")


def load_feature_module() -> object:
    base_path = Path(__file__).resolve()
    module_path = base_path.with_name('02_FeatureFunctions.py')
    if not module_path.exists():
        raise FileNotFoundError(f'No se encontró el módulo de features: {module_path}')

    spec = importlib.util.spec_from_file_location('feature_functions', module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'No se pudo cargar el spec para {module_path}')

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def main() -> None:
    features = load_feature_module()

    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parent.parent
    data_root = repo_root / '00_data'

    output_path = (
        data_root
        / '00d_featurized'
        / '00a_WL_prediction'
        / 'teamgamelogs.parquet'
    )
    seasons_root = data_root / '00c_final'

    if not seasons_root.exists():
        raise FileNotFoundError('No existe el directorio de temporadas: 00_data/00c_final')

    season_dirs = sorted([p for p in seasons_root.iterdir() if p.is_dir()])
    dfs: list[pd.DataFrame] = []

    for season_dir in season_dirs:
        team_input = season_dir / 'teamgamelogs_by_game.parquet'
        venue_path = (
            season_dir / 'dashboards' / 'team_dashboard_by_general_splits__dataset_1.parquet'
        )
        daysrest_path = (
            season_dir / 'dashboards' / 'team_dashboard_by_days_rest__dataset_1.parquet'
        )

        if not team_input.exists():
            print(f"⚠️  Sin datos en {season_dir.name}, se omite.")
            continue

        df = pd.read_parquet(team_input)
        print(
            f"Dataset {season_dir.name}: {df.shape[0]} filas, {df.shape[1]} columnas"
        )
        print('✅ Filtro competitivo')

        if 'WL_NUM' not in df.columns:
            if 'WL' in df.columns:
                df['WL_NUM'] = (
                    df['WL']
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .map({'W': 1, 'L': 0})
                )
                print('🔧 Se creó WL_NUM a partir de WL.')
            else:
                raise ValueError("Faltan 'WL' y 'WL_NUM' en el dataset base.")

        cols0 = set(df.columns)
        df = features.features_baseline(df)
        _print_feature_delta('BASELINE', cols0, set(df.columns))

        if venue_path.exists():
            cols0 = set(df.columns)
            df = features.features_venue(df, venue_path=str(venue_path))
            _print_feature_delta('VENUE', cols0, set(df.columns))
        else:
            print(f"ℹ️  VENUE omitido en {season_dir.name} (sin dashboard)")

        enhanced_config: dict[str, object] = {
            'days_rest_path': str(daysrest_path) if daysrest_path.exists() else None,
            'new_features': ['DAYS_REST', 'LAST_5_PCT', 'BACK_TO_BACK_FLAG'],
        }
        cols0 = set(df.columns)
        df = features.features_enhanced(df, config=enhanced_config)
        _print_feature_delta('ENHANCED', cols0, set(df.columns))

        cols0 = set(df.columns)
        df = features.features_elo(df)
        _print_feature_delta('ELO', cols0, set(df.columns))

        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            'No se encontraron temporadas con datos válidos en 00c_final'
        )

    df_all = pd.concat(dfs, ignore_index=True)
    print(
        f"✅ Consolidado multi-temporada: {df_all.shape[0]} filas, {df_all.shape[1]} columnas"
    )

    match_df = features.build_match_level(df_all)
    print(f"✅ Match-level: {match_df.shape[0]} filas x {match_df.shape[1]} columnas")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    match_df.to_parquet(output_path, index=False)
    print(f"Parquet final guardado en: {output_path}")
    print('✅ Match-level final guardado')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:  # pragma: no cover - ejecución script
        print(f"❌ Error en la construcción de features: {exc}")
        raise
