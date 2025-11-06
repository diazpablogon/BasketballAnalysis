"""Pipeline para construir el parquet featurizado de partidos."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


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


def read_optional_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - robustez
        print(f"⚠️ No se pudo leer {path.name}: {exc}")
        return None


def main() -> None:
    features = load_feature_module()

    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parent.parent
    data_root = repo_root / '00_data'
    season = '2024-25'

    team_input = data_root / '00c_final' / season / 'teamgamelogs_by_game.parquet'
    venue_path = data_root / '00c_final' / season / 'dashboards' / 'team_dashboard_by_general_splits__dataset_1.parquet'
    lineup_team_path = data_root / '00c_final' / season / 'dashboards' / 'team_dash_lineups__dataset_1.parquet'
    lineup_score_path = Path(
        '/Users/pablo/Documents/BigData/BasketballAnalysis/00_data/00c_final/2024-25/dashboards/'
        'team_dash_lineups__dataset_1.parquet'
    )
    output_path = data_root / '00d_featurized' / season / 'teamgamelogs_featurized.parquet'

    if not team_input.exists():
        raise FileNotFoundError(f'No existe el parquet de entrada requerido: {team_input}')

    df = pd.read_parquet(team_input)
    print(f"Dataset base cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
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

    df = features.features_baseline(df)
    print('✅ BASELINE aplicado')

    df = features.features_venue(df, venue_path=str(venue_path))
    print('✅ VENUE aplicado')

    df = features.features_enhanced(df, config={})
    features.impute_advanced_inplace(df)
    print('✅ ENHANCED aplicado')

    df = features.features_lineup(df, lineup_path=str(lineup_team_path))
    lineup_cols = [
        'LINEUP_WEIGHTED_NET',
        'LINEUP_TOP5_NET',
        'LINEUP_TOP5_MIN',
        'LINEUP_STABILITY_HHI',
        'LINEUP_VARIETY_5MIN',
        'LINEUP_TOTAL_MIN',
    ]
    if any(col in df.columns for col in lineup_cols):
        print('✅ LINEUP aplicado')
    else:
        print('⚠️ LINEUP omitido (sin métricas disponibles)')

    df = features.add_calendar_no_leak(df)
    print('✅ Calendario sin fuga añadido')

    df = features.add_elo_no_leak(df)
    print('✅ Elo sin fuga añadido')

    lineup_scores_df = read_optional_parquet(lineup_score_path)
    if lineup_scores_df is not None:
        print(f"ℹ️  Lineup por partido aplicado desde '{lineup_score_path.name}'")

    match_df = features.build_match_level(df, lineup_scores=lineup_scores_df)
    print(f"✅ Match-level: {match_df.shape[0]} filas x {match_df.shape[1]} columnas")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    match_df.to_parquet(output_path, index=False)
    print(f"Parquet final guardado en: {output_path}")


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:  # pragma: no cover - ejecución script
        print(f"❌ Error en la construcción de features: {exc}")
        raise
