"""Utilities to normalise box score matchup tables."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Mapping, Sequence

try:  # pragma: no cover - exercised via tests when pandas is available
    import pandas as pd
except ImportError:  # pragma: no cover - best effort fallback for environments without pandas
    pd = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

_MATCHUP_RENAME_TEMPLATE: Mapping[str, str] = {
    "OFF_TEAM_ID": "TEAM_ID",
    "OFF_TEAM_ABBREVIATION": "TEAM_ABBREVIATION",
    "OFF_TEAM_CITY": "TEAM_CITY",
    "OFF_TEAM_NAME": "TEAM_NAME",
    "OFF_TEAM_NICKNAME": "TEAM_NICKNAME",
    "OFF_TEAM_SLUG": "TEAM_SLUG",
    "OFF_PLAYER_ID": "PLAYER_ID",
    "OFF_PLAYER_FIRST_NAME": "PLAYER_FIRST_NAME",
    "OFF_PLAYER_LAST_NAME": "PLAYER_LAST_NAME",
    "OFF_PLAYER_NAME": "PLAYER_NAME",
    "OFF_PLAYER_NUMBER": "PLAYER_NUMBER",
    "OFF_PLAYER_POSITION": "PLAYER_POSITION",
    "OFF_PLAYER_SLUG": "PLAYER_SLUG",
}

_REQUIRED_COLUMNS: Sequence[str] = ("GAME_ID", "TEAM_ID", "PLAYER_ID")


def _build_matchup_rename_map(columns: Iterable[str]) -> Dict[str, str]:
    """Return a rename map keeping only keys present in ``columns``."""

    return {key: value for key, value in _MATCHUP_RENAME_TEMPLATE.items() if key in columns}


def _transform_matchups(df: "pd.DataFrame") -> "pd.DataFrame":
    """Normalise offensive matchup columns to the shared schema.

    Parameters
    ----------
    df:
        Raw dataframe returned by ``BoxScoreMatchupsV3``. The function performs
        a best-effort rename of ``OFF_*`` columns so the offensive team/player
        fields align with the standard box-score schema. Only columns that are
        present in ``df`` are renamed, allowing matchups without optional
        metadata (nickname, city, etc.) to be processed without raising errors.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with the available offensive columns renamed. The
        function validates that ``GAME_ID``, ``TEAM_ID`` and ``PLAYER_ID`` exist
        after the rename, raising ``KeyError`` if any of them are missing.
    """

    if pd is None:  # pragma: no cover - executed only when pandas is missing
        raise ImportError("pandas is required to transform matchups data")

    if df.empty:
        return df.copy()

    rename_map = _build_matchup_rename_map(df.columns)
    transformed = df.rename(columns=rename_map).copy()

    missing_required = [column for column in _REQUIRED_COLUMNS if column not in transformed.columns]
    if missing_required:
        message = (
            "Matchups frame is missing required columns after renaming: "
            + ", ".join(sorted(missing_required))
        )
        LOGGER.error(message)
        raise KeyError(message)

    return transformed


__all__ = ["_transform_matchups"]
