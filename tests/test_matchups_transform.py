import pytest

pd = pytest.importorskip("pandas")

from basketball_analysis.matchups import _transform_matchups


def test_transform_matchups_keeps_required_columns_with_partial_off_fields():
    df = pd.DataFrame(
        {
            "GAME_ID": ["0022400001"],
            "OFF_TEAM_ID": [1610612737],
            "OFF_PLAYER_ID": [203500],
            "OFF_PLAYER_NAME": ["Test Player"],
            "OFF_PLAYER_FIRST_NAME": ["Test"],
            "OFF_PLAYER_LAST_NAME": ["Player"],
            # Optional metadata intentionally missing (city, nickname, etc.).
            "MATCHUP_MINUTES": [12.5],
        }
    )

    transformed = _transform_matchups(df)

    assert transformed.loc[0, "TEAM_ID"] == 1610612737
    assert transformed.loc[0, "PLAYER_ID"] == 203500
    assert "OFF_TEAM_ID" not in transformed.columns
    assert "OFF_PLAYER_NAME" not in transformed.columns
    assert transformed.loc[0, "MATCHUP_MINUTES"] == pytest.approx(12.5)


def test_transform_matchups_accepts_existing_required_columns():
    df = pd.DataFrame(
        {
            "GAME_ID": ["0022400002"],
            "TEAM_ID": [1610612738],
            "PLAYER_ID": [2544],
            "SOME_OTHER_COLUMN": [1],
        }
    )

    transformed = _transform_matchups(df)

    assert transformed.equals(df)


def test_transform_matchups_raises_when_required_columns_missing():
    df = pd.DataFrame({"OFF_TEAM_ID": [1], "OFF_PLAYER_ID": [2]})

    with pytest.raises(KeyError):
        _transform_matchups(df)
