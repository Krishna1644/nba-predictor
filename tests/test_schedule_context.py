"""Schedule context must not collapse to identical rest for every team."""

from __future__ import annotations

import pandas as pd

from nba_ml.live_sources import rest_features_from_dates


def test_rest_features_vary_across_teams():
    as_of = "2026-11-20"
    dates = pd.DataFrame(
        {
            "TEAM_ID": [
                1610612737,
                1610612737,
                1610612751,
                1610612751,
                1610612751,
                1610612741,
            ],
            "GAME_DATE": pd.to_datetime(
                [
                    "2026-11-18",
                    "2026-11-15",
                    "2026-11-19",  # B2B into 11-20
                    "2026-11-17",
                    "2026-11-14",
                    "2026-11-12",
                ]
            ),
        }
    )
    team_ids = [1610612737, 1610612751, 1610612741, 1610612762]
    ctx = rest_features_from_dates(team_ids, dates, as_of)
    assert len(ctx) == 4
    combos = set(
        zip(ctx["rest_days"], ctx["is_back_to_back"], ctx["games_last_7"])
    )
    assert len(combos) > 1, f"all teams share identical rest values: {combos}"
    bkn = ctx.loc[ctx["TEAM_ID"] == 1610612751].iloc[0]
    assert int(bkn["rest_days"]) == 1
    assert int(bkn["is_back_to_back"]) == 1


def test_assert_schedule_context_varies_fails_when_uniform():
    from nba_ml.live_sources import assert_schedule_context_varies

    ctx = pd.DataFrame(
        {
            "TEAM_ID": [1, 2, 3, 4],
            "rest_days": [7, 7, 7, 7],
            "is_back_to_back": [0, 0, 0, 0],
            "games_last_7": [0, 0, 0, 0],
        }
    )
    try:
        assert_schedule_context_varies(ctx)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "expected AssertionError when all teams share identical rest"
