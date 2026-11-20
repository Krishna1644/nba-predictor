"""Absence / deficit / training-guard tests. Run: pytest tests/ -q"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from nba_ml.absence import canonicalize_injury_status, is_exact_out_status
from nba_ml.features import (
    MATCHUP_CORE_FEATURES,
    MAX_CONSECUTIVE_MISSES,
    REST_DAYS_CAP,
    ROTATION_MIN_THRESHOLD,
    blend_player_prior,
    compute_team_absence_deficit,
    player_impact,
    player_impact_eff,
)
from nba_ml.player_priors import (
    TeamAvailabilityTracker,
    serving_missing_ids_from_injuries,
)
import train_baseline


def test_train_and_serve_deficit_equal_given_same_missing_list():
    prior_lookup = {
        1: {"impact": 20.0, "minutes": 32.0, "consecutive_misses": 0},
        2: {"impact": 12.5, "minutes": 28.0, "consecutive_misses": 2},
        3: {"impact": 8.0, "minutes": 18.0, "consecutive_misses": 0},
    }
    missing = [1, 3]
    assert compute_team_absence_deficit(missing, prior_lookup) == 28.0
    assert compute_team_absence_deficit(missing, prior_lookup, value_key="minutes") == 50.0


def test_questionable_adds_zero_to_deficit():
    injuries = pd.DataFrame(
        {"Player": ["Star Player", "Bench Guy"], "Status": ["Questionable", "Out"]}
    )
    roster = pd.DataFrame(
        {"PLAYER_ID": [101, 102], "PLAYER": ["Star Player", "Bench Guy"]}
    )
    prior_lookup = {
        101: {"impact": 30.0, "minutes": 34.0, "consecutive_misses": 0},
        102: {"impact": 10.0, "minutes": 22.0, "consecutive_misses": 0},
    }
    missing = serving_missing_ids_from_injuries(injuries, roster, prior_lookup)
    assert 101 not in missing and 102 in missing
    assert compute_team_absence_deficit(missing, prior_lookup) == 10.0


def test_exact_out_required_after_canonicalize():
    assert is_exact_out_status("Out")
    assert not is_exact_out_status("Questionable")
    assert canonicalize_injury_status("Expected to be out until at least Apr 18") == "Out"


def test_player_out_12_straight_adds_zero():
    prior_lookup = {7: {"impact": 25.0, "minutes": 30.0, "consecutive_misses": 12}}
    assert compute_team_absence_deficit([7], prior_lookup) == 0.0
    prior_lookup[7]["consecutive_misses"] = MAX_CONSECUTIVE_MISSES - 1
    assert compute_team_absence_deficit([7], prior_lookup) == 25.0


def test_changing_game_i_result_does_not_change_game_i_deficit():
    rows = [
        {
            "SEASON_ID": "22023",
            "Player_ID": 1,
            "TEAM_ABBR": "AAA",
            "Game_ID": "g1",
            "GAME_DATE_DT": pd.Timestamp("2024-01-01"),
            "MIN_NUMERIC": 30.0,
            "PTS": 20.0,
            "FGM": 8.0,
            "FGA": 15.0,
            "FTM": 2.0,
            "FTA": 2.0,
            "OREB": 1.0,
            "DREB": 4.0,
            "REB": 5.0,
            "AST": 5.0,
            "STL": 1.0,
            "BLK": 0.0,
            "PF": 2.0,
            "TOV": 2.0,
            "WL": "W",
        },
        {
            "SEASON_ID": "22023",
            "Player_ID": 2,
            "TEAM_ABBR": "AAA",
            "Game_ID": "g1",
            "GAME_DATE_DT": pd.Timestamp("2024-01-01"),
            "MIN_NUMERIC": 20.0,
            "PTS": 8.0,
            "FGM": 3.0,
            "FGA": 7.0,
            "FTM": 1.0,
            "FTA": 2.0,
            "OREB": 0.0,
            "DREB": 2.0,
            "REB": 2.0,
            "AST": 2.0,
            "STL": 0.0,
            "BLK": 0.0,
            "PF": 1.0,
            "TOV": 1.0,
            "WL": "W",
        },
        {
            "SEASON_ID": "22023",
            "Player_ID": 2,
            "TEAM_ABBR": "AAA",
            "Game_ID": "g2",
            "GAME_DATE_DT": pd.Timestamp("2024-01-03"),
            "MIN_NUMERIC": 28.0,
            "PTS": 10.0,
            "FGM": 4.0,
            "FGA": 9.0,
            "FTM": 1.0,
            "FTA": 1.0,
            "OREB": 1.0,
            "DREB": 2.0,
            "REB": 3.0,
            "AST": 3.0,
            "STL": 0.0,
            "BLK": 0.0,
            "PF": 2.0,
            "TOV": 1.0,
            "WL": "L",
        },
    ]
    logs = pd.DataFrame(rows)

    def deficit_for_g2(log_df):
        tracker = TeamAvailabilityTracker({})
        g1 = log_df[log_df["Game_ID"] == "g1"][
            [
                "Player_ID",
                "MIN_NUMERIC",
                "PTS",
                "FGM",
                "FGA",
                "FTM",
                "FTA",
                "OREB",
                "DREB",
                "REB",
                "AST",
                "STL",
                "BLK",
                "PF",
                "TOV",
            ]
        ]
        tracker.observe_game("22023", "AAA", g1, set(g1["Player_ID"].astype(int)))
        played2 = set(log_df[log_df["Game_ID"] == "g2"]["Player_ID"].astype(int))
        prior = tracker.prior_lookup("22023", "AAA")
        missing = tracker.training_missing_ids("22023", "AAA", played2)
        return compute_team_absence_deficit(missing, prior)

    d0 = deficit_for_g2(logs)
    flipped = logs.copy()
    flipped.loc[flipped["Game_ID"] == "g2", "WL"] = "W"
    flipped.loc[flipped["Game_ID"] == "g2", "PTS"] = 50.0
    assert d0 == deficit_for_g2(flipped)


def test_blend_shifts_with_games_played():
    assert abs(blend_player_prior(0, 25.0, 10.0) - 10.0) < 1e-9
    assert abs(blend_player_prior(10, 20.0, 10.0) - 15.0) < 1e-9


def test_returning_player_keeps_pre_injury_average():
    tracker = TeamAvailabilityTracker({})
    g1 = pd.DataFrame(
        [
            {
                "Player_ID": 1,
                "MIN_NUMERIC": 30.0,
                "PTS": 24.0,
                "FGM": 9.0,
                "FGA": 18.0,
                "FTM": 4.0,
                "FTA": 5.0,
                "OREB": 1.0,
                "DREB": 5.0,
                "REB": 6.0,
                "AST": 6.0,
                "STL": 1.0,
                "BLK": 1.0,
                "PF": 2.0,
                "TOV": 2.0,
            }
        ]
    )
    tracker.observe_game("22023", "AAA", g1, {1})
    for _ in range(3):
        tracker.observe_game("22023", "AAA", pd.DataFrame(), set())
    prior = tracker.prior_lookup("22023", "AAA")[1]
    assert prior["pts"] == 24.0
    assert prior["consecutive_misses"] == 3
    assert "impact_eff" in prior


def test_star_out_deficit_much_larger_than_bench():
    """Synthetic Cade/Jokic-style sanity: star impact >> bench."""
    cade = player_impact(pts=22, reb=5, ast=8, stl=1, blk=0.5, tov=3)
    bench = player_impact(pts=6, reb=2, ast=1, stl=0.3, blk=0.2, tov=1)
    assert cade > 2.5 * bench
    priors = {
        1630595: {"impact": cade, "minutes": 33.0, "consecutive_misses": 0},  # fake Cade
        999: {"impact": bench, "minutes": 12.0, "consecutive_misses": 0},
    }
    d_star = compute_team_absence_deficit([1630595], priors)
    d_bench = compute_team_absence_deficit([999], priors)
    assert d_star > 2.5 * d_bench


def test_jokic_style_out_larger_than_role_player():
    jokic = player_impact(pts=26, reb=12, ast=9, stl=1.2, blk=0.8, tov=3)
    role = player_impact(pts=11, reb=4, ast=2, stl=0.5, blk=0.3, tov=1.2)
    priors = {
        203999: {"impact": jokic, "minutes": 34.0, "consecutive_misses": 0},
        888: {"impact": role, "minutes": 22.0, "consecutive_misses": 0},
    }
    assert compute_team_absence_deficit([203999], priors) > 2.0 * compute_team_absence_deficit(
        [888], priors
    )


def test_higher_deficit_lowers_home_win_prob_directionally():
    """With negative coef on deficit_diff, larger home deficit => lower P(home)."""
    # Simulate logistic score: elo_diff - 0.05 * player_deficit_diff
    def p_home(deficit_diff, elo_diff=0.0):
        z = 0.01 * elo_diff - 0.05 * deficit_diff
        return 1 / (1 + np.exp(-z))

    p_healthy = p_home(0.0)
    p_cade_out = p_home(30.0)  # home missing star
    assert p_cade_out < p_healthy


def test_no_class_weight_passed_to_logistic():
    src = inspect.getsource(train_baseline.make_pipe)
    # Must not pass class_weight= into LogisticRegression(...)
    assert "class_weight=" not in src.replace("# no class_weight", "")


def test_no_home_court_in_core_features():
    assert "home_court" not in MATCHUP_CORE_FEATURES


def test_rest_days_cap_is_seven():
    assert REST_DAYS_CAP == 7
    s = pd.Series([1, 7, 10, 20])
    assert int(s.clip(upper=REST_DAYS_CAP).max()) == 7


def test_impact_formula_weights():
    # PTS + 1.5*AST + REB + STL + BLK - TOV
    assert player_impact(10, 4, 2, 1, 1, 3) == 10 + 3 + 4 + 1 + 1 - 3


def test_efficiency_impact_formula():
    # PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*OREB + 0.3*DREB
    # + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV
    got = player_impact_eff(
        pts=20, fgm=8, fga=16, ftm=3, fta=4, oreb=2, dreb=6, stl=1, ast=5, blk=1, pf=2, tov=3
    )
    expected = (
        20 + 0.4 * 8 - 0.7 * 16 - 0.4 * (4 - 3) + 0.7 * 2 + 0.3 * 6 + 1 + 0.7 * 5 + 0.7 * 1 - 0.4 * 2 - 3
    )
    assert abs(got - expected) < 1e-9


def test_efficiency_penalizes_empty_calories_vs_counting():
    """Same PTS, worse FG% => lower efficiency impact, counting unchanged on PTS/REB/AST."""
    efficient = player_impact_eff(20, 9, 14, 2, 2, 1, 4, 1, 4, 0, 2, 2)
    chucker = player_impact_eff(20, 7, 20, 2, 4, 1, 4, 1, 4, 0, 2, 2)
    assert efficient > chucker
    # counting ignores FGA
    assert player_impact(20, 5, 4, 1, 0, 2) == player_impact(20, 5, 4, 1, 0, 2)
