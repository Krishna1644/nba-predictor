"""
Leak-safe team-game and matchup feature engineering.

Rules:
- Elo / opponent strength use only pre-game information.
- Starter status uses expanding mean minutes from *prior* games only.
- Schedule fatigue uses only prior game dates.
- Same-night box scores are labels / lookback inputs, never future leakage.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

STARTER_MIN_THRESHOLD = 25.0
MIN_PRIOR_GAMES_FOR_STARTER = 3
ELO_K = 20
SEASON_CARRY = 0.75
MODELS_DIR = "models"
LOOKBACK = 10

FEATURE_COLUMNS = [
    "team_pts",
    "team_reb",
    "team_ast",
    "team_stl",
    "team_blk",
    "team_tov",
    "max_pts",
    "max_reb",
    "max_ast",
    "fg_pct",
    "fg3_pct",
    "ft_pct",
    "efg_pct",
    "ts_pct",
    "tov_pct",
    "plus_minus",
    "is_home",
    "rest_days",
    "is_back_to_back",
    "games_last_7",
    "missing_starter_minutes",
    "team_elo",
    "opp_win_pct",
    "opp_pts_allowed_avg",
]

assert len(FEATURE_COLUMNS) == 24

# Pre-game matchup features for logistic baseline (home-centric).
MATCHUP_FEATURE_COLUMNS = [
    "elo_diff",
    "rest_diff",
    "b2b_diff",
    "g7_diff",
    "missing_starter_diff",
    "form_pts_diff",
    "form_reb_diff",
    "form_ast_diff",
    "form_pm_diff",
    "home_court",
]

# Seasons: train through 2023-24, hold out 2024-25, reserve 2025-26 for live.
TRAIN_SEASON_MAX = "22023"  # 2023-24
TEST_SEASON = "22024"  # 2024-25
LIVE_SEASON = "22025"  # 2025-26


def compute_advanced_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "EFG_PCT" not in out.columns:
        out["EFG_PCT"] = np.where(
            out["FGA"] > 0,
            (out["FGM"] + 0.5 * out["FG3M"]) / out["FGA"],
            0.0,
        )
    if "TS_PCT" not in out.columns:
        tsa = 2 * (out["FGA"] + 0.44 * out["FTA"])
        out["TS_PCT"] = np.where(tsa > 0, out["PTS"] / tsa, 0.0)
    if "TOV_PCT" not in out.columns:
        tov_denom = out["FGA"] + 0.44 * out["FTA"] + out["TOV"]
        out["TOV_PCT"] = np.where(tov_denom > 0, out["TOV"] / tov_denom, 0.0)
    return out


def extract_team_from_matchup(matchup) -> str:
    if isinstance(matchup, str):
        return matchup.split(" ")[0].strip()
    return ""


def add_leak_safe_prior_minutes(player_logs: pd.DataFrame) -> pd.DataFrame:
    """Expanding mean minutes per (season, team, player), excluding current game."""
    logs = player_logs.sort_values(
        ["SEASON_ID", "TEAM_ABBR", "Player_ID", "GAME_DATE_DT", "Game_ID"]
    ).copy()
    logs["MIN_NUMERIC"] = pd.to_numeric(logs["MIN"], errors="coerce").fillna(0.0)
    group = logs.groupby(["SEASON_ID", "TEAM_ABBR", "Player_ID"], sort=False)
    logs["prior_games"] = group.cumcount()
    logs["prior_min_sum"] = group["MIN_NUMERIC"].cumsum() - logs["MIN_NUMERIC"]
    logs["prior_avg_min"] = np.where(
        logs["prior_games"] > 0,
        logs["prior_min_sum"] / logs["prior_games"],
        np.nan,
    )
    logs["is_prior_starter"] = (
        (logs["prior_games"] >= MIN_PRIOR_GAMES_FOR_STARTER)
        & (logs["prior_avg_min"] >= STARTER_MIN_THRESHOLD)
    )
    return logs


def compute_missing_starter_minutes(logs: pd.DataFrame, team_games: pd.DataFrame) -> np.ndarray:
    """
    Chronological, leak-safe missing starter minutes.

    Maintain the latest post-game expanding mean for anyone who has qualified
    as a starter for that team-season; at each game, sum those values for
    starters who do not appear in the box score. State is updated only after
    the game is scored.
    """
    played_map = {
        key: set(group["Player_ID"].tolist())
        for key, group in logs.groupby(["Game_ID", "TEAM_ABBR"])
    }
    logs_by_gt = {
        key: group[["Player_ID", "MIN_NUMERIC", "prior_games", "prior_avg_min"]]
        for key, group in logs.groupby(["Game_ID", "TEAM_ABBR"])
    }

    # Active starters keyed by (season, team) -> {player_id: avg_min}
    active = defaultdict(dict)
    missing_vals = {}

    ordered = team_games.sort_values(["game_date", "Game_ID", "TEAM_ABBR"])
    for row in ordered.itertuples():
        key = (row.Game_ID, row.TEAM_ABBR)
        season, team = row.season_id, row.TEAM_ABBR
        played = played_map.get(key, set())
        roster = active[(season, team)]

        missing_vals[key] = float(
            sum(avg for pid, avg in roster.items() if pid not in played)
        )

        game_logs = logs_by_gt.get(key)
        if game_logs is None:
            continue
        for pref in game_logs.itertuples(index=False):
            pid = pref.Player_ID
            mins = float(pref.MIN_NUMERIC)
            prev_n = int(pref.prior_games)
            prev_avg = pref.prior_avg_min
            if prev_n <= 0 or pd.isna(prev_avg):
                new_avg, new_n = mins, 1
            else:
                new_avg = (float(prev_avg) * prev_n + mins) / (prev_n + 1)
                new_n = prev_n + 1

            if new_n >= MIN_PRIOR_GAMES_FOR_STARTER and new_avg >= STARTER_MIN_THRESHOLD:
                roster[pid] = new_avg
            elif pid in roster and new_n >= MIN_PRIOR_GAMES_FOR_STARTER:
                del roster[pid]

    return np.array(
        [missing_vals.get((r.Game_ID, r.TEAM_ABBR), 0.0) for r in team_games.itertuples()],
        dtype=np.float64,
    )


def compute_elo_and_strength(team_games: pd.DataFrame, models_dir: str = MODELS_DIR):
    """Pre-game Elo, opp win%, and opp pts-allowed for each team-game."""
    elo = {}
    season_wins = {}
    season_games_n = {}
    season_pts_allowed = {}
    current_season = None

    game_order = (
        team_games.drop_duplicates("Game_ID")
        .sort_values("game_date")["Game_ID"]
        .values
    )
    game_groups = team_games.groupby("Game_ID")

    game_elo, game_opp_wp, game_opp_def = {}, {}, {}

    for gid in game_order:
        group = game_groups.get_group(gid)
        if len(group) != 2:
            continue

        row_a, row_b = group.iloc[0], group.iloc[1]
        team_a, team_b = row_a["TEAM_ABBR"], row_b["TEAM_ABBR"]
        season = row_a["season_id"]
        win_a = float(row_a["win"])

        if season != current_season:
            current_season = season
            for t in list(elo.keys()):
                elo[t] = SEASON_CARRY * elo[t] + (1 - SEASON_CARRY) * 1500
            season_wins.clear()
            season_games_n.clear()
            season_pts_allowed.clear()

        for t in (team_a, team_b):
            elo.setdefault(t, 1500.0)
            season_wins.setdefault(t, 0)
            season_games_n.setdefault(t, 0)
            season_pts_allowed.setdefault(t, 0.0)

        game_elo[(gid, team_a)] = elo[team_a]
        game_elo[(gid, team_b)] = elo[team_b]
        game_opp_wp[(gid, team_a)] = season_wins[team_b] / max(season_games_n[team_b], 1)
        game_opp_wp[(gid, team_b)] = season_wins[team_a] / max(season_games_n[team_a], 1)
        game_opp_def[(gid, team_a)] = season_pts_allowed[team_b] / max(
            season_games_n[team_b], 1
        )
        game_opp_def[(gid, team_b)] = season_pts_allowed[team_a] / max(
            season_games_n[team_a], 1
        )

        e_a = 1 / (1 + 10 ** ((elo[team_b] - elo[team_a]) / 400))
        elo[team_a] += ELO_K * (win_a - e_a)
        elo[team_b] += ELO_K * ((1 - win_a) - (1 - e_a))

        season_games_n[team_a] += 1
        season_games_n[team_b] += 1
        if win_a:
            season_wins[team_a] += 1
        else:
            season_wins[team_b] += 1

        pts_a, pts_b = float(row_a["team_pts"]), float(row_b["team_pts"])
        season_pts_allowed[team_a] += pts_b
        season_pts_allowed[team_b] += pts_a

    out = team_games.copy()
    out["team_elo"] = out.apply(
        lambda r: game_elo.get((r["Game_ID"], r["TEAM_ABBR"]), 1500.0), axis=1
    )
    out["opp_win_pct"] = out.apply(
        lambda r: game_opp_wp.get((r["Game_ID"], r["TEAM_ABBR"]), 0.5), axis=1
    )
    out["opp_pts_allowed_avg"] = out.apply(
        lambda r: game_opp_def.get((r["Game_ID"], r["TEAM_ABBR"]), 105.0), axis=1
    )

    game_context = {
        (gid, team): {
            "team_elo": float(e),
            "opp_win_pct": float(game_opp_wp.get((gid, team), 0.5)),
            "opp_pts_allowed_avg": float(game_opp_def.get((gid, team), 105.0)),
        }
        for (gid, team), e in game_elo.items()
    }

    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "elo_ratings.json"), "w", encoding="utf-8") as f:
        json.dump({k: round(v, 1) for k, v in elo.items()}, f, indent=2)
    joblib.dump(game_context, os.path.join(models_dir, "game_context.pkl"))
    return out, elo, game_context


def build_team_game_features(player_logs: pd.DataFrame, models_dir: str = MODELS_DIR):
    """Aggregate player logs into one leak-safe feature row per team-game."""
    print("  Aggregating player stats to team-game level (leak-safe)...")

    logs = player_logs.copy()
    logs["TEAM_ABBR"] = logs["MATCHUP"].apply(extract_team_from_matchup)
    logs["IS_HOME"] = logs["MATCHUP"].apply(lambda x: 1 if "vs." in str(x) else 0)
    logs["GAME_DATE_DT"] = pd.to_datetime(logs["GAME_DATE"], format="mixed")
    logs["WIN"] = (logs["WL"] == "W").astype(int)
    logs = add_leak_safe_prior_minutes(logs)

    team_games = logs.groupby(["Game_ID", "TEAM_ABBR"], as_index=False).agg(
        team_pts=("PTS", "sum"),
        team_reb=("REB", "sum"),
        team_ast=("AST", "sum"),
        team_stl=("STL", "sum"),
        team_blk=("BLK", "sum"),
        team_tov=("TOV", "sum"),
        max_pts=("PTS", "max"),
        max_reb=("REB", "max"),
        max_ast=("AST", "max"),
        fg_pct=("FG_PCT", "mean"),
        fg3_pct=("FG3_PCT", "mean"),
        ft_pct=("FT_PCT", "mean"),
        efg_pct=("EFG_PCT", "mean"),
        ts_pct=("TS_PCT", "mean"),
        tov_pct=("TOV_PCT", "mean"),
        plus_minus=("PLUS_MINUS", "mean"),
        is_home=("IS_HOME", "first"),
        win=("WIN", "first"),
        game_date=("GAME_DATE_DT", "first"),
        season_id=("SEASON_ID", "first"),
    )

    opp_info = team_games[["Game_ID", "TEAM_ABBR", "team_pts"]].rename(
        columns={"TEAM_ABBR": "OPP_ABBR", "team_pts": "opponent_pts"}
    )
    team_games = team_games.merge(opp_info, on="Game_ID", how="left")
    team_games = team_games[team_games["TEAM_ABBR"] != team_games["OPP_ABBR"]].copy()
    team_games["mov"] = team_games["team_pts"] - team_games["opponent_pts"]

    print("  Computing leak-safe missing_starter_minutes...")
    team_games["missing_starter_minutes"] = compute_missing_starter_minutes(
        logs, team_games
    )

    print("  Computing schedule context...")
    team_games = team_games.sort_values(["TEAM_ABBR", "game_date"]).reset_index(drop=True)
    team_games["prev_game_date"] = team_games.groupby("TEAM_ABBR")["game_date"].shift(1)
    team_games["rest_days"] = (
        team_games["game_date"] - team_games["prev_game_date"]
    ).dt.days.fillna(3).astype(int)
    team_games["is_back_to_back"] = (team_games["rest_days"] == 1).astype(int)

    games_last_7 = np.zeros(len(team_games), dtype=int)
    for team_abbr, idx in team_games.groupby("TEAM_ABBR").groups.items():
        idx = list(idx)
        dates = team_games.loc[idx, "game_date"].tolist()
        for i, row_idx in enumerate(idx):
            current = dates[i]
            seven_ago = current - timedelta(days=7)
            games_last_7[row_idx] = sum(1 for d in dates[:i] if d >= seven_ago)
    team_games["games_last_7"] = games_last_7
    team_games = team_games.drop(columns=["prev_game_date"], errors="ignore")

    # Rolling prior form (shift 1 so current game excluded)
    print("  Computing leak-safe rolling form...")
    for col, out_col in [
        ("team_pts", "form_pts"),
        ("team_reb", "form_reb"),
        ("team_ast", "form_ast"),
        ("plus_minus", "form_pm"),
    ]:
        team_games[out_col] = (
            team_games.groupby("TEAM_ABBR")[col]
            .transform(lambda s: s.shift(1).rolling(LOOKBACK, min_periods=3).mean())
            .fillna(team_games[col].mean())
        )

    print("  Computing Elo ratings and opponent strength...")
    team_games, current_elo, _ = compute_elo_and_strength(
        team_games, models_dir=models_dir
    )
    team_games = team_games.drop(columns=["OPP_ABBR"], errors="ignore")
    print(f"  Built {len(team_games)} team-game rows.")
    return team_games, current_elo


def build_matchup_frame(team_games: pd.DataFrame) -> pd.DataFrame:
    """One row per game; features are home-minus-away / home-centric pre-game."""
    home = team_games[team_games["is_home"] == 1][
        [
            "Game_ID",
            "TEAM_ABBR",
            "season_id",
            "game_date",
            "win",
            "team_elo",
            "rest_days",
            "is_back_to_back",
            "games_last_7",
            "missing_starter_minutes",
            "form_pts",
            "form_reb",
            "form_ast",
            "form_pm",
        ]
    ].rename(
        columns={
            "TEAM_ABBR": "home_team",
            "win": "home_win",
            "team_elo": "home_elo",
            "rest_days": "home_rest",
            "is_back_to_back": "home_b2b",
            "games_last_7": "home_g7",
            "missing_starter_minutes": "home_missing_starter_min",
            "form_pts": "home_form_pts",
            "form_reb": "home_form_reb",
            "form_ast": "home_form_ast",
            "form_pm": "home_form_pm",
        }
    )
    away = team_games[team_games["is_home"] == 0][
        [
            "Game_ID",
            "TEAM_ABBR",
            "team_elo",
            "rest_days",
            "is_back_to_back",
            "games_last_7",
            "missing_starter_minutes",
            "form_pts",
            "form_reb",
            "form_ast",
            "form_pm",
        ]
    ].rename(
        columns={
            "TEAM_ABBR": "away_team",
            "team_elo": "away_elo",
            "rest_days": "away_rest",
            "is_back_to_back": "away_b2b",
            "games_last_7": "away_g7",
            "missing_starter_minutes": "away_missing_starter_min",
            "form_pts": "away_form_pts",
            "form_reb": "away_form_reb",
            "form_ast": "away_form_ast",
            "form_pm": "away_form_pm",
        }
    )

    m = home.merge(away, on="Game_ID", how="inner")
    m["elo_diff"] = m["home_elo"] - m["away_elo"]
    m["rest_diff"] = m["home_rest"] - m["away_rest"]
    m["b2b_diff"] = m["home_b2b"] - m["away_b2b"]
    m["g7_diff"] = m["home_g7"] - m["away_g7"]
    m["missing_starter_diff"] = (
        m["home_missing_starter_min"] - m["away_missing_starter_min"]
    )
    m["form_pts_diff"] = m["home_form_pts"] - m["away_form_pts"]
    m["form_reb_diff"] = m["home_form_reb"] - m["away_form_reb"]
    m["form_ast_diff"] = m["home_form_ast"] - m["away_form_ast"]
    m["form_pm_diff"] = m["home_form_pm"] - m["away_form_pm"]
    m["home_court"] = 1
    return m.sort_values("game_date").reset_index(drop=True)


def build_sequences(team_games: pd.DataFrame, lookback: int = LOOKBACK):
    """Rolling lookback windows; y is win label of the target game."""
    X_sequences, y_labels, meta = [], [], []

    for team_abbr, team_df in team_games.groupby("TEAM_ABBR"):
        team_df = team_df.sort_values("game_date").reset_index(drop=True)
        if len(team_df) < lookback + 1:
            continue
        feature_matrix = team_df[FEATURE_COLUMNS].values
        labels = team_df["win"].values
        for i in range(lookback, len(team_df)):
            X_sequences.append(feature_matrix[i - lookback : i])
            y_labels.append(labels[i])
            meta.append(
                {
                    "Game_ID": team_df.loc[i, "Game_ID"],
                    "TEAM_ABBR": team_abbr,
                    "game_date": team_df.loc[i, "game_date"],
                    "season_id": team_df.loc[i, "season_id"],
                    "is_home": int(team_df.loc[i, "is_home"]),
                }
            )

    X = np.array(X_sequences, dtype=np.float32)
    y = np.array(y_labels, dtype=np.float32)
    meta_df = pd.DataFrame(meta)
    return X, y, meta_df
