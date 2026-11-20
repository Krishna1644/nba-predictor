"""
Tonight's predictions with the production logistic matchup model.

Team absence deficit always goes through
`nba_ml.features.compute_team_absence_deficit` — same function as training.
"""

from __future__ import annotations

import datetime
import json
import os
import sqlite3

import joblib
import pandas as pd

from nba_ml.features import LOOKBACK, REST_DAYS_CAP, compute_team_absence_deficit
from nba_ml.player_priors import (
    annotate_consecutive_misses,
    build_prior_lookup_from_db,
    serving_missing_ids_from_injuries,
)

DB_NAME = "nba_stats.db"
MODELS_DIR = "models"
HISTORY_DIR = "history"
LIVE_SEASON_ID = "22026"


def get_team_name(team_id):
    try:
        from nba_api.stats.static import teams

        info = teams.find_team_name_by_id(team_id)
        return info["full_name"] if info else f"Team {team_id}"
    except Exception:
        return f"Team {team_id}"


def team_abbr_from_id(team_id) -> str | None:
    try:
        from nba_api.stats.static import teams

        info = teams.find_team_name_by_id(int(team_id))
        return info["abbreviation"] if info else None
    except Exception:
        return None


def load_schedule_ctx():
    try:
        ctx = pd.read_csv("schedule_context.csv")
        out = {}
        for r in ctx.itertuples():
            rest = int(min(int(r.rest_days), REST_DAYS_CAP))
            out[int(r.TEAM_ID)] = {
                "rest_days": rest,
                "is_back_to_back": int(r.is_back_to_back),
                "games_last_7": int(r.games_last_7),
            }
        return out
    except Exception:
        return {}


def rolling_form_from_db(conn, team_abbr: str) -> dict:
    q = """
        SELECT Game_ID, GAME_DATE,
               SUM(PTS) as team_pts, SUM(REB) as team_reb, SUM(AST) as team_ast,
               AVG(PLUS_MINUS) as plus_minus
        FROM player_logs
        WHERE MATCHUP LIKE ?
        GROUP BY Game_ID, GAME_DATE
        ORDER BY GAME_DATE DESC
        LIMIT ?
    """
    hist = pd.read_sql(q, conn, params=(f"{team_abbr} %", LOOKBACK))
    if hist.empty:
        return {"form_pts": 110.0, "form_reb": 44.0, "form_ast": 25.0, "form_pm": 0.0}
    return {
        "form_pts": float(hist["team_pts"].mean()),
        "form_reb": float(hist["team_reb"].mean()),
        "form_ast": float(hist["team_ast"].mean()),
        "form_pm": float(hist["plus_minus"].mean()),
    }


def resolve_season_id(conn) -> str:
    rows = pd.read_sql(
        "SELECT SEASON_ID, MAX(GAME_DATE) mx FROM player_logs GROUP BY SEASON_ID ORDER BY mx DESC",
        conn,
    )
    if rows.empty:
        return LIVE_SEASON_ID
    ids = set(rows["SEASON_ID"].astype(str))
    if LIVE_SEASON_ID in ids:
        return LIVE_SEASON_ID
    return str(rows.iloc[0]["SEASON_ID"])


def team_absence_deficit_for_serve(
    conn,
    season_id: str,
    team_id: int,
    team_abbr: str,
    rosters,
    injuries,
    min_minutes: float,
    value_key: str = "impact",
) -> float:
    """Build missing list + priors, then call the shared deficit function."""
    roster = rosters[rosters["TeamID"] == team_id]
    if roster.empty:
        return 0.0
    pids = [int(x) for x in roster["PLAYER_ID"].tolist()]
    prior_lookup = build_prior_lookup_from_db(conn, season_id, pids)
    prior_lookup = annotate_consecutive_misses(
        conn, season_id, team_abbr, prior_lookup
    )
    adjusted = {
        pid: {**p, "minutes": p["minutes"] if p["minutes"] >= min_minutes else 0.0}
        for pid, p in prior_lookup.items()
    }
    missing_ids = serving_missing_ids_from_injuries(injuries, roster, adjusted)
    return compute_team_absence_deficit(
        missing_ids, prior_lookup, value_key=value_key
    )


def main():
    print("--- PREDICTING TONIGHT (logistic matchup) ---")
    model_path = os.path.join(MODELS_DIR, "logistic_baseline.pkl")
    elo_path = os.path.join(MODELS_DIR, "elo_ratings.json")
    cols_path = os.path.join(MODELS_DIR, "logistic_feature_columns.json")

    if not os.path.exists(model_path):
        print(f"[FAIL] Missing {model_path}. Run train_baseline.py first.")
        return
    if not os.path.exists(elo_path):
        print(f"[FAIL] Missing {elo_path}. Run build_datasets.py first.")
        return

    model = joblib.load(model_path)
    with open(elo_path, encoding="utf-8") as f:
        elo = json.load(f)
    value_key = "impact"
    if os.path.exists(cols_path):
        with open(cols_path, encoding="utf-8") as f:
            meta = json.load(f)
            feature_cols = meta["columns"]
            value_key = meta.get("value_key", "impact")
    else:
        from nba_ml.features import ABSENCE_FEATURE_PLAYER, MATCHUP_CORE_FEATURES

        feature_cols = MATCHUP_CORE_FEATURES + [ABSENCE_FEATURE_PLAYER]

    games = pd.read_csv("todays_games.csv")
    rosters = pd.read_csv("todays_rosters.csv")
    try:
        injuries = pd.read_csv("injuries.csv")
    except Exception:
        print("  [WARNING] injuries.csv missing — deficits will be 0.")
        injuries = pd.DataFrame()
    schedule_ctx = load_schedule_ctx()
    conn = sqlite3.connect(DB_NAME)
    season_id = resolve_season_id(conn)
    print(f"  Using season_id={season_id} for player priors")

    preds = []
    for _, g in games.iterrows():
        home_id, away_id = int(g["HOME_TEAM_ID"]), int(g["VISITOR_TEAM_ID"])
        home_abbr = team_abbr_from_id(home_id)
        away_abbr = team_abbr_from_id(away_id)
        if not home_abbr or not away_abbr:
            print(f"  [SKIP] Could not resolve team abbrs for {home_id}/{away_id}")
            continue

        home_name = get_team_name(home_id)
        away_name = get_team_name(away_id)
        home_elo = float(elo.get(home_abbr, 1500))
        away_elo = float(elo.get(away_abbr, 1500))

        hs = schedule_ctx.get(
            home_id, {"rest_days": 2, "is_back_to_back": 0, "games_last_7": 3}
        )
        as_ = schedule_ctx.get(
            away_id, {"rest_days": 2, "is_back_to_back": 0, "games_last_7": 3}
        )
        hf = rolling_form_from_db(conn, home_abbr)
        af = rolling_form_from_db(conn, away_abbr)
        hm = team_absence_deficit_for_serve(
            conn, season_id, home_id, home_abbr, rosters, injuries, 10.0, value_key
        )
        am = team_absence_deficit_for_serve(
            conn, season_id, away_id, away_abbr, rosters, injuries, 10.0, value_key
        )
        deficit_diff = hm - am

        row = {
            "elo_diff": home_elo - away_elo,
            "rest_diff": hs["rest_days"] - as_["rest_days"],
            "b2b_diff": hs["is_back_to_back"] - as_["is_back_to_back"],
            "g7_diff": hs["games_last_7"] - as_["games_last_7"],
            "player_deficit_diff": deficit_diff,
            "player_deficit_eff_diff": deficit_diff,
            "player_deficit_minutes_diff": deficit_diff,
            "missing_starter_diff": deficit_diff,
            "form_pts_diff": hf["form_pts"] - af["form_pts"],
            "form_reb_diff": hf["form_reb"] - af["form_reb"],
            "form_ast_diff": hf["form_ast"] - af["form_ast"],
            "form_pm_diff": hf["form_pm"] - af["form_pm"],
        }

        X = pd.DataFrame([{c: row[c] for c in feature_cols}])
        home_prob = float(model.predict_proba(X)[0, 1])
        away_prob = 1.0 - home_prob
        winner = home_name if home_prob >= away_prob else away_name
        conf = max(home_prob, away_prob)

        print(f"  {away_name} @ {home_name}: home {home_prob:.1%} -> {winner}")
        preds.append(
            {
                "GAME_ID": g["GAME_ID"],
                "Home_Team": home_name,
                "Away_Team": away_name,
                "Home_Win_Prob": home_prob,
                "Away_Win_Prob": away_prob,
                "Predicted_Winner": winner,
                "Confidence": conf,
                "Model": "logistic_matchup",
                "Home_Player_Deficit": hm,
                "Away_Player_Deficit": am,
            }
        )

    conn.close()
    if not preds:
        print("[FAIL] No predictions generated.")
        return

    out = pd.DataFrame(preds)
    out.to_csv("final_predictions.csv", index=False)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    stamp = datetime.date.today().isoformat()
    out.to_csv(os.path.join(HISTORY_DIR, f"preds_{stamp}.csv"), index=False)
    print(f"\n[SUCCESS] Wrote final_predictions.csv ({len(out)} games)")


if __name__ == "__main__":
    main()
