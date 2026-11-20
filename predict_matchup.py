"""
Tonight's predictions with the production logistic matchup model.

Builds one row per game (home vs away), using pre-game Elo, schedule
context, rolling form, and injury-based missing-starter minutes.
"""

from __future__ import annotations

import datetime
import json
import os
import sqlite3
import unicodedata

import joblib
import numpy as np
import pandas as pd

from nba_ml.features import (
    LOOKBACK,
    MATCHUP_FEATURE_COLUMNS,
    STARTER_MIN_THRESHOLD,
)

DB_NAME = "nba_stats.db"
MODELS_DIR = "models"
HISTORY_DIR = "history"
DATA_DIR = "artifacts"


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return "".join(
        c
        for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    ).lower().strip()


def get_team_name(team_id):
    try:
        from nba_api.stats.static import teams

        info = teams.find_team_name_by_id(team_id)
        return info["full_name"] if info else f"Team {team_id}"
    except Exception:
        return f"Team {team_id}"


def team_abbr_from_id(team_id, rosters: pd.DataFrame) -> str | None:
    # Prefer roster join via player logs / static teams
    try:
        from nba_api.stats.static import teams

        info = teams.find_team_name_by_id(int(team_id))
        return info["abbreviation"] if info else None
    except Exception:
        return None


def load_injured_names():
    try:
        injuries = pd.read_csv("injuries.csv")
        return set(injuries["Player"].map(normalize_name))
    except Exception:
        print("  [WARNING] injuries.csv missing — assuming no injuries.")
        return set()


def load_schedule_ctx():
    try:
        ctx = pd.read_csv("schedule_context.csv")
        return {
            int(r.TEAM_ID): {
                "rest_days": int(r.rest_days),
                "is_back_to_back": int(r.is_back_to_back),
                "games_last_7": int(r.games_last_7),
            }
            for r in ctx.itertuples()
        }
    except Exception:
        return {}


def rolling_form_from_db(conn, team_abbr: str) -> dict:
    """Leak-safe-ish recent form from the last LOOKBACK games in SQLite."""
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


def missing_starter_minutes_tonight(
    team_id, team_abbr, rosters, injured_names, conn
) -> float:
    roster = rosters[rosters["TeamID"] == team_id][["PLAYER_ID", "PLAYER"]]
    if roster.empty:
        return 0.0
    ids = ",".join(str(int(x)) for x in roster["PLAYER_ID"].tolist())
    logs = pd.read_sql(
        f"""
        SELECT Player_ID, MIN, GAME_DATE
        FROM player_logs
        WHERE Player_ID IN ({ids})
        ORDER BY GAME_DATE DESC
        """,
        conn,
    )
    if logs.empty:
        return 0.0
    logs["MIN_NUMERIC"] = pd.to_numeric(logs["MIN"], errors="coerce").fillna(0)
    recent = (
        logs.groupby("Player_ID")["MIN_NUMERIC"]
        .apply(lambda s: s.head(15).mean())
    )
    starters = recent[recent >= STARTER_MIN_THRESHOLD]
    missing = 0.0
    for _, row in roster.iterrows():
        pid = row["PLAYER_ID"]
        if normalize_name(row["PLAYER"]) in injured_names and pid in starters.index:
            missing += float(starters[pid])
    return missing


def main():
    print("--- PREDICTING TONIGHT (logistic matchup) ---")
    model_path = os.path.join(MODELS_DIR, "logistic_baseline.pkl")
    elo_path = os.path.join(MODELS_DIR, "elo_ratings.json")

    if not os.path.exists(model_path):
        print(f"[FAIL] Missing {model_path}. Run train_baseline.py first.")
        return
    if not os.path.exists(elo_path):
        print(f"[FAIL] Missing {elo_path}. Run build_datasets.py first.")
        return

    model = joblib.load(model_path)
    with open(elo_path, encoding="utf-8") as f:
        elo = json.load(f)

    games = pd.read_csv("todays_games.csv")
    rosters = pd.read_csv("todays_rosters.csv")
    injured = load_injured_names()
    schedule_ctx = load_schedule_ctx()
    conn = sqlite3.connect(DB_NAME)

    preds = []
    for _, g in games.iterrows():
        home_id, away_id = int(g["HOME_TEAM_ID"]), int(g["VISITOR_TEAM_ID"])
        home_abbr = team_abbr_from_id(home_id, rosters)
        away_abbr = team_abbr_from_id(away_id, rosters)
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
        hm = missing_starter_minutes_tonight(
            home_id, home_abbr, rosters, injured, conn
        )
        am = missing_starter_minutes_tonight(
            away_id, away_abbr, rosters, injured, conn
        )

        row = {
            "elo_diff": home_elo - away_elo,
            "rest_diff": hs["rest_days"] - as_["rest_days"],
            "b2b_diff": hs["is_back_to_back"] - as_["is_back_to_back"],
            "g7_diff": hs["games_last_7"] - as_["games_last_7"],
            "missing_starter_diff": hm - am,
            "form_pts_diff": hf["form_pts"] - af["form_pts"],
            "form_reb_diff": hf["form_reb"] - af["form_reb"],
            "form_ast_diff": hf["form_ast"] - af["form_ast"],
            "form_pm_diff": hf["form_pm"] - af["form_pm"],
            "home_court": 1,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "home_rest": hs["rest_days"],
            "away_rest": as_["rest_days"],
            "home_b2b": hs["is_back_to_back"],
            "away_b2b": as_["is_back_to_back"],
            "home_g7": hs["games_last_7"],
            "away_g7": as_["games_last_7"],
            "home_missing_starter_min": hm,
            "away_missing_starter_min": am,
        }
        # Model was trained on MATCHUP_FEATURE_COLUMNS only
        X = pd.DataFrame([{c: row[c] for c in MATCHUP_FEATURE_COLUMNS}])
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
