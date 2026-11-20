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

from nba_ml.absence import canonicalize_injury_status, is_exact_out_status
from nba_ml.features import LOOKBACK, REST_DAYS_CAP, compute_team_absence_deficit
from nba_ml.player_priors import (
    annotate_consecutive_misses,
    build_prior_lookup_from_db,
    match_name_key,
    serving_missing_ids_from_injuries,
)

DB_NAME = "nba_stats.db"
MODELS_DIR = "models"
HISTORY_DIR = "history"
LIVE_SEASON_ID = "22026"


def get_team_name(team_id):
    from nba_ml.team_ids import team_name

    return team_name(int(team_id))


def team_abbr_from_id(team_id) -> str | None:
    from nba_ml.team_ids import team_abbr

    return team_abbr(int(team_id))


def load_schedule_ctx():
    for path in (
        "schedule_context.csv",
        os.path.join("data", "live", "schedule_context.csv"),
    ):
        try:
            if not os.path.exists(path):
                continue
            ctx = pd.read_csv(path)
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
            continue
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


def injuries_for_tonight(
    injuries: pd.DataFrame,
    team_ids: list[int],
    rosters: pd.DataFrame,
) -> pd.DataFrame:
    """Keep Out/injury rows for teams playing tonight only."""
    if injuries is None or injuries.empty:
        return pd.DataFrame()
    tonight = set(int(t) for t in team_ids)
    if "TeamID" in injuries.columns and injuries["TeamID"].notna().any():
        return injuries[injuries["TeamID"].apply(lambda x: pd.notna(x) and int(x) in tonight)].copy()
    # No team column (CBS): restrict to names on tonight's rosters
    if rosters is None or rosters.empty or "PLAYER" not in rosters.columns:
        return injuries.copy()
    keys = {match_name_key(str(p)) for p in rosters["PLAYER"].tolist()}
    return injuries[injuries["Player"].map(lambda n: match_name_key(str(n)) in keys)].copy()


def team_absence_deficit_for_serve(
    conn,
    season_id: str,
    team_id: int,
    team_abbr: str,
    rosters,
    injuries,
    min_minutes: float,
    value_key: str = "impact",
    unmatched_out: list[str] | None = None,
) -> float:
    """Build missing list + priors, then call the shared deficit function."""
    roster = rosters[rosters["TeamID"] == team_id]
    if roster.empty:
        return 0.0
    # Prefer team-scoped injuries when TeamID is present
    team_inj = injuries
    if injuries is not None and not injuries.empty and "TeamID" in injuries.columns:
        team_inj = injuries[
            injuries["TeamID"].apply(lambda x: pd.notna(x) and int(x) == int(team_id))
        ]
    pids = [int(x) for x in roster["PLAYER_ID"].tolist()]
    prior_lookup = build_prior_lookup_from_db(conn, season_id, pids)
    # Track consecutive misses on the LIVE season only — never inflate from prior year
    prior_lookup = annotate_consecutive_misses(
        conn,
        season_id,
        team_abbr,
        prior_lookup,
        miss_season_id=LIVE_SEASON_ID,
    )
    adjusted = {
        pid: {**p, "minutes": p["minutes"] if p["minutes"] >= min_minutes else 0.0}
        for pid, p in prior_lookup.items()
    }
    missing_ids = serving_missing_ids_from_injuries(
        team_inj, roster, adjusted, unmatched_out=unmatched_out
    )
    return compute_team_absence_deficit(
        missing_ids, prior_lookup, value_key=value_key
    )


def biggest_feature_reason(row: dict, coefs: dict, home_name: str, away_name: str) -> str:
    if not coefs:
        return "Model blend of Elo, rest, form, and absences."
    scored = []
    for feat, coef in coefs.items():
        if feat not in row:
            continue
        contrib = float(coef) * float(row[feat])
        scored.append((abs(contrib), feat, contrib, row[feat]))
    if not scored:
        return "Model blend of Elo, rest, form, and absences."
    scored.sort(reverse=True)
    _, feat, contrib, raw = scored[0]
    favors_home = contrib > 0
    side = home_name if favors_home else away_name
    labels = {
        "elo_diff": f"Elo edge for {side}",
        "rest_diff": f"Rest edge for {side}",
        "b2b_diff": f"Back-to-back load favors {side}",
        "g7_diff": f"Schedule density favors {side}",
        "player_deficit_eff_diff": (
            f"Absence hole larger for {away_name if favors_home else home_name}"
            if abs(float(raw)) > 0.05
            else f"Absence differential favors {side}"
        ),
        "player_deficit_diff": f"Absence differential favors {side}",
        "form_pts_diff": f"Recent scoring form favors {side}",
        "form_reb_diff": f"Recent rebound form favors {side}",
        "form_ast_diff": f"Recent assist form favors {side}",
        "form_pm_diff": f"Recent plus/minus favors {side}",
    }
    return labels.get(feat, f"{feat} favors {side}")


def main():
    print("--- PREDICTING TONIGHT (logistic matchup) ---")
    model_path = os.path.join(MODELS_DIR, "logistic_baseline.pkl")
    elo_path = os.path.join(MODELS_DIR, "elo_ratings.json")
    cols_path = os.path.join(MODELS_DIR, "logistic_feature_columns.json")
    coef_path = os.path.join(MODELS_DIR, "logistic_coefficients.json")

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
    calib = "none"
    if os.path.exists(cols_path):
        with open(cols_path, encoding="utf-8") as f:
            meta = json.load(f)
            feature_cols = meta["columns"]
            value_key = meta.get("value_key", "impact")
            calib = meta.get("calibration", "none")
    else:
        from nba_ml.features import ABSENCE_FEATURE_PLAYER, MATCHUP_CORE_FEATURES

        feature_cols = MATCHUP_CORE_FEATURES + [ABSENCE_FEATURE_PLAYER]
    coefs = {}
    if os.path.exists(coef_path):
        with open(coef_path, encoding="utf-8") as f:
            coefs = json.load(f)
    print(f"  Calibration (VAL-selected): {calib}")

    def _read_csv(*names: str) -> pd.DataFrame:
        for n in names:
            if os.path.exists(n):
                return pd.read_csv(n)
        raise FileNotFoundError(names[0])

    games = _read_csv("todays_games.csv", os.path.join("data", "live", "todays_games.csv"))
    try:
        rosters = _read_csv(
            "todays_rosters.csv", os.path.join("data", "live", "todays_rosters.csv")
        )
    except FileNotFoundError:
        print("  [WARNING] todays_rosters.csv missing.")
        rosters = pd.DataFrame()
    try:
        injuries_raw = _read_csv(
            "injuries.csv", os.path.join("data", "live", "injuries.csv")
        )
    except Exception:
        print("  [WARNING] injuries.csv missing — deficits will be 0.")
        injuries_raw = pd.DataFrame()

    team_ids = (
        pd.concat([games["HOME_TEAM_ID"], games["VISITOR_TEAM_ID"]])
        .astype(int)
        .unique()
        .tolist()
    )
    injuries = injuries_for_tonight(injuries_raw, team_ids, rosters)
    print(
        f"  Injuries tonight-only: {len(injuries)} / {len(injuries_raw)} rows "
        f"({(injuries['Status'].map(lambda s: is_exact_out_status(canonicalize_injury_status(s))).sum()) if not injuries.empty else 0} Out)"
    )

    schedule_ctx = load_schedule_ctx()
    conn = sqlite3.connect(DB_NAME)
    season_id = resolve_season_id(conn)
    print(f"  Using season_id={season_id} for player priors; miss-track={LIVE_SEASON_ID}")

    preds = []
    all_unmatched: list[str] = []
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
        unmatched_h: list[str] = []
        unmatched_a: list[str] = []
        hm = team_absence_deficit_for_serve(
            conn,
            season_id,
            home_id,
            home_abbr,
            rosters,
            injuries,
            10.0,
            value_key,
            unmatched_out=unmatched_h,
        )
        am = team_absence_deficit_for_serve(
            conn,
            season_id,
            away_id,
            away_abbr,
            rosters,
            injuries,
            10.0,
            value_key,
            unmatched_out=unmatched_a,
        )
        # Only log unmatched when team-scoped injuries exist (TeamID present)
        if "TeamID" in injuries.columns:
            for name in unmatched_h:
                print(f"  [unmatched Out] {home_abbr}: {name}")
                all_unmatched.append(f"{home_abbr}:{name}")
            for name in unmatched_a:
                print(f"  [unmatched Out] {away_abbr}: {name}")
                all_unmatched.append(f"{away_abbr}:{name}")

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
        # Pipeline may be raw logistic or CalibratedClassifierCV (Platt/isotonic)
        home_prob = float(model.predict_proba(X)[0, 1])
        away_prob = 1.0 - home_prob
        winner = home_name if home_prob >= away_prob else away_name
        conf = max(home_prob, away_prob)
        reason = biggest_feature_reason(row, coefs, home_name, away_name)

        print(
            f"  {away_name} @ {home_name}: home {home_prob:.1%} -> {winner} "
            f"(def {hm:.1f}/{am:.1f}; rest {hs['rest_days']}/{as_['rest_days']})"
        )
        preds.append(
            {
                "GAME_ID": g["GAME_ID"],
                "Home_Team": home_name,
                "Away_Team": away_name,
                "Home_Win_Prob": home_prob,
                "Away_Win_Prob": away_prob,
                "Predicted_Winner": winner,
                "Confidence": conf,
                "Win_Probability": conf,
                "Model": f"logistic_matchup_{calib}",
                "Home_Player_Deficit": hm,
                "Away_Player_Deficit": am,
                "Reason": reason,
                "Home_Rest": hs["rest_days"],
                "Away_Rest": as_["rest_days"],
                "Home_B2B": hs["is_back_to_back"],
                "Away_B2B": as_["is_back_to_back"],
                "Home_G7": hs["games_last_7"],
                "Away_G7": as_["games_last_7"],
            }
        )

    conn.close()
    if all_unmatched:
        print(f"  Unmatched Out names ({len(all_unmatched)}): {all_unmatched[:20]}")
    if not preds:
        print("[FAIL] No predictions generated.")
        return

    out = pd.DataFrame(preds)
    out.to_csv("final_predictions.csv", index=False)
    live_dir = os.path.join("data", "live")
    os.makedirs(live_dir, exist_ok=True)
    out.to_csv(os.path.join(live_dir, "final_predictions.csv"), index=False)
    for src in (
        "todays_games.csv",
        "todays_rosters.csv",
        "injuries.csv",
        "schedule_context.csv",
    ):
        if os.path.exists(src):
            pd.read_csv(src).to_csv(os.path.join(live_dir, src), index=False)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    stamp = datetime.date.today().isoformat()
    out.to_csv(os.path.join(HISTORY_DIR, f"preds_{stamp}.csv"), index=False)
    print(f"\n[SUCCESS] Wrote final_predictions.csv + data/live/ ({len(out)} games)")


if __name__ == "__main__":
    main()
