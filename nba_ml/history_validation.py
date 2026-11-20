"""Grade history/preds_*.csv against final scores (DB and/or ESPN)."""

from __future__ import annotations

import glob
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from sklearn.metrics import log_loss

from nba_ml.live_sources import ESPN_HEADERS, EspnSource
from nba_ml.team_ids import espn_team_id_to_nba, team_name

HISTORY_DIR = "history"


def _ll(y, p) -> float:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return float(log_loss(y, p, labels=[0, 1]))


def list_pred_dates(history_dir: str = HISTORY_DIR) -> list[str]:
    dates = []
    for path in glob.glob(os.path.join(history_dir, "preds_*.csv")):
        base = os.path.basename(path)
        stamp = base[len("preds_") : -len(".csv")]
        try:
            datetime.strptime(stamp, "%Y-%m-%d")
        except ValueError:
            continue
        # Skip legacy player-level files
        try:
            df = pd.read_csv(path, nrows=1)
        except Exception:
            continue
        if "Predicted_Winner" not in df.columns or "Home_Win_Prob" not in df.columns:
            continue
        dates.append(stamp)
    return sorted(dates)


def load_preds(date_str: str, history_dir: str = HISTORY_DIR) -> pd.DataFrame:
    path = os.path.join(history_dir, f"preds_{date_str}.csv")
    return pd.read_csv(path)


def _norm_name(s: str) -> str:
    return str(s).strip().lower()


def results_from_db(date_str: str, db_path: str = "nba_stats.db") -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    actuals = pd.read_sql(
        "SELECT DISTINCT Game_ID, WL, MATCHUP, GAME_DATE, TEAM_ABBR FROM player_logs "
        "WHERE GAME_DATE LIKE ?",
        conn,
        params=(f"%{date_str}%",),
    )
    conn.close()
    if actuals.empty:
        return pd.DataFrame()
    homes = actuals[actuals["MATCHUP"].astype(str).str.contains("vs.", na=False)]
    rows = []
    for _, r in homes.iterrows():
        home_won = str(r["WL"]).upper() == "W"
        rows.append(
            {
                "GAME_DATE": date_str,
                "home_abbr": r["TEAM_ABBR"],
                "home_win": int(home_won),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["home_abbr"])


def results_from_espn(date_str: str) -> pd.DataFrame:
    stamp = date_str.replace("-", "")
    try:
        resp = requests.get(
            EspnSource.SCOREBOARD,
            params={"dates": stamp},
            headers=ESPN_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return pd.DataFrame()
    rows = []
    for event in payload.get("events", []):
        comps = event.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        status = (comp.get("status") or {}).get("type") or {}
        if not status.get("completed") and status.get("name") != "STATUS_FINAL":
            continue
        home = away = None
        home_score = away_score = None
        for c in comp.get("competitors", []):
            nba_id = espn_team_id_to_nba(c.get("team", {}).get("id"))
            if nba_id is None:
                continue
            entry = {
                "team_id": nba_id,
                "name": team_name(nba_id),
                "score": float(c.get("score") or 0),
                "winner": bool(c.get("winner")),
            }
            if c.get("homeAway") == "home":
                home = entry
                home_score = entry["score"]
            else:
                away = entry
                away_score = entry["score"]
        if not home or not away:
            continue
        home_win = int(home["winner"] or (home_score is not None and home_score > away_score))
        rows.append(
            {
                "GAME_DATE": date_str,
                "Home_Team": home["name"],
                "Away_Team": away["name"],
                "home_win": home_win,
                "home_score": home_score,
                "away_score": away_score,
            }
        )
    return pd.DataFrame(rows)


def grade_preds(preds: pd.DataFrame, date_str: str, db_path: str = "nba_stats.db") -> pd.DataFrame:
    espn = results_from_espn(date_str)
    graded = []
    for _, pr in preds.iterrows():
        home = pr["Home_Team"]
        away = pr["Away_Team"]
        actual = None
        if not espn.empty:
            hit = espn[
                (espn["Home_Team"].map(_norm_name) == _norm_name(home))
                & (espn["Away_Team"].map(_norm_name) == _norm_name(away))
            ]
            if not hit.empty:
                actual = int(hit.iloc[0]["home_win"])
        if actual is None:
            # DB fallback via abbr is weaker; skip if ESPN miss
            continue
        p_home = float(pr["Home_Win_Prob"])
        pred_home = p_home >= 0.5
        graded.append(
            {
                "GAME_DATE": date_str,
                "Matchup": f"{away} @ {home}",
                "Home_Team": home,
                "Away_Team": away,
                "Home_Win_Prob": p_home,
                "Predicted_Winner": pr["Predicted_Winner"],
                "home_win": actual,
                "Actual_Winner": home if actual else away,
                "Hit": int(pred_home == bool(actual)),
                "Elo_Home_Prior": np.nan,
            }
        )
    return pd.DataFrame(graded)


def rolling_metrics(graded: pd.DataFrame) -> dict:
    if graded is None or graded.empty:
        return {}
    y = graded["home_win"].astype(int).values
    p = graded["Home_Win_Prob"].astype(float).clip(1e-6, 1 - 1e-6).values
    elo_p = graded["Elo_Home_Prior"].fillna(0.5).astype(float).clip(1e-6, 1 - 1e-6).values
    return {
        "n": int(len(graded)),
        "accuracy": float((graded["Hit"] == 1).mean()),
        "log_loss": float(_ll(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "home_always_acc": float((y == 1).mean()),
        "home_always_ll": float(_ll(y, np.full(len(y), 0.55))),
        "elo_acc": (
            float(((elo_p >= 0.5) == (y == 1)).mean())
            if graded["Elo_Home_Prior"].notna().any()
            else None
        ),
        "elo_ll": (
            float(_ll(y, elo_p)) if graded["Elo_Home_Prior"].notna().any() else None
        ),
    }


def summarize_season(dates: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    dates = dates or list_pred_dates()
    frames = []
    for d in dates:
        try:
            preds = load_preds(d)
        except Exception:
            continue
        g = grade_preds(preds, d)
        if not g.empty:
            frames.append(g)
    if not frames:
        return pd.DataFrame(), {}
    all_g = pd.concat(frames, ignore_index=True)
    y = all_g["home_win"].astype(int).values
    p = all_g["Home_Win_Prob"].astype(float).clip(1e-6, 1 - 1e-6).values
    metrics = {
        "n": int(len(all_g)),
        "accuracy": float(all_g["Hit"].mean()),
        "log_loss": float(_ll(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "home_always_acc": float(y.mean()),
        "home_always_ll": float(_ll(y, np.full(len(y), 0.55))),
    }
    return all_g, metrics


def fetch_espn_standings() -> pd.DataFrame:
    url = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
    try:
        resp = requests.get(url, headers=ESPN_HEADERS, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return pd.DataFrame()
    rows = []
    for conf in payload.get("children", []):
        conf_name = conf.get("abbreviation") or conf.get("name") or ""
        standings = (conf.get("standings") or {}).get("entries") or []
        for entry in standings:
            team = entry.get("team") or {}
            espn_id = team.get("id")
            nba_id = espn_team_id_to_nba(espn_id)
            stats = {s.get("name"): s.get("value") for s in entry.get("stats", [])}
            rows.append(
                {
                    "Team": team.get("displayName")
                    or (team_name(nba_id) if nba_id else team.get("shortDisplayName")),
                    "Abbr": team.get("abbreviation"),
                    "Conference": conf_name,
                    "W": int(float(stats.get("wins") or 0)),
                    "L": int(float(stats.get("losses") or 0)),
                    "Win%": float(stats.get("winPercent") or stats.get("winpercent") or 0),
                    "GB": stats.get("gamesBehind") or stats.get("gamesbehind"),
                    "TEAM_ID": nba_id,
                }
            )
    return pd.DataFrame(rows)
