"""
Live slate extractors behind one interface.

Providers:
  - NbaApiSource  — stats.nba.com (works from home IPs; blocked on GHA)
  - EspnSource    — site.api.espn.com (works from cloud)

Both return the same DataFrames so fetch_live / Actions can swap sources.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from datetime import datetime, timedelta
from io import StringIO
from typing import Protocol
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from nba_ml.absence import canonicalize_injury_status
from nba_ml.features import LIVE_SEASON, REST_DAYS_CAP
from nba_ml.team_ids import espn_team_id_to_nba

LIVE_DIR = os.path.join("data", "live")
SNAPSHOT_DIR = os.path.join("data", "injury_snapshots")
DB_NAME = "nba_stats.db"
NBA_TZ = ZoneInfo("America/New_York")

GAMES_COLS = [
    "GAME_DATE_EST",
    "GAME_ID",
    "HOME_TEAM_ID",
    "VISITOR_TEAM_ID",
    "GAME_STATUS_TEXT",
]
ROSTER_COLS = ["TeamID", "PLAYER_ID", "PLAYER", "POSITION"]
INJURY_COLS = ["Player", "Injury", "Status"]
CONTEXT_COLS = ["TEAM_ID", "rest_days", "is_back_to_back", "games_last_7"]

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}

ESPN_HEADERS = {
    "User-Agent": BROWSER_HEADERS["User-Agent"],
    "Accept": "application/json",
}


class LiveSource(Protocol):
    name: str

    def fetch_games(self, game_date: str | None = None) -> pd.DataFrame: ...

    def fetch_injuries(self) -> pd.DataFrame: ...

    def fetch_rosters(self, team_ids: list[int], season: str = "2026-27") -> pd.DataFrame: ...

    def refresh_player_stats(self, season: str = "2026-27", season_id: str = "22026") -> bool: ...


def nba_today_str() -> str:
    return datetime.now(NBA_TZ).strftime("%Y-%m-%d")


def _empty_games() -> pd.DataFrame:
    return pd.DataFrame(columns=GAMES_COLS)


def _empty_rosters() -> pd.DataFrame:
    return pd.DataFrame(columns=ROSTER_COLS)


def _empty_injuries() -> pd.DataFrame:
    return pd.DataFrame(columns=INJURY_COLS)


# ---------------------------------------------------------------------------
# NBA stats.nba.com
# ---------------------------------------------------------------------------


class NbaApiSource:
    name = "nba_api"

    def fetch_games(self, game_date: str | None = None) -> pd.DataFrame:
        from nba_api.stats.endpoints import scoreboardv2

        day = game_date or nba_today_str()
        board = scoreboardv2.ScoreboardV2(game_date=day, timeout=60)
        games_df = board.game_header.get_data_frame()
        if games_df.empty:
            return _empty_games()
        summary = games_df[GAMES_COLS].drop_duplicates(subset=["GAME_ID"])
        return summary.reset_index(drop=True)

    def fetch_injuries(self) -> pd.DataFrame:
        """CBS HTML table (nba_api has no public injury endpoint)."""
        url = "https://www.cbssports.com/nba/injuries/"
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=60)
        resp.raise_for_status()
        dfs = pd.read_html(StringIO(resp.text))
        if not dfs:
            return _empty_injuries()
        all_injuries = pd.concat(dfs, ignore_index=True)
        if len(all_injuries.columns) >= 5:
            all_injuries.columns = ["Player", "Pos", "Date", "Injury", "Status"]
        all_injuries["Player"] = all_injuries["Player"].apply(_clean_cbs_player_name)
        if "Status" in all_injuries.columns:
            all_injuries["Status"] = all_injuries["Status"].apply(canonicalize_injury_status)
        out = all_injuries.reindex(columns=["Player", "Injury", "Status", "Pos", "Date"])
        return out

    def fetch_rosters(self, team_ids: list[int], season: str = "2026-27") -> pd.DataFrame:
        from nba_api.stats.endpoints import commonteamroster

        frames = []
        for team_id in team_ids:
            try:
                endpoint = commonteamroster.CommonTeamRoster(
                    team_id=int(team_id), season=season, timeout=60
                )
                frames_list = endpoint.get_data_frames()
                if frames_list:
                    roster = frames_list[0].copy()
                    roster["TeamID"] = int(team_id)
                    frames.append(roster)
            except Exception as exc:
                print(f"  [nba_api] roster fail {team_id}: {exc}")
            time.sleep(0.6)
        if not frames:
            return _empty_rosters()
        df = pd.concat(frames, ignore_index=True)
        rename = {}
        if "PLAYER" not in df.columns and "PLAYER_NAME" in df.columns:
            rename["PLAYER_NAME"] = "PLAYER"
        df = df.rename(columns=rename)
        cols = [c for c in ROSTER_COLS if c in df.columns]
        return df[cols].copy() if cols else _empty_rosters()

    def refresh_player_stats(self, season: str = "2026-27", season_id: str = "22026") -> bool:
        from nba_api.stats.endpoints import leaguegamelog

        if not os.path.exists(DB_NAME):
            print(f"  [nba_api] missing {DB_NAME}; skip stats refresh")
            return False
        try:
            logs = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                player_or_team_abbreviation="P",
                timeout=60,
            )
            df = logs.get_data_frames()[0]
            if df.empty:
                print("  [nba_api] empty LeagueGameLog")
                return False
            col_map = {
                "PLAYER_ID": "Player_ID",
                "GAME_ID": "Game_ID",
                "TEAM_ABBREVIATION": "TEAM_ABBR",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            df = _compute_advanced_metrics(df)
            conn = sqlite3.connect(DB_NAME)
            conn.execute(f"DELETE FROM player_logs WHERE SEASON_ID = '{season_id}'")
            conn.commit()
            df.to_sql("player_logs", conn, if_exists="append", index=False)
            conn.close()
            print(f"  [nba_api] refreshed {len(df)} player-log rows for {season}")
            return True
        except Exception as exc:
            print(f"  [nba_api] stats refresh failed: {exc}")
            return False


# ---------------------------------------------------------------------------
# ESPN (cloud-friendly)
# ---------------------------------------------------------------------------


class EspnSource:
    name = "espn"
    SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

    def fetch_games(self, game_date: str | None = None) -> pd.DataFrame:
        day = game_date or nba_today_str()
        resp = requests.get(
            self.SCOREBOARD,
            params={"dates": day.replace("-", "")},
            headers=ESPN_HEADERS,
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
        rows = []
        for event in payload.get("events", []):
            comps = event.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            home_id = away_id = None
            for c in comp.get("competitors", []):
                nba_id = espn_team_id_to_nba(c.get("team", {}).get("id"))
                if c.get("homeAway") == "home":
                    home_id = nba_id
                elif c.get("homeAway") == "away":
                    away_id = nba_id
            if home_id is None or away_id is None:
                continue
            status = (
                (comp.get("status") or {}).get("type", {}).get("shortDetail")
                or (comp.get("status") or {}).get("type", {}).get("description")
                or event.get("status", {}).get("type", {}).get("description")
                or ""
            )
            # Keep ESPN event id; dashboard normalizes GAME_ID strings.
            rows.append(
                {
                    "GAME_DATE_EST": f"{day}T00:00:00",
                    "GAME_ID": str(event.get("id")),
                    "HOME_TEAM_ID": int(home_id),
                    "VISITOR_TEAM_ID": int(away_id),
                    "GAME_STATUS_TEXT": status,
                }
            )
        if not rows:
            return _empty_games()
        return pd.DataFrame(rows)[GAMES_COLS].drop_duplicates(subset=["GAME_ID"])

    def fetch_injuries(self) -> pd.DataFrame:
        resp = requests.get(self.INJURIES, headers=ESPN_HEADERS, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        rows = []
        for team_block in payload.get("injuries", []):
            for inj in team_block.get("injuries", []):
                athlete = inj.get("athlete") or {}
                player = (
                    athlete.get("displayName")
                    or athlete.get("fullName")
                    or " ".join(
                        filter(
                            None,
                            [athlete.get("firstName"), athlete.get("lastName")],
                        )
                    )
                ).strip()
                if not player:
                    continue
                injury_type = (inj.get("type") or {}).get("description") or ""
                detail = inj.get("shortComment") or inj.get("longComment") or injury_type
                status = canonicalize_injury_status(inj.get("status") or injury_type)
                rows.append(
                    {
                        "Player": player,
                        "Injury": detail or injury_type or "Undisclosed",
                        "Status": status,
                    }
                )
        if not rows:
            return _empty_injuries()
        return pd.DataFrame(rows)

    def fetch_rosters(self, team_ids: list[int], season: str = "2026-27") -> pd.DataFrame:
        """ESPN athlete IDs ≠ NBA PLAYER_ID — build rosters from local DB."""
        return rosters_from_db(team_ids)

    def refresh_player_stats(self, season: str = "2026-27", season_id: str = "22026") -> bool:
        print("  [espn] player box-score refresh not implemented (use nba_api locally)")
        return False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def get_source(name: str) -> LiveSource:
    key = (name or "auto").lower().strip()
    if key in ("nba", "nba_api", "stats"):
        return NbaApiSource()
    if key == "espn":
        return EspnSource()
    raise ValueError(f"Unknown live source: {name}")


def fetch_with_fallback(
    preferred: str = "auto",
    game_date: str | None = None,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Return (source_name, games, injuries).

    auto: try nba_api first, fall back to ESPN on failure/empty.
    """
    order: list[LiveSource]
    if preferred == "auto":
        order = [NbaApiSource(), EspnSource()]
    elif preferred == "espn":
        order = [EspnSource()]
    else:
        order = [get_source(preferred), EspnSource()]

    last_err: Exception | None = None
    for src in order:
        try:
            print(f"  Trying source={src.name} for games/injuries...")
            games = src.fetch_games(game_date=game_date)
            injuries = src.fetch_injuries()
            if games is not None and not games.empty:
                print(f"  [OK] {src.name}: {len(games)} games, {len(injuries)} injuries")
                return src.name, games, injuries
            print(f"  [{src.name}] no games for date; trying next source")
        except Exception as exc:
            last_err = exc
            print(f"  [{src.name}] failed: {exc}")
    if last_err:
        raise RuntimeError(f"All live sources failed: {last_err}") from last_err
    return "none", _empty_games(), _empty_injuries()


def rosters_from_db(team_ids: list[int], db_path: str = DB_NAME) -> pd.DataFrame:
    if not team_ids or not os.path.exists(db_path):
        return _empty_rosters()
    conn = sqlite3.connect(db_path)
    seasons = pd.read_sql(
        "SELECT SEASON_ID, MAX(GAME_DATE) mx FROM player_logs GROUP BY SEASON_ID ORDER BY mx DESC",
        conn,
    )
    if seasons.empty:
        conn.close()
        return _empty_rosters()
    season_id = str(seasons.iloc[0]["SEASON_ID"])
    if LIVE_SEASON in set(seasons["SEASON_ID"].astype(str)):
        season_id = LIVE_SEASON
    placeholders = ",".join("?" for _ in team_ids)
    q = f"""
        SELECT DISTINCT TEAM_ID as TeamID, Player_ID as PLAYER_ID,
               PLAYER_NAME as PLAYER, '' as POSITION
        FROM player_logs
        WHERE SEASON_ID = ? AND TEAM_ID IN ({placeholders})
    """
    df = pd.read_sql(q, conn, params=[season_id, *[int(t) for t in team_ids]])
    conn.close()
    if df.empty:
        return _empty_rosters()
    return df[ROSTER_COLS]


def compute_schedule_context(
    games: pd.DataFrame,
    rosters: pd.DataFrame | None = None,
    db_path: str = DB_NAME,
    game_date: str | None = None,
) -> pd.DataFrame:
    day = game_date or nba_today_str()
    today_dt = pd.to_datetime(day)
    if games.empty:
        return pd.DataFrame(columns=CONTEXT_COLS)
    team_ids = (
        pd.concat([games["HOME_TEAM_ID"], games["VISITOR_TEAM_ID"]]).astype(int).unique()
    )
    if not os.path.exists(db_path):
        return pd.DataFrame(
            [
                {
                    "TEAM_ID": int(tid),
                    "rest_days": 3,
                    "is_back_to_back": 0,
                    "games_last_7": 0,
                }
                for tid in team_ids
            ]
        )

    conn = sqlite3.connect(db_path)
    logs = pd.read_sql("SELECT GAME_DATE, TEAM_ID, Player_ID FROM player_logs", conn)
    conn.close()
    if logs.empty:
        return pd.DataFrame(
            [
                {
                    "TEAM_ID": int(tid),
                    "rest_days": 3,
                    "is_back_to_back": 0,
                    "games_last_7": 0,
                }
                for tid in team_ids
            ]
        )

    logs["GAME_DATE_PARSED"] = pd.to_datetime(logs["GAME_DATE"], format="mixed")
    rows = []
    for tid in team_ids:
        team_dates = (
            logs.loc[logs["TEAM_ID"] == int(tid), "GAME_DATE_PARSED"]
            .drop_duplicates()
            .sort_values(ascending=False)
        )
        if len(team_dates) == 0 and rosters is not None and not rosters.empty:
            pids = rosters.loc[rosters["TeamID"] == int(tid), "PLAYER_ID"].unique()
            team_dates = (
                logs.loc[logs["Player_ID"].isin(pids), "GAME_DATE_PARSED"]
                .drop_duplicates()
                .sort_values(ascending=False)
            )
        if len(team_dates) > 0:
            rest_days = int(min((today_dt - team_dates.iloc[0]).days, REST_DAYS_CAP))
            is_b2b = 1 if rest_days == 1 else 0
            seven_ago = today_dt - timedelta(days=7)
            games_last_7 = int((team_dates >= seven_ago).sum())
        else:
            rest_days, is_b2b, games_last_7 = 3, 0, 0
        rows.append(
            {
                "TEAM_ID": int(tid),
                "rest_days": rest_days,
                "is_back_to_back": is_b2b,
                "games_last_7": games_last_7,
            }
        )
    return pd.DataFrame(rows)[CONTEXT_COLS]


def write_live_bundle(
    games: pd.DataFrame,
    injuries: pd.DataFrame,
    rosters: pd.DataFrame,
    context: pd.DataFrame,
    live_dir: str = LIVE_DIR,
    also_root: bool = True,
) -> None:
    os.makedirs(live_dir, exist_ok=True)
    pairs = {
        "todays_games.csv": games,
        "injuries.csv": injuries,
        "todays_rosters.csv": rosters,
        "schedule_context.csv": context,
    }
    for name, df in pairs.items():
        path = os.path.join(live_dir, name)
        df.to_csv(path, index=False)
        if also_root:
            df.to_csv(name, index=False)
        print(f"  Wrote {path} ({len(df)} rows)")


def write_injury_snapshot(
    injuries: pd.DataFrame,
    day: str | None = None,
    snapshot_dir: str = SNAPSHOT_DIR,
) -> str | None:
    if injuries is None or injuries.empty:
        return None
    os.makedirs(snapshot_dir, exist_ok=True)
    stamp = day or nba_today_str()
    path = os.path.join(snapshot_dir, f"{stamp}.csv")
    # Append-only: do not overwrite an existing dated snapshot.
    if os.path.exists(path):
        print(f"  Snapshot exists, leaving: {path}")
        return path
    injuries.to_csv(path, index=False)
    print(f"  Wrote injury snapshot {path} ({len(injuries)} rows)")
    return path


def _compute_advanced_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EFG_PCT"] = np.where(
        df["FGA"] > 0, (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"], 0.0
    )
    tsa = 2 * (df["FGA"] + 0.44 * df["FTA"])
    df["TS_PCT"] = np.where(tsa > 0, df["PTS"] / tsa, 0.0)
    tov_denom = df["FGA"] + 0.44 * df["FTA"] + df["TOV"]
    df["TOV_PCT"] = np.where(tov_denom > 0, df["TOV"] / tov_denom, 0.0)
    return df


def _clean_cbs_player_name(raw_name: str) -> str:
    raw_name = str(raw_name).strip()
    suffix_pattern = r"(?:Jr\.|Sr\.|III|II|IV)(?=[A-Z])"
    match = re.search(suffix_pattern, raw_name)
    if match:
        return raw_name[match.end() :].strip()
    standard_pattern = (
        r"(?<=[a-z])(?!(?:Van|De|Mc|Mac|La|Le|Di|St)[A-Z])(?=[A-Z])"
    )
    match = re.search(standard_pattern, raw_name)
    if match:
        return raw_name[match.end() :].strip()
    return raw_name


__all__ = [
    "LiveSource",
    "NbaApiSource",
    "EspnSource",
    "fetch_with_fallback",
    "get_source",
    "rosters_from_db",
    "compute_schedule_context",
    "write_live_bundle",
    "write_injury_snapshot",
    "nba_today_str",
]
