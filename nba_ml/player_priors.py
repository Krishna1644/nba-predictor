"""
Pre-game player priors and missing-list builders.

Deficit math lives in `nba_ml.features.compute_team_absence_deficit` only.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from nba_ml.absence import (
    canonicalize_injury_status,
    is_exact_out_status,
    normalize_name,
    season_id_to_prev,
)
from nba_ml.features import (
    ROTATION_MIN_THRESHOLD,
    blend_player_prior,
    compute_team_absence_deficit,
    player_impact,
    player_impact_eff,
)

# Keys stored as expanding sums / season-end averages
STAT_KEYS = (
    "pts",
    "fgm",
    "fga",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "stl",
    "ast",
    "blk",
    "pf",
    "tov",
    "reb",
    "minutes",
)

# Box-score columns on player log rows (observe_game)
LOG_COLS = [
    "Player_ID",
    "MIN_NUMERIC",
    "PTS",
    "FGM",
    "FGA",
    "FTM",
    "FTA",
    "OREB",
    "DREB",
    "STL",
    "AST",
    "BLK",
    "PF",
    "TOV",
    "REB",
]


def _season_end_avgs(logs: pd.DataFrame) -> dict:
    g = logs.groupby(["SEASON_ID", "Player_ID"], as_index=False).agg(
        pts=("PTS", "mean"),
        fgm=("FGM", "mean"),
        fga=("FGA", "mean"),
        ftm=("FTM", "mean"),
        fta=("FTA", "mean"),
        oreb=("OREB", "mean"),
        dreb=("DREB", "mean"),
        stl=("STL", "mean"),
        ast=("AST", "mean"),
        blk=("BLK", "mean"),
        pf=("PF", "mean"),
        tov=("TOV", "mean"),
        reb=("REB", "mean"),
        minutes=("MIN_NUMERIC", "mean"),
    )
    return {
        (str(r.SEASON_ID), int(r.Player_ID)): {k: float(getattr(r, k)) for k in STAT_KEYS}
        for r in g.itertuples(index=False)
    }


def _impacts_from_avgs(avgs: dict) -> tuple[float, float]:
    counting = player_impact(
        avgs["pts"], avgs["reb"], avgs["ast"], avgs["stl"], avgs["blk"], avgs["tov"]
    )
    eff = player_impact_eff(
        avgs["pts"],
        avgs["fgm"],
        avgs["fga"],
        avgs["ftm"],
        avgs["fta"],
        avgs["oreb"],
        avgs["dreb"],
        avgs["stl"],
        avgs["ast"],
        avgs["blk"],
        avgs["pf"],
        avgs["tov"],
    )
    return float(counting), float(eff)


def _empty_prior(consecutive_misses: int) -> dict:
    base = {k: 0.0 for k in STAT_KEYS}
    base.update(
        {
            "impact": 0.0,
            "impact_eff": 0.0,
            "gp": 0,
            "consecutive_misses": int(consecutive_misses),
        }
    )
    return base


def _prior_dict(st: dict | None, prev: dict | None, consecutive_misses: int) -> dict:
    if st is None or st["n"] <= 0:
        if prev is None:
            return _empty_prior(consecutive_misses)
        counting, eff = _impacts_from_avgs(prev)
        return {
            **{k: float(prev[k]) for k in STAT_KEYS},
            "impact": counting,
            "impact_eff": eff,
            "gp": 0,
            "consecutive_misses": int(consecutive_misses),
        }

    n = st["n"]
    curr = {k: st[k] / n for k in STAT_KEYS}
    blended = {
        k: blend_player_prior(n, curr[k], prev[k] if prev else None) for k in STAT_KEYS
    }
    counting, eff = _impacts_from_avgs(blended)
    return {
        **blended,
        "impact": counting,
        "impact_eff": eff,
        "gp": n,
        "consecutive_misses": int(consecutive_misses),
    }


class TeamAvailabilityTracker:
    def __init__(self, season_end: dict):
        self.season_end = season_end
        self.curr_state = {}
        self.consec = defaultdict(int)
        self.members = defaultdict(set)

    def prior_lookup(self, season: str, team: str) -> dict:
        prev_season = season_id_to_prev(season)
        out = {}
        for pid in self.members[(season, team)]:
            prev = self.season_end.get((prev_season, pid))
            out[pid] = _prior_dict(
                self.curr_state.get((season, pid)),
                prev,
                self.consec[(season, team, pid)],
            )
        return out

    def training_missing_ids(self, season: str, team: str, played: set[int]) -> list[int]:
        lookup = self.prior_lookup(season, team)
        return [
            pid
            for pid, prior in lookup.items()
            if pid not in played and prior["minutes"] >= ROTATION_MIN_THRESHOLD
        ]

    def observe_game(
        self,
        season: str,
        team: str,
        played_rows: pd.DataFrame,
        played_ids: set[int],
    ):
        for pid in list(self.members[(season, team)]):
            if pid not in played_ids:
                self.consec[(season, team, pid)] += 1

        if played_rows is None or played_rows.empty:
            return

        for pref in played_rows.itertuples(index=False):
            pid = int(pref.Player_ID)
            self.members[(season, team)].add(pid)
            self.consec[(season, team, pid)] = 0
            st_key = (season, pid)
            st = self.curr_state.get(st_key)
            if st is None:
                st = {"n": 0, **{k: 0.0 for k in STAT_KEYS}}
            st["n"] += 1
            st["pts"] += float(pref.PTS)
            st["fgm"] += float(pref.FGM)
            st["fga"] += float(pref.FGA)
            st["ftm"] += float(pref.FTM)
            st["fta"] += float(pref.FTA)
            st["oreb"] += float(pref.OREB)
            st["dreb"] += float(pref.DREB)
            st["stl"] += float(pref.STL)
            st["ast"] += float(pref.AST)
            st["blk"] += float(pref.BLK)
            st["pf"] += float(pref.PF)
            st["tov"] += float(pref.TOV)
            st["reb"] += float(pref.REB)
            st["minutes"] += float(pref.MIN_NUMERIC)
            self.curr_state[st_key] = st


def compute_team_game_player_deficits(
    logs: pd.DataFrame, team_games: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (counting_impact, efficiency_impact, minutes) deficits."""
    logs = logs.copy()
    if "MIN_NUMERIC" not in logs.columns:
        logs["MIN_NUMERIC"] = pd.to_numeric(logs["MIN"], errors="coerce").fillna(0.0)
    for col in ("FGM", "FGA", "FTM", "FTA", "OREB", "DREB", "STL", "BLK", "PF", "TOV", "REB"):
        if col not in logs.columns:
            logs[col] = 0.0
        else:
            logs[col] = pd.to_numeric(logs[col], errors="coerce").fillna(0.0)

    season_end = _season_end_avgs(logs)
    tracker = TeamAvailabilityTracker(season_end)

    played_map = {
        key: {int(x) for x in group["Player_ID"].tolist()}
        for key, group in logs.groupby(["Game_ID", "TEAM_ABBR"])
    }
    logs_by_gt = {
        key: group[LOG_COLS]
        for key, group in logs.groupby(["Game_ID", "TEAM_ABBR"])
    }

    ordered = team_games.sort_values(["game_date", "Game_ID", "TEAM_ABBR"])
    impact_vals, eff_vals, minutes_vals = {}, {}, {}

    for row in ordered.itertuples():
        key = (row.Game_ID, row.TEAM_ABBR)
        season, team = str(row.season_id), row.TEAM_ABBR
        played = played_map.get(key, set())
        prior_lookup = tracker.prior_lookup(season, team)
        missing_ids = tracker.training_missing_ids(season, team, played)
        impact_vals[key] = compute_team_absence_deficit(
            missing_ids, prior_lookup, value_key="impact"
        )
        eff_vals[key] = compute_team_absence_deficit(
            missing_ids, prior_lookup, value_key="impact_eff"
        )
        minutes_vals[key] = compute_team_absence_deficit(
            missing_ids, prior_lookup, value_key="minutes"
        )
        tracker.observe_game(season, team, logs_by_gt.get(key), played)

    def align(d):
        return np.array(
            [d.get((r.Game_ID, r.TEAM_ABBR), 0.0) for r in team_games.itertuples()],
            dtype=np.float64,
        )

    return align(impact_vals), align(eff_vals), align(minutes_vals)


def compute_team_game_player_deficit(logs, team_games):
    impact, _, _ = compute_team_game_player_deficits(logs, team_games)
    return impact


def _strip_name_suffix(norm: str) -> str:
    n = norm.strip()
    for suf in (" jr.", " jr", " sr.", " sr", " iii", " ii", " iv"):
        if n.endswith(suf):
            return n[: -len(suf)].strip()
    return n


def match_name_key(name: str) -> str:
    return _strip_name_suffix(normalize_name(name))


def serving_missing_ids_from_injuries(
    injuries: pd.DataFrame,
    roster: pd.DataFrame,
    prior_lookup: dict,
    unmatched_out: list[str] | None = None,
) -> list[int]:
    """
    Map Status==Out injury names onto this team's roster PLAYER_IDs.

    Only roster players count (tonight's team). Optionally append Out names that
    did not match any roster player into `unmatched_out`.
    """
    if injuries is None or injuries.empty or roster is None or roster.empty:
        return []

    name_col = "Player" if "Player" in injuries.columns else injuries.columns[0]
    if "Status" not in injuries.columns:
        return []

    out_names: set[str] = set()
    out_raw_by_key: dict[str, str] = {}
    for n, st in zip(injuries[name_col], injuries["Status"]):
        status = str(st).strip()
        if status != "Out":
            status = canonicalize_injury_status(status)
        if not is_exact_out_status(status):
            continue
        key = match_name_key(str(n))
        if not key:
            continue
        out_names.add(key)
        out_raw_by_key.setdefault(key, str(n))

    pid_col = "PLAYER_ID" if "PLAYER_ID" in roster.columns else "Player_ID"
    pname_col = "PLAYER" if "PLAYER" in roster.columns else "PLAYER_NAME"

    roster_keys = {match_name_key(str(row[pname_col])) for _, row in roster.iterrows()}
    if unmatched_out is not None:
        for key in sorted(out_names - roster_keys):
            unmatched_out.append(out_raw_by_key.get(key, key))

    missing = []
    for _, row in roster.iterrows():
        if match_name_key(str(row[pname_col])) not in out_names:
            continue
        pid = int(row[pid_col])
        prior = prior_lookup.get(pid)
        if not prior or prior.get("minutes", 0.0) < ROTATION_MIN_THRESHOLD:
            continue
        missing.append(pid)
    return missing


def build_prior_lookup_from_db(conn, season_id: str, player_ids: list[int]) -> dict:
    if not player_ids:
        return {}
    prev_season = season_id_to_prev(season_id)
    ids = ",".join(str(int(x)) for x in player_ids)

    curr = pd.read_sql(
        f"""
        SELECT Player_ID, COUNT(*) as n,
               SUM(PTS) as pts, SUM(FGM) as fgm, SUM(FGA) as fga,
               SUM(FTM) as ftm, SUM(FTA) as fta,
               SUM(OREB) as oreb, SUM(DREB) as dreb,
               SUM(STL) as stl, SUM(AST) as ast, SUM(BLK) as blk,
               SUM(PF) as pf, SUM(TOV) as tov, SUM(REB) as reb,
               SUM(CAST(MIN AS FLOAT)) as minutes
        FROM player_logs
        WHERE SEASON_ID = ? AND Player_ID IN ({ids})
        GROUP BY Player_ID
        """,
        conn,
        params=(str(season_id),),
    )
    prev = pd.read_sql(
        f"""
        SELECT Player_ID,
               AVG(PTS) as pts, AVG(FGM) as fgm, AVG(FGA) as fga,
               AVG(FTM) as ftm, AVG(FTA) as fta,
               AVG(OREB) as oreb, AVG(DREB) as dreb,
               AVG(STL) as stl, AVG(AST) as ast, AVG(BLK) as blk,
               AVG(PF) as pf, AVG(TOV) as tov, AVG(REB) as reb,
               AVG(CAST(MIN AS FLOAT)) as minutes
        FROM player_logs
        WHERE SEASON_ID = ? AND Player_ID IN ({ids})
        GROUP BY Player_ID
        """,
        conn,
        params=(prev_season,),
    )
    prev_map = {
        int(r.Player_ID): {k: float(getattr(r, k)) for k in STAT_KEYS}
        for r in prev.itertuples(index=False)
    }
    curr_map = {
        int(r.Player_ID): {
            "n": int(r.n),
            **{k: float(getattr(r, k)) for k in STAT_KEYS},
        }
        for r in curr.itertuples(index=False)
    }
    return {
        int(pid): _prior_dict(curr_map.get(int(pid)), prev_map.get(int(pid)), 0)
        for pid in player_ids
    }


def annotate_consecutive_misses(
    conn,
    season_id: str,
    team_abbr: str,
    prior_lookup: dict,
    *,
    miss_season_id: str | None = None,
) -> dict:
    """
    Count consecutive team games missed in `miss_season_id` (default: season_id).

    When the miss-tracking season has no games yet (e.g. live 22026 empty while
    priors come from 22025), leave consecutive_misses at 0 so long prior-season
    DNPs do not zero out tonight's Out deficits.
    """
    track_season = str(miss_season_id or season_id)
    games = pd.read_sql(
        """
        SELECT DISTINCT Game_ID, GAME_DATE
        FROM player_logs
        WHERE SEASON_ID = ? AND MATCHUP LIKE ?
        ORDER BY GAME_DATE
        """,
        conn,
        params=(track_season, f"{team_abbr} %"),
    )
    out = {pid: dict(p) for pid, p in prior_lookup.items()}
    for pid in out:
        out[pid]["consecutive_misses"] = int(out[pid].get("consecutive_misses", 0) or 0)

    if games.empty:
        for pid in out:
            out[pid]["consecutive_misses"] = 0
        return out

    gids = list(games["Game_ID"].astype(str))
    for pid in list(out.keys()):
        appeared = pd.read_sql(
            """
            SELECT DISTINCT Game_ID
            FROM player_logs
            WHERE SEASON_ID = ? AND Player_ID = ? AND MATCHUP LIKE ?
            """,
            conn,
            params=(track_season, int(pid), f"{team_abbr} %"),
        )
        appeared_ids = set(appeared["Game_ID"].astype(str))
        last_idx = -1
        for i, gid in enumerate(gids):
            if gid in appeared_ids:
                last_idx = i
        # Never appeared this season → treat as 0 for serving (new roster / no logs)
        out[pid]["consecutive_misses"] = (
            0 if last_idx < 0 else int(len(gids) - last_idx - 1)
        )
    return out
