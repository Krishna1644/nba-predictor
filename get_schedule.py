"""Thin wrapper — prefer `python scripts/fetch_live.py`."""

from __future__ import annotations

import pandas as pd

from nba_ml.live_sources import (
    NbaApiSource,
    compute_schedule_context,
    fetch_with_fallback,
    nba_today_str,
    rosters_from_db,
    write_injury_snapshot,
    write_live_bundle,
)


def main() -> int:
    day = nba_today_str()
    print(f"\nCHECKING SCHEDULE FOR: {day}")
    src, games, injuries = fetch_with_fallback(preferred="auto", game_date=day)
    if games.empty:
        print(f"\nNO GAMES FOUND for {day}.")
        write_injury_snapshot(injuries, day=day)
        return 1
    print(f"\nFound {len(games)} unique games via {src}.")
    print(games[["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]].head(10))
    team_ids = (
        pd.concat([games["HOME_TEAM_ID"], games["VISITOR_TEAM_ID"]])
        .astype(int)
        .unique()
        .tolist()
    )
    try:
        rosters = NbaApiSource().fetch_rosters(team_ids)
    except Exception:
        rosters = rosters_from_db(team_ids)
    if rosters.empty:
        rosters = rosters_from_db(team_ids)
    context = compute_schedule_context(games, rosters, game_date=day)
    write_live_bundle(games, injuries, rosters, context)
    write_injury_snapshot(injuries, day=day)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
