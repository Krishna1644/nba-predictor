"""Thin wrapper — prefer `python scripts/fetch_live.py`."""

from __future__ import annotations

import pandas as pd

from nba_ml.live_sources import NbaApiSource, rosters_from_db


def main() -> int:
    try:
        games = pd.read_csv("todays_games.csv")
    except FileNotFoundError:
        print("Error: 'todays_games.csv' not found. Run scripts/fetch_live.py first.")
        return 1
    team_ids = (
        pd.concat([games["HOME_TEAM_ID"], games["VISITOR_TEAM_ID"]])
        .astype(int)
        .unique()
        .tolist()
    )
    print(f"Found {len(team_ids)} unique teams playing today.")
    try:
        roster = NbaApiSource().fetch_rosters(team_ids)
    except Exception as exc:
        print(f"nba_api roster failed ({exc}); using DB fallback")
        roster = rosters_from_db(team_ids)
    if roster.empty:
        roster = rosters_from_db(team_ids)
    if roster.empty:
        print("No rosters found.")
        return 1
    roster.to_csv("todays_rosters.csv", index=False)
    print(f"\nSuccess! Saved {len(roster)} players to 'todays_rosters.csv'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
