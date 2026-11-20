"""
Extract today's live slate into data/live/.

Sources (same interface in nba_ml.live_sources):
  --source auto   try nba_api, fall back to ESPN (default; for home machines)
  --source espn   ESPN only (for GitHub Actions / cloud IPs)
  --source nba_api  stats.nba.com only (no fallback)

Also writes append-only injury snapshots under data/injury_snapshots/YYYY-MM-DD.csv.

Usage:
  python scripts/fetch_live.py
  python scripts/fetch_live.py --source espn
"""

from __future__ import annotations

import argparse
import os
import sys

# repo root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nba_ml.live_sources import (
    EspnSource,
    NbaApiSource,
    compute_schedule_context,
    fetch_with_fallback,
    get_source,
    nba_today_str,
    rosters_from_db,
    write_injury_snapshot,
    write_live_bundle,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch live NBA slate into data/live/")
    ap.add_argument(
        "--source",
        default="auto",
        choices=["auto", "espn", "nba_api"],
        help="Data provider (auto = nba_api then ESPN)",
    )
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: America/New_York today)")
    ap.add_argument(
        "--skip-stats",
        action="store_true",
        help="Do not refresh current-season player logs via nba_api",
    )
    ap.add_argument(
        "--season",
        default="2026-27",
        help="Season string for nba_api roster/stats calls",
    )
    ap.add_argument("--season-id", default="22026", help="SEASON_ID for DB refresh")
    args = ap.parse_args()

    day = args.date or nba_today_str()
    print(f"=== fetch_live  date={day}  source={args.source} ===")

    src_name, games, injuries = fetch_with_fallback(
        preferred=args.source, game_date=day
    )
    if games.empty:
        print("[FAIL] No games found.")
        write_injury_snapshot(injuries, day=day)
        return 1

    team_ids = (
        pd_concat_team_ids(games)
    )
    print(f"  Teams tonight: {len(team_ids)}")

    # Rosters: prefer nba_api when that source won (or was requested);
    # otherwise / on failure, map from local DB (ESPN-safe).
    rosters = None
    if src_name == "nba_api" or args.source == "nba_api":
        try:
            rosters = NbaApiSource().fetch_rosters(team_ids, season=args.season)
        except Exception as exc:
            print(f"  [WARN] nba_api rosters failed: {exc}")
    if rosters is None or rosters.empty:
        print("  Building rosters from local DB...")
        rosters = rosters_from_db(team_ids)
        if rosters.empty and args.source != "espn":
            try:
                rosters = NbaApiSource().fetch_rosters(team_ids, season=args.season)
            except Exception:
                pass

    context = compute_schedule_context(games, rosters, game_date=day)
    write_live_bundle(games, injuries, rosters, context)
    write_injury_snapshot(injuries, day=day)

    if not args.skip_stats and args.source != "espn":
        print("  Refreshing player stats (nba_api)...")
        NbaApiSource().refresh_player_stats(
            season=args.season, season_id=args.season_id
        )
    elif args.source == "espn":
        print("  Skipping nba_api stats refresh (espn mode).")

    print(f"=== done via {src_name} ===")
    return 0


def pd_concat_team_ids(games):
    import pandas as pd

    return (
        pd.concat([games["HOME_TEAM_ID"], games["VISITOR_TEAM_ID"]])
        .astype(int)
        .unique()
        .tolist()
    )


if __name__ == "__main__":
    raise SystemExit(main())
