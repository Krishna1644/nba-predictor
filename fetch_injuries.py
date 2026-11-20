"""Thin wrapper — prefer `python scripts/fetch_live.py`."""

from __future__ import annotations

import argparse

from nba_ml.live_sources import EspnSource, NbaApiSource, write_injury_snapshot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="espn", choices=["espn", "nba_api", "cbs"])
    args = ap.parse_args()
    print("--- FETCHING INJURY REPORT ---")
    if args.source in ("nba_api", "cbs"):
        src = NbaApiSource()
    else:
        src = EspnSource()
    print(f"Source: {src.name}")
    injuries = src.fetch_injuries()
    injuries.to_csv("injuries.csv", index=False)
    print(f"[OK] Found {len(injuries)} injured players.")
    write_injury_snapshot(injuries)
    return 0 if not injuries.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
