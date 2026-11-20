"""Thin wrapper — prefer `python scripts/fetch_live.py` (local / home IP)."""

from __future__ import annotations

from nba_ml.live_sources import NbaApiSource


def fetch_stats() -> None:
    NbaApiSource().refresh_player_stats(season="2026-27", season_id="22026")


if __name__ == "__main__":
    fetch_stats()
