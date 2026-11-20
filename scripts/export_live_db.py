"""
Export a slim SQLite DB for daily GitHub Actions / serve.

Daily predict only needs:
  - current live season (priors, form, rest)
  - previous season (gp/(gp+10) blend)

Full history stays in local nba_stats.db for training only.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import pandas as pd

# Allow `python scripts/export_live_db.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nba_ml.features import LIVE_SEASON, TEST_SEASON, TRAIN_SEASON_MAX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="nba_stats.db")
    ap.add_argument("--dst", default="nba_stats_live.db")
    ap.add_argument(
        "--seasons",
        nargs="*",
        default=None,
        help="SEASON_IDs to keep (default: train_max, report, live if present)",
    )
    args = ap.parse_args()

    src = sqlite3.connect(args.src)
    have = set(
        pd.read_sql("SELECT DISTINCT SEASON_ID FROM player_logs", src)["SEASON_ID"]
        .astype(str)
        .tolist()
    )
    default = [TRAIN_SEASON_MAX, TEST_SEASON, LIVE_SEASON]  # 22024, 22025, 22026
    want = [str(s) for s in (args.seasons or default) if str(s) in have]
    if not want:
        raise SystemExit(f"None of requested seasons in {args.src}. Have: {sorted(have)}")

    q = f"SELECT * FROM player_logs WHERE SEASON_ID IN ({','.join('?' * len(want))})"
    df = pd.read_sql(q, src, params=want)
    src.close()

    if os.path.exists(args.dst):
        os.remove(args.dst)
    dst = sqlite3.connect(args.dst)
    df.to_sql("player_logs", dst, if_exists="replace", index=False)
    dst.execute("CREATE INDEX IF NOT EXISTS ix_logs_season ON player_logs(SEASON_ID)")
    dst.execute("CREATE INDEX IF NOT EXISTS ix_logs_player ON player_logs(Player_ID)")
    dst.commit()
    dst.close()

    mb = os.path.getsize(args.dst) / (1024 * 1024)
    print(f"Wrote {args.dst} ({mb:.1f} MB) seasons={want} rows={len(df)}")

    # Also write tracked gzip for GitHub Actions (no Release upload).
    import gzip
    import shutil

    gz_path = os.path.join("data", "nba_stats_live.db.gz")
    os.makedirs("data", exist_ok=True)
    with open(args.dst, "rb") as f_in, gzip.open(gz_path, "wb", compresslevel=9) as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"Wrote {gz_path} ({os.path.getsize(gz_path) / (1024 * 1024):.1f} MB)")


if __name__ == "__main__":
    main()
