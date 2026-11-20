"""
Daily pipeline: extract live slate → predict → write data/live/.

  python run_pipeline.py                # auto: nba_api then ESPN (local)
  python run_pipeline.py --source espn  # cloud / GitHub Actions
  python run_pipeline.py --predict-only # skip extract; use existing data/live
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys

HISTORY_DIR = "history"
DATA_DIR = "data"
LOG_FILE = "logs/pipeline_log.txt"
MODELS_DIR = "models"


def log(message: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    print(full_msg)
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")


def ensure_folders() -> None:
    for folder in [HISTORY_DIR, DATA_DIR, "logs", MODELS_DIR, "data/live", "data/injury_snapshots"]:
        os.makedirs(folder, exist_ok=True)


def run_script(script_name: str, args: list[str] | None = None) -> bool:
    cmd = [sys.executable, script_name] + (args or [])
    log(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"ERROR in {script_name}:\n{result.stderr}")
            if result.stdout:
                log(f"stdout:\n{result.stdout}")
            return False
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                log(f"  > {line}")
        return True
    except Exception as e:
        log(f"CRITICAL FAIL: {e}")
        return False


def check_model_exists() -> bool:
    model_path = os.path.join(MODELS_DIR, "logistic_baseline.pkl")
    elo_path = os.path.join(MODELS_DIR, "elo_ratings.json")
    if not os.path.exists(model_path):
        log(f"MODEL NOT FOUND: {model_path}")
        log("Run: python build_datasets.py && python train_baseline.py")
        return False
    if not os.path.exists(elo_path):
        log(f"ELO NOT FOUND: {elo_path}")
        return False
    log(f"Model found: {model_path}")
    return True


def stage_live_inputs_to_root() -> None:
    """predict_matchup currently reads root CSVs; copy from data/live if needed."""
    live = os.path.join("data", "live")
    for name in (
        "todays_games.csv",
        "todays_rosters.csv",
        "injuries.csv",
        "schedule_context.csv",
    ):
        live_path = os.path.join(live, name)
        if os.path.exists(live_path) and not os.path.exists(name):
            import shutil

            shutil.copy2(live_path, name)
            log(f"Copied {live_path} → {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default="auto",
        choices=["auto", "espn", "nba_api"],
        help="Live extract source (espn for Actions)",
    )
    ap.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip extract; predict from existing data/live CSVs",
    )
    ap.add_argument("--skip-stats", action="store_true", help="Pass through to fetch_live")
    args = ap.parse_args()

    ensure_folders()
    log("=== PIPELINE STARTED (logistic matchup) ===")

    if not check_model_exists():
        log("ABORTING: No trained model.")
        return 1

    if not args.predict_only:
        log(f">>> STEP 1: Extract live slate (source={args.source})...")
        fetch_args = ["--source", args.source]
        if args.skip_stats or args.source == "espn":
            fetch_args.append("--skip-stats")
        if not run_script(os.path.join("scripts", "fetch_live.py"), fetch_args):
            log("Extract failed. Aborting.")
            return 1
    else:
        log(">>> STEP 1: Skipped extract (--predict-only)")
        stage_live_inputs_to_root()

    log(">>> STEP 2: Running logistic matchup predictions...")
    if not run_script("predict_matchup.py"):
        log("Prediction failed.")
        return 1

    log("=== PIPELINE COMPLETE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
