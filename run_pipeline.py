import os
import sys
import subprocess
import datetime
import glob
import pandas as pd

# CONFIG
HISTORY_DIR = "history"
DATA_DIR = "data"
LOG_FILE = "logs/pipeline_log.txt"

def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    print(full_msg)
    # Ensure logs folder exists
    if not os.path.exists("logs"): os.makedirs("logs")
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")

def ensure_folders():
    for folder in [HISTORY_DIR, DATA_DIR, "logs"]:
        if not os.path.exists(folder):
            os.makedirs(folder)

def run_script(script_name, args=[]):
    cmd = [sys.executable, script_name] + args
    log(f"Running: {script_name} {' '.join(args)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"ERROR in {script_name}:\n{result.stderr}")
            return False
        return True
    except Exception as e:
        log(f"CRITICAL FAIL: {e}")
        return False

def check_last_run_status():
    """
    Returns 'DAILY' if we have yesterday's file.
    Returns 'RESET' if the file is missing or old.
    """
    files = glob.glob(f"{HISTORY_DIR}/*.csv")
    if not files:
        log("No history found. Triggering RESET.")
        return "RESET"
    
    latest_file = max(files, key=os.path.getctime)
    
    try:
        filename = os.path.basename(latest_file)
        date_str = filename.replace("preds_", "").replace(".csv", "")
        last_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        today = datetime.date.today()
        
        diff = (today - last_date).days
        
        if diff <= 1:
            log(f"Found recent history ({last_date}). Running DAILY update.")
            return "DAILY"
        else:
            log(f"Last run was {diff} days ago. Gap too large. Triggering RESET.")
            return "RESET"
            
    except Exception as e:
        log(f"Error checking dates ({e}). Defaulting to RESET.")
        return "RESET"

def main():
    ensure_folders()
    log("=== PIPELINE STARTED ===")
    
    # 1. ALWAYS Fetch Fresh Data (Rosters, Injuries, Stats)
    run_script("get_schedule.py")
    run_script("fetch_rosters.py")
    run_script("fetch_injuries.py")
    
    # Critical: Stats fetcher now grabs 82 games
    if not run_script("fetch_player_stats.py"): 
        log("Stats fetch failed. Aborting.")
        return

    # 2. DECIDE MODE
    mode = check_last_run_status()
    
    # 3. RUN TEACHER
    if mode == "RESET":
        log(">>> ENTERING PART 1: SEASON REPLAY")
        if not run_script("optimize_weights.py", ["--mode", "replay"]): return
    else:
        log(">>> ENTERING PART 2: DAILY UPDATE")
        if not run_script("optimize_weights.py", ["--mode", "daily"]): return

    # 4. PREDICT
    log(">>> GENERATING PREDICTIONS")
    if not run_script("predict_tonight.py"): return
    
    log("=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main()