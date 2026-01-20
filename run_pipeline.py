import subprocess
import time
import sys
import datetime

# Configuration
LOG_FILE = "pipeline_log.txt"

def log(message):
    """Writes a message to both the console and a log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    
    # Print to console (handle safe printing for Windows)
    try:
        print(full_msg)
    except UnicodeEncodeError:
        print(full_msg.encode('ascii', 'replace').decode('ascii'))
        
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(full_msg + "\n")

def run_script(script_name):
    """Runs a python script and checks if it succeeded."""
    log(f"Starting {script_name}...")
    start_time = time.time()
    
    try:
        # --- THE FIX IS HERE ---
        # We added errors='replace' so "Danté" doesn't crash the reader.
        result = subprocess.run(
            [sys.executable, script_name], 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            errors='replace' 
        )
        
        duration = round(time.time() - start_time, 2)
        
        if result.returncode == 0:
            log(f"{script_name} finished successfully ({duration}s).")
            return True
        else:
            log(f"{script_name} FAILED ({duration}s).")
            log(f"Error Output:\n{result.stderr}")
            return False
            
    except Exception as e:
        log(f"Critical Error running {script_name}: {e}")
        return False

def main():
    log("=========================================")
    log("   STARTING NBA PREDICTION PIPELINE")
    log("=========================================")

    # 1. Get the Schedule
    if not run_script("get_schedule.py"):
        log("Pipeline stopped due to schedule error.")
        return

    # 2. Get Rosters
    if not run_script("fetch_rosters.py"):
        log("Pipeline stopped due to roster error.")
        return

    # 3. Update Database (The slow part)
    if not run_script("fetch_player_stats.py"):
        log("Pipeline stopped due to stats download error.")
        return

    # 4. Retrain the Model
    if not run_script("train_model.py"):
        log("Pipeline stopped due to training error.")
        return

    # 5. Predict
    if not run_script("predict_tonight.py"):
        log("Pipeline stopped due to prediction error.")
        return

    log("=========================================")
    log("   PIPELINE COMPLETE - SUCCESS")
    log("=========================================")

if __name__ == "__main__":
    main()