import subprocess
import time
import sys
import datetime

LOG_FILE = "pipeline_log.txt"

def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    
    try:
        print(full_msg)
    except:
        pass # Silence print errors
        
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(full_msg + "\n")

def run_script(script_name):
    log(f"Starting {script_name}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name], 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            errors='replace'
        )
        
        duration = round(time.time() - start_time, 2)
        
        if result.returncode == 0:
            log(f"FINISHED: {script_name} ({duration}s)")
            return True
        else:
            log(f"FAILED: {script_name} ({duration}s)")
            log(f"Error Output:\n{result.stderr}")
            return False
            
    except Exception as e:
        log(f"CRITICAL ERROR: {script_name}: {e}")
        return False

def main():
    log("-----------------------------------------")
    log("   STARTING NBA PIPELINE")
    log("-----------------------------------------")

    # 1. Schedule
    if not run_script("get_schedule.py"): return

    # 2. Rosters
    if not run_script("fetch_rosters.py"): return
    
    # 3. Injuries (NEW STEP)
    if not run_script("fetch_injuries.py"): return

    # 4. Stats
    if not run_script("fetch_player_stats.py"): return

    # 5. Train
    if not run_script("train_model.py"): return

    # 6. Predict
    if not run_script("predict_tonight.py"): return

    log("-----------------------------------------")
    log("   PIPELINE COMPLETE")
    log("-----------------------------------------")

if __name__ == "__main__":
    main()