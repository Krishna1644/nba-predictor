import os
import sys
import subprocess
import datetime

# CONFIG
HISTORY_DIR = "history"
DATA_DIR = "data"
LOG_FILE = "logs/pipeline_log.txt"
MODELS_DIR = "models"

def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    print(full_msg)
    # Ensure logs folder exists
    if not os.path.exists("logs"): os.makedirs("logs")
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")

def ensure_folders():
    for folder in [HISTORY_DIR, DATA_DIR, "logs", MODELS_DIR]:
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
        if result.stdout:
            # Print key lines from script output
            for line in result.stdout.strip().split('\n'):
                log(f"  > {line}")
        return True
    except Exception as e:
        log(f"CRITICAL FAIL: {e}")
        return False

def check_model_exists():
    """Pre-flight check: ensure LSTM model artifacts are present."""
    model_path = os.path.join(MODELS_DIR, 'lstm_model.keras')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    
    if not os.path.exists(model_path):
        log(f"MODEL NOT FOUND: {model_path}")
        log("Train the model in Colab first, then place files in models/")
        return False
    if not os.path.exists(scaler_path):
        log(f"SCALER NOT FOUND: {scaler_path}")
        return False
    
    log(f"Model found: {model_path}")
    return True

def main():
    ensure_folders()
    log("=== PIPELINE STARTED (LSTM) ===")
    
    # 1. Pre-flight: Check for trained model
    if not check_model_exists():
        log("ABORTING: No trained model. Run train_lstm.py in Colab first.")
        return
    
    # 2. Fetch Fresh Data
    log(">>> STEP 1: Fetching schedule...")
    if not run_script("get_schedule.py"):
        log("Schedule fetch failed. Aborting.")
        return

    log(">>> STEP 2: Fetching rosters...")
    run_script("fetch_rosters.py")
    
    log(">>> STEP 3: Fetching injuries...")
    run_script("fetch_injuries.py")
    
    log(">>> STEP 4: Refreshing player stats...")
    if not run_script("fetch_player_stats.py"): 
        log("Stats fetch failed. Aborting.")
        return

    # 3. Generate Predictions
    log(">>> STEP 5: Running LSTM predictions...")
    if not run_script("predict_tonight.py"):
        log("Prediction failed.")
        return
    
    log("=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main()