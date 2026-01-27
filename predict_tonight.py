import pandas as pd
import sqlite3
import json
import datetime
import os

# CONFIG
WEIGHTS_FILE = 'weights.json'
HISTORY_DIR = 'history'
DB_NAME = 'nba_stats.db'

def get_weights_safe(weights, team_id, role, location):
    tid = str(team_id)
    # Default if new team or missing location data
    if tid not in weights:
        return 0.5, 0.5 
    
    # Check for Home/Away split structure
    if location in weights[tid]:
         return weights[tid][location][role]['L3_Weight'], weights[tid][location][role]['L10_Weight']
    
    # Fallback for old structure (shouldn't happen after retrain)
    return 0.5, 0.5

def main():
    print("--- PREDICTING TONIGHT ---")
    
    # Load Weights
    with open(WEIGHTS_FILE, 'r') as f:
        weights = json.load(f)
        
    rosters = pd.read_csv('todays_rosters.csv')
    games = pd.read_csv('todays_games.csv')
    conn = sqlite3.connect(DB_NAME)
    
    # Injury Filter
    try:
        import unicodedata
        def normalize_name(name):
            if not isinstance(name, str): return ""
            return ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn').lower().strip()
            
        injuries = pd.read_csv('injuries.csv')
        # Normalize the injury list names
        injured_players = set(injuries['Player'].apply(normalize_name))
    except:
        injured_players = set()

    # Pre-map Team -> Location (HOME/AWAY) for tonight
    team_locations = {}
    if not games.empty:
        for _, row in games.iterrows():
            team_locations[row['HOME_TEAM_ID']] = 'HOME'
            team_locations[row['VISITOR_TEAM_ID']] = 'AWAY'

    predictions = []
    
    for _, row in rosters.iterrows():
        pid = row['PLAYER_ID']
        name = row['PLAYER']
        team_id = row['TeamID']
        
        # Check normalized name against normalized injury set
        if normalize_name(name) in injured_players: continue
            
        # Get Stats (Newest First for Prediction)
        query = f"SELECT * FROM player_logs WHERE PLAYER_ID = {pid} ORDER BY GAME_DATE DESC LIMIT 20"
        df_p = pd.read_sql(query, conn)
        
        if len(df_p) < 10: continue 
        
        # Calculate Inputs
        l3 = df_p.head(3)['PTS'].mean()
        l10 = df_p.head(10)['PTS'].mean()
        avg_min = df_p['MIN'].mean()
        role = "STARTER" if avg_min >= 25 else "BENCH"
        
        # Determine Location
        location = team_locations.get(team_id, "HOME") # Default to HOME if unknown? Or handle error?
        
        # Apply The Brain
        w_l3, w_l10 = get_weights_safe(weights, team_id, role, location)
        pred = (l3 * w_l3) + (l10 * w_l10)
        
        predictions.append({
            "Player": name,
            "TeamID": team_id,
            "Predicted_PTS": round(pred, 1),
            "Role": role,
            "L3": round(l3, 1),
            "L10": round(l10, 1),
            "Weight_Form": round(w_l3, 2)
        })

    # Save
    if predictions:
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        
        # 1. Save to History (For the Manager)
        if not os.path.exists(HISTORY_DIR): os.makedirs(HISTORY_DIR)
        pd.DataFrame(predictions).to_csv(f"{HISTORY_DIR}/preds_{today_str}.csv", index=False)
        
        # 2. Save to Dashboard
        pd.DataFrame(predictions).to_csv("final_predictions.csv", index=False)
        print(f"[SUCCESS] Predictions generated.")
    else:
        print("[FAIL] No predictions generated.")

if __name__ == "__main__":
    main()