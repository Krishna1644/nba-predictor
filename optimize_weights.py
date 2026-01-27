import pandas as pd
import sqlite3
import json
import numpy as np
import os
import argparse

# CONFIG
DB_NAME = "nba_stats.db"
WEIGHTS_FILE = "weights.json"
LEARNING_RATE = 0.01

def get_role(avg_min):
    if avg_min is None or np.isnan(avg_min): return "BENCH"
    return "STARTER" if avg_min >= 25 else "BENCH"

def load_weights():
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE, 'r') as f:
            return json.load(f)
    return {}

def ensure_team_exists(weights, team_id):
    tid = str(team_id)
    if tid not in weights:
        weights[tid] = {
            "STARTER": {"L3_Weight": 0.5, "L10_Weight": 0.5},
            "BENCH":   {"L3_Weight": 0.5, "L10_Weight": 0.5}
        }
    return weights

def update_weight_logic(weights, role, team_id, l3_val, l10_val, actual_score):
    tid = str(team_id)
    err_l3 = abs(actual_score - l3_val)
    err_l10 = abs(actual_score - l10_val)
    
    # If L3 was better, nudge L3 up
    if err_l3 < err_l10:
        weights[tid][role]['L3_Weight'] += LEARNING_RATE
        weights[tid][role]['L10_Weight'] -= LEARNING_RATE
    # If L10 was better, nudge L10 up
    elif err_l10 < err_l3:
        weights[tid][role]['L3_Weight'] -= LEARNING_RATE
        weights[tid][role]['L10_Weight'] += LEARNING_RATE
        
    # Clamp to keep it sane (10% - 90%)
    weights[tid][role]['L3_Weight'] = max(0.1, min(0.9, weights[tid][role]['L3_Weight']))
    weights[tid][role]['L10_Weight'] = max(0.1, min(0.9, weights[tid][role]['L10_Weight']))
    return weights

def run_teacher(mode):
    print(f"--- TEACHER RUNNING IN [{mode.upper()}] MODE ---")
    
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM player_logs", conn)
    conn.close()

    # Prep Data
    df.columns = [x.upper() for x in df.columns]

    # [FIX] Map TEAM_ID from rosters
    try:
        rosters = pd.read_csv('todays_rosters.csv')
        pid_to_tid = dict(zip(rosters['PLAYER_ID'], rosters['TeamID']))
        df['TEAM_ID'] = df['PLAYER_ID'].map(pid_to_tid)
        df = df.dropna(subset=['TEAM_ID'])
        df['TEAM_ID'] = df['TEAM_ID'].astype(int)
    except Exception as e:
        print(f"Error mapping teams: {e}")
        return

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by=['PLAYER_ID', 'GAME_DATE'], ascending=[True, True])
    
    weights = load_weights()
    players = df['PLAYER_ID'].unique()
    updates = 0

    for pid in players:
        games = df[df['PLAYER_ID'] == pid].reset_index(drop=True)
        if len(games) < 11: continue
        
        # Determine Range based on Mode
        if mode == 'replay':
            # Replay entire season starting from game 11
            target_indices = range(10, len(games))
        else:
            # Daily: Only look at the most recent game
            target_indices = [len(games) - 1]

        for i in target_indices:
            current_game = games.iloc[i]
            
            # Historical Inputs (The 10 games BEFORE this one)
            history = games.iloc[i-10 : i]
            
            l3_val = history.tail(3)['PTS'].mean()
            l10_val = history['PTS'].mean()
            avg_min = history['MIN'].mean()
            
            role = get_role(avg_min)
            weights = ensure_team_exists(weights, current_game['TEAM_ID'])
            
            weights = update_weight_logic(
                weights, role, current_game['TEAM_ID'], 
                l3_val, l10_val, current_game['PTS']
            )
            updates += 1

    with open(WEIGHTS_FILE, 'w') as f:
        json.dump(weights, f, indent=4)
    print(f"[DONE] Processed {updates} updates.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['daily', 'replay'], required=True)
    args = parser.parse_args()
    run_teacher(args.mode)