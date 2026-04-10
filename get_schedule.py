from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import numpy as np

# CONFIG
DB_NAME = "nba_stats.db"

# 1. Get "NBA Time" (ET)
# Adjusting to ensure we get the correct "NBA Day"
utc_now = datetime.utcnow()
nba_time = utc_now - timedelta(hours=5) 
today_str = nba_time.strftime('%Y-%m-%d')

print(f"\nCHECKING SCHEDULE FOR: {today_str}")

# 2. Call the API
board = scoreboardv2.ScoreboardV2(game_date=today_str)
games_df = board.game_header.get_data_frame()

# 3. Save
if not games_df.empty:
    # Select columns
    summary = games_df[['GAME_DATE_EST', 'GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'GAME_STATUS_TEXT']]
    
    # --- THE FIX: REMOVE DUPLICATES ---
    # The API returns one row per TV station (ESPN, TNT, Local).
    # We drop duplicates based on 'GAME_ID' so we only get the game once.
    summary = summary.drop_duplicates(subset=['GAME_ID'])
    
    print(f"\nFound {len(summary)} unique games.")
    print(summary[['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']].head(10))
    
    summary.to_csv('todays_games.csv', index=False)
    print("\nSaved clean schedule to 'todays_games.csv'")
    
    # --- SCHEDULE CONTEXT: REST DAYS & FATIGUE ---
    print("\n--- COMPUTING SCHEDULE CONTEXT ---")
    
    try:
        conn = sqlite3.connect(DB_NAME)
        
        # Get all team IDs playing tonight
        team_ids = pd.concat([summary['HOME_TEAM_ID'], summary['VISITOR_TEAM_ID']]).unique()
        
        # We need to map TEAM_ID from player_logs via MATCHUP parsing
        # Load all game dates from DB grouped by team
        all_logs = pd.read_sql("SELECT GAME_DATE, MATCHUP, Player_ID FROM player_logs", conn)
        conn.close()
        
        if not all_logs.empty:
            # Parse team abbreviation from MATCHUP (e.g. "LAL vs. BOS" -> "LAL")
            all_logs['GAME_DATE_PARSED'] = pd.to_datetime(all_logs['GAME_DATE'], format='mixed')
            
            today_dt = pd.to_datetime(today_str)
            
            context_rows = []
            
            for team_id in team_ids:
                # Get rosters to map team_id -> player_ids for this team
                try:
                    rosters = pd.read_csv('todays_rosters.csv')
                    team_players = rosters[rosters['TeamID'] == team_id]['PLAYER_ID'].unique()
                    
                    # Get this team's game dates from DB
                    team_logs = all_logs[all_logs['Player_ID'].isin(team_players)]
                    team_dates = team_logs['GAME_DATE_PARSED'].drop_duplicates().sort_values(ascending=False)
                    
                    if len(team_dates) > 0:
                        last_game_date = team_dates.iloc[0]
                        rest_days = (today_dt - last_game_date).days
                        is_b2b = 1 if rest_days == 1 else 0
                        
                        # Games in last 7 days
                        seven_days_ago = today_dt - timedelta(days=7)
                        games_last_7 = len(team_dates[team_dates >= seven_days_ago])
                    else:
                        rest_days = 3  # Default if no history
                        is_b2b = 0
                        games_last_7 = 0
                        
                except Exception as e:
                    print(f"  Warning: Could not compute context for team {team_id}: {e}")
                    rest_days = 3
                    is_b2b = 0
                    games_last_7 = 0
                
                context_rows.append({
                    'TEAM_ID': team_id,
                    'rest_days': rest_days,
                    'is_back_to_back': is_b2b,
                    'games_last_7': games_last_7
                })
            
            context_df = pd.DataFrame(context_rows)
            context_df.to_csv('schedule_context.csv', index=False)
            print(f"[OK] Saved schedule context for {len(context_df)} teams.")
            print(context_df.to_string(index=False))
        else:
            print("[WARNING] No game logs in DB to compute schedule context.")
            
    except Exception as e:
        print(f"[WARNING] Could not compute schedule context: {e}")

else:
    print(f"\nNO GAMES FOUND for {today_str}.")