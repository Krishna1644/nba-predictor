from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime, timedelta
import pandas as pd

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
else:
    print(f"\nNO GAMES FOUND for {today_str}.")