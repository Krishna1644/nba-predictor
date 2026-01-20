import pandas as pd
import sqlite3
import time
from nba_api.stats.endpoints import playergamelog

# Configuration
# Since we are in Jan 2026, the season is 2025-26
SEASON_ID = '2025-26' 

# 1. Setup Database
db_name = "nba_stats.db"
conn = sqlite3.connect(db_name)
print(f"Connected to database: {db_name}")

# CLEAR OLD DATA (Optional)
# For this test, we wipe the table so we don't get duplicates if you run it twice.
conn.execute("DROP TABLE IF EXISTS player_logs")
print("Cleaned up old 'player_logs' table.")

# 2. Load Players
try:
    roster_df = pd.read_csv('todays_rosters.csv')
    # Get unique IDs to avoid duplicate calls
    player_ids = roster_df['PLAYER_ID'].unique()
    print(f"Found {len(player_ids)} unique players to fetch.")
except FileNotFoundError:
    print("Error: todays_rosters.csv not found.")
    exit()

# 3. Fetch Loop
counter = 0
total = len(player_ids)

print("Starting download... (This will take ~3 minutes)")

for pid in player_ids:
    counter += 1
    # Simple progress indicator
    pct = round((counter / total) * 100, 1)
    print(f"[{counter}/{total}] {pct}% - Fetching {pid}...", end="\r")
    
    try:
        # Fetch last 10 games for this player
        # timeout=5 helps if the API hangs
        log = playergamelog.PlayerGameLog(player_id=pid, season=SEASON_ID, timeout=5)
        dfs = log.get_data_frames()
        
        if dfs:
            game_log = dfs[0]
            # Keep last 10 games only
            last_10 = game_log.head(10)
            
            # Save to SQL
            # if_exists='append' adds rows to the table
            last_10.to_sql('player_logs', conn, if_exists='append', index=False)
            
    except Exception as e:
        print(f"\nError fetching {pid}: {e}")
    
    # IMPORTANT: Sleep to respect rate limits
    time.sleep(0.6)

print(f"\nDone! Data saved to {db_name} in table 'player_logs'.")
conn.close()