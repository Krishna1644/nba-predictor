import pandas as pd
import sqlite3
import time
from nba_api.stats.endpoints import playergamelog

# CONFIG
SEASON_ID = '2025-26' 
DB_NAME = "nba_stats.db"

def fetch_stats():
    conn = sqlite3.connect(DB_NAME)
    print(f"--- FETCHING STATS ({SEASON_ID}) ---")
    
    # Clean old data to prevent duplicates
    conn.execute("DROP TABLE IF EXISTS player_logs")
    
    try:
        roster_df = pd.read_csv('todays_rosters.csv')
        player_ids = roster_df['PLAYER_ID'].unique()
        print(f"Found {len(player_ids)} players to fetch.")
    except:
        print("CRITICAL: todays_rosters.csv not found.")
        return

    all_frames = []
    count = 0
    
    for pid in player_ids:
        try:
            # timeout=5 helps if the API hangs
            log = playergamelog.PlayerGameLog(player_id=pid, season=SEASON_ID, timeout=5)
            dfs = log.get_data_frames()
            
            if dfs:
                # FETCH FULL SEASON (82 Games)
                all_frames.append(dfs[0].head(82))
                
            count += 1
            print(f"Fetched {count}/{len(player_ids)}", end="\r")
            time.sleep(0.6) 
            
        except Exception as e:
            print(f"Error {pid}: {e}")

    if all_frames:
        final_df = pd.concat(all_frames)
        final_df.to_sql('player_logs', conn, if_exists='replace', index=False)
        print(f"\n[SUCCESS] Saved {len(final_df)} games to database.")
    else:
        print("\n[WARNING] No data fetched.")
        
    conn.close()

if __name__ == "__main__":
    fetch_stats()