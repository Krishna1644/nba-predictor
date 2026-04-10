import pandas as pd
import sqlite3
import numpy as np
from nba_api.stats.endpoints import leaguegamelog

# CONFIG
SEASON_STR = '2025-26' 
SEASON_ID = '22025'
DB_NAME = "nba_stats.db"

def compute_advanced_metrics(df):
    """Compute eFG%, TS%, and TOV% from raw box score columns."""
    df['EFG_PCT'] = np.where(
        df['FGA'] > 0,
        (df['FGM'] + 0.5 * df['FG3M']) / df['FGA'],
        0.0
    )
    tsa = 2 * (df['FGA'] + 0.44 * df['FTA'])
    df['TS_PCT'] = np.where(tsa > 0, df['PTS'] / tsa, 0.0)
    tov_denom = df['FGA'] + 0.44 * df['FTA'] + df['TOV']
    df['TOV_PCT'] = np.where(tov_denom > 0, df['TOV'] / tov_denom, 0.0)
    return df

def fetch_stats():
    conn = sqlite3.connect(DB_NAME)
    print(f"--- FETCHING STATS ({SEASON_STR}) ---")
    
    try:
        # Fetch entire season in one API call
        print("  Fetching daily LeagueGameLog...")
        season_logs = leaguegamelog.LeagueGameLog(
            season=SEASON_STR,
            season_type_all_star='Regular Season',
            player_or_team_abbreviation='P',
            timeout=60
        )
        df = season_logs.get_data_frames()[0]
        
        if df.empty:
            print("  [WARNING] Empty response.")
            return

        # Rename columns to match existing schema
        col_map = {
            'PLAYER_ID': 'Player_ID',
            'GAME_ID': 'Game_ID',
            'TEAM_ABBREVIATION': 'TEAM_ABBR',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        
        # Compute advanced metrics
        df = compute_advanced_metrics(df)
        
        # Delete the CURRENT season from the database so we can safely refresh it
        # without destroying the 10 years of historical backfill data
        conn.execute(f"DELETE FROM player_logs WHERE SEASON_ID = '{SEASON_ID}'")
        conn.commit()
        
        # Append updated current season
        df.to_sql('player_logs', conn, if_exists='append', index=False)
        print(f"\n[SUCCESS] Updated {len(df)} games for {SEASON_STR} in database.")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to fetch stats: {e}")
        
    conn.close()

if __name__ == "__main__":
    fetch_stats()