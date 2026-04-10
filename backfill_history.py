"""
backfill_history.py — Multi-season historical data fetcher.

Populates nba_stats.db with player game logs from 2015-16 through 2025-26.
Designed to run ONCE, separate from the daily pipeline.

Uses LeagueGameLog endpoint to fetch ALL player logs for an entire 
season in a SINGLE API call — orders of magnitude faster than 
fetching player-by-player.
"""

import pandas as pd
import sqlite3
import time
import random
import numpy as np
from nba_api.stats.endpoints import leaguegamelog

# CONFIG
DB_NAME = "nba_stats.db"
SEASONS = [
    '2015-16', '2016-17', '2017-18', '2018-19', '2019-20',
    '2020-21', '2021-22', '2022-23', '2023-24', '2024-25',
    '2025-26'
]

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

def get_completed_seasons(conn):
    """Check which seasons are already in the DB to support resume."""
    try:
        existing = pd.read_sql(
            "SELECT DISTINCT SEASON_ID FROM player_logs", conn
        )
        return set(existing['SEASON_ID'].astype(str).tolist())
    except:
        return set()

def season_str_to_id(season_str):
    """Convert '2015-16' to '22015' (the NBA API format)."""
    start_year = season_str.split('-')[0]
    return f"2{start_year}"

def fetch_season(season_str, conn):
    """Fetch ALL player game logs for a season in one API call."""
    season_id = season_str_to_id(season_str)
    
    print(f"\n{'='*60}")
    print(f"  SEASON: {season_str} (ID: {season_id})")
    print(f"{'='*60}")
    
    max_retries = 5
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Fetching full season via LeagueGameLog (attempt {attempt})...")
            
            season_logs = leaguegamelog.LeagueGameLog(
                season=season_str,
                season_type_all_star='Regular Season',
                player_or_team_abbreviation='P',  # Player-level logs
                timeout=60
            )
            
            df = season_logs.get_data_frames()[0]
            
            if df.empty:
                print(f"  [WARNING] Empty response for {season_str}")
                return False
            
            # Rename columns to match pipeline's expected schema
            col_map = {
                'PLAYER_ID': 'Player_ID',
                'GAME_ID': 'Game_ID',
                'TEAM_ABBREVIATION': 'TEAM_ABBR',
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            
            # Compute advanced metrics
            df = compute_advanced_metrics(df)
            
            # Checkpoint: commit to DB
            df.to_sql('player_logs', conn, if_exists='append', index=False)
            conn.commit()
            
            unique_players = df['Player_ID'].nunique() if 'Player_ID' in df.columns else '?'
            unique_games = df['Game_ID'].nunique() if 'Game_ID' in df.columns else '?'
            
            print(f"\n  [CHECKPOINT] Season {season_str}:")
            print(f"    Rows:    {len(df)}")
            print(f"    Players: {unique_players}")
            print(f"    Games:   {unique_games}")
            return True
            
        except Exception as e:
            print(f"  Error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                backoff = 60 * attempt
                print(f"  Backing off {backoff}s...")
                time.sleep(backoff)
            else:
                print(f"  FAILED after {max_retries} attempts.")
                return False
    
    return False

def main():
    print("=" * 60)
    print("  NBA HISTORICAL BACKFILL (Fast Mode)")
    print(f"  Target Seasons: {SEASONS[0]} → {SEASONS[-1]}")
    print(f"  Method: LeagueGameLog (1 API call per season)")
    print("=" * 60)
    
    conn = sqlite3.connect(DB_NAME)
    
    # Check for existing data (resume support)
    completed = get_completed_seasons(conn)
    if completed:
        print(f"\n  Already in DB: {sorted(completed)}")
    
    total_added = 0
    
    for season in SEASONS:
        season_id = season_str_to_id(season)
        
        if season_id in completed:
            print(f"\n  SKIPPING {season} — already in database.")
            continue
        
        success = fetch_season(season, conn)
        
        if success:
            total_added += 1
            # Brief pause between seasons to be respectful
            time.sleep(random.uniform(5, 10))
        else:
            print(f"\n  [ERROR] Season {season} failed. "
                  f"Stopping to preserve progress.")
            print(f"  Re-run this script to resume from {season}.")
            break
    
    # Final summary
    final_count = pd.read_sql("SELECT COUNT(*) as n FROM player_logs", conn)
    season_count = pd.read_sql("SELECT COUNT(DISTINCT SEASON_ID) as n FROM player_logs", conn)
    print(f"\n{'='*60}")
    print(f"  BACKFILL COMPLETE")
    print(f"  Seasons added this run: {total_added}")
    print(f"  Total seasons in DB:    {season_count['n'].iloc[0]}")
    print(f"  Total rows in DB:       {final_count['n'].iloc[0]}")
    print(f"{'='*60}")
    
    conn.close()

if __name__ == "__main__":
    main()
