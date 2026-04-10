"""
build_sequences.py — Sequential Data Generator for LSTM.

Transforms flat player_logs from the SQLite DB into 3D tensors:
    X: (N, 10, 24) — 10-game lookback windows with 24 features per timestep
    y: (N,)         — binary win/loss label for the target game

Features per timestep (24 total):
    Team sums (6):        pts, reb, ast, stl, blk, tov
    Star-absence max (3): max_pts, max_reb, max_ast
    Efficiency (6):       fg_pct, fg3_pct, ft_pct, efg_pct, ts_pct, tov_pct
    Context (2):          plus_minus, is_home
    Schedule fatigue (3): rest_days, is_back_to_back, games_last_7
    Injury impact (1):    missing_starter_minutes
    Strength (3):         team_elo, opp_win_pct, opp_pts_allowed_avg

Output: X.npy, y.npy, models/scaler.pkl, models/elo_ratings.json, models/game_context.pkl
"""

import pandas as pd
import sqlite3
import numpy as np
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

# CONFIG
DB_NAME = "nba_stats.db"
LOOKBACK = 10
MODELS_DIR = "models"
STARTER_MIN_THRESHOLD = 25  # Minutes per game threshold for "starter"

# The exact order of features — must match inference in predict_tonight.py
FEATURE_COLUMNS = [
    'team_pts', 'team_reb', 'team_ast', 'team_stl', 'team_blk', 'team_tov',
    'max_pts', 'max_reb', 'max_ast',
    'fg_pct', 'fg3_pct', 'ft_pct', 'efg_pct', 'ts_pct', 'tov_pct',
    'plus_minus', 'is_home',
    'rest_days', 'is_back_to_back', 'games_last_7',
    'missing_starter_minutes',
    'team_elo', 'opp_win_pct', 'opp_pts_allowed_avg'
]

assert len(FEATURE_COLUMNS) == 24, f"Expected 24 features, got {len(FEATURE_COLUMNS)}"


def compute_advanced_metrics(df):
    """Compute eFG%, TS%, TOV% if not already present."""
    if 'EFG_PCT' not in df.columns:
        df['EFG_PCT'] = np.where(
            df['FGA'] > 0,
            (df['FGM'] + 0.5 * df['FG3M']) / df['FGA'],
            0.0
        )
    if 'TS_PCT' not in df.columns:
        tsa = 2 * (df['FGA'] + 0.44 * df['FTA'])
        df['TS_PCT'] = np.where(tsa > 0, df['PTS'] / tsa, 0.0)
    if 'TOV_PCT' not in df.columns:
        tov_denom = df['FGA'] + 0.44 * df['FTA'] + df['TOV']
        df['TOV_PCT'] = np.where(tov_denom > 0, df['TOV'] / tov_denom, 0.0)
    return df


def extract_team_from_matchup(matchup):
    """Extract team abbreviation from matchup string like 'LAL vs. BOS' or 'LAL @ BOS'."""
    if isinstance(matchup, str):
        return matchup.split(' ')[0].strip()
    return ''


def compute_elo_and_strength(team_games):
    """
    Compute rolling Elo ratings, opponent win%, and opponent defensive rating
    for each team-game.
    
    Elo: Standard Elo system (K=20), with 75% carry-over between seasons.
    Opp Win%: The opponent's season record going into each game.
    Opp Pts Allowed Avg: Opponent's season average points allowed per game.
    
    Adds 'team_elo', 'opp_win_pct', and 'opp_pts_allowed_avg' columns.
    Returns (team_games, current_elo_dict).
    """
    K = 20
    SEASON_CARRY = 0.75  # How much Elo carries over between seasons
    
    elo = {}              # team_abbr -> current Elo rating
    season_wins = {}      # team_abbr -> wins this season
    season_games = {}     # team_abbr -> games played this season
    season_pts_allowed = {}  # team_abbr -> total points allowed this season
    current_season = None
    
    # Get chronological order of unique Game_IDs
    game_order = (
        team_games.drop_duplicates('Game_ID')
        .sort_values('game_date')['Game_ID']
        .values
    )
    
    # Group by Game_ID for fast lookup
    game_groups = team_games.groupby('Game_ID')
    
    # Storage: (Game_ID, TEAM_ABBR) -> pre-game value
    game_elo = {}
    game_opp_wp = {}
    game_opp_def = {}   # (Game_ID, TEAM_ABBR) -> opponent's avg pts allowed
    
    for gid in game_order:
        group = game_groups.get_group(gid)
        if len(group) != 2:
            continue
        
        row_a, row_b = group.iloc[0], group.iloc[1]
        team_a = row_a['TEAM_ABBR']
        team_b = row_b['TEAM_ABBR']
        season = row_a['season_id']
        win_a = float(row_a['win'])
        
        # Season reset: regress Elo toward 1500
        if season != current_season:
            current_season = season
            for t in list(elo.keys()):
                elo[t] = SEASON_CARRY * elo[t] + (1 - SEASON_CARRY) * 1500
            season_wins.clear()
            season_games.clear()
            season_pts_allowed.clear()
        
        # Initialize new teams
        for t in [team_a, team_b]:
            elo.setdefault(t, 1500)
            season_wins.setdefault(t, 0)
            season_games.setdefault(t, 0)
            season_pts_allowed.setdefault(t, 0.0)
        
        # Store pre-game Elo
        game_elo[(gid, team_a)] = elo[team_a]
        game_elo[(gid, team_b)] = elo[team_b]
        
        # Store opponent's pre-game season win%
        game_opp_wp[(gid, team_a)] = season_wins[team_b] / max(season_games[team_b], 1)
        game_opp_wp[(gid, team_b)] = season_wins[team_a] / max(season_games[team_a], 1)
        
        # Store opponent's pre-game avg points allowed
        game_opp_def[(gid, team_a)] = season_pts_allowed[team_b] / max(season_games[team_b], 1)
        game_opp_def[(gid, team_b)] = season_pts_allowed[team_a] / max(season_games[team_a], 1)
        
        # Update Elo
        e_a = 1 / (1 + 10 ** ((elo[team_b] - elo[team_a]) / 400))
        elo[team_a] += K * (win_a - e_a)
        elo[team_b] += K * ((1 - win_a) - (1 - e_a))
        
        # Update season records
        season_games[team_a] += 1
        season_games[team_b] += 1
        if win_a:
            season_wins[team_a] += 1
        else:
            season_wins[team_b] += 1
        
        # Update points allowed (team_a allowed team_b's score, and vice versa)
        pts_a = float(row_a['team_pts'])  # points scored by team_a
        pts_b = float(row_b['team_pts'])  # points scored by team_b
        season_pts_allowed[team_a] += pts_b  # team_a allowed team_b's points
        season_pts_allowed[team_b] += pts_a  # team_b allowed team_a's points
    
    # Map back to team_games DataFrame
    team_games['team_elo'] = team_games.apply(
        lambda r: game_elo.get((r['Game_ID'], r['TEAM_ABBR']), 1500), axis=1
    )
    team_games['opp_win_pct'] = team_games.apply(
        lambda r: game_opp_wp.get((r['Game_ID'], r['TEAM_ABBR']), 0.5), axis=1
    )
    team_games['opp_pts_allowed_avg'] = team_games.apply(
        lambda r: game_opp_def.get((r['Game_ID'], r['TEAM_ABBR']), 105.0), axis=1
    )
    
    # Build game_context for inference lookback
    game_context = {}
    for (gid, team), e in game_elo.items():
        game_context[(gid, team)] = {
            'team_elo': float(e),
            'opp_win_pct': float(game_opp_wp.get((gid, team), 0.5)),
            'opp_pts_allowed_avg': float(game_opp_def.get((gid, team), 105.0))
        }
    
    # Save context files
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    elo_path = os.path.join(MODELS_DIR, 'elo_ratings.json')
    with open(elo_path, 'w') as f:
        json.dump({k: round(v, 1) for k, v in elo.items()}, f, indent=2)
    
    context_path = os.path.join(MODELS_DIR, 'game_context.pkl')
    joblib.dump(game_context, context_path)
    
    print(f"    Elo range: [{min(elo.values()):.0f}, {max(elo.values()):.0f}]")
    print(f"    [SAVED] {elo_path} ({len(elo)} teams)")
    print(f"    [SAVED] {context_path} ({len(game_context)} game-team pairs)")
    
    return team_games, elo


def build_team_game_features(player_logs):
    """
    Aggregate player-level logs into team-game-level features.
    Returns a DataFrame with one row per team-game.
    """
    print("  Aggregating player stats to team-game level...")
    
    # Parse team abbreviation and determine home/away
    player_logs['TEAM_ABBR'] = player_logs['MATCHUP'].apply(extract_team_from_matchup)
    player_logs['IS_HOME'] = player_logs['MATCHUP'].apply(
        lambda x: 1 if 'vs.' in str(x) else 0
    )
    
    # Parse game date
    player_logs['GAME_DATE_DT'] = pd.to_datetime(player_logs['GAME_DATE'], format='mixed')
    
    # Parse win/loss
    player_logs['WIN'] = (player_logs['WL'] == 'W').astype(int)
    
    # Compute per-player season-average minutes for missing_starter_minutes
    # Group by player and season to get their average minutes
    player_logs['MIN_NUMERIC'] = pd.to_numeric(player_logs['MIN'], errors='coerce').fillna(0)
    player_avg_min = player_logs.groupby(['Player_ID', 'SEASON_ID'])['MIN_NUMERIC'].transform('mean')
    player_logs['PLAYER_AVG_MIN'] = player_avg_min
    
    # --- AGGREGATION ---
    # Group by Game_ID and TEAM_ABBR (using first() for shared game-level cols)
    team_games = player_logs.groupby(['Game_ID', 'TEAM_ABBR']).agg(
        # Team sums
        team_pts=('PTS', 'sum'),
        team_reb=('REB', 'sum'),
        team_ast=('AST', 'sum'),
        team_stl=('STL', 'sum'),
        team_blk=('BLK', 'sum'),
        team_tov=('TOV', 'sum'),
        # Star-absence maxes
        max_pts=('PTS', 'max'),
        max_reb=('REB', 'max'),
        max_ast=('AST', 'max'),
        # Efficiency (team averages of individual rates)
        fg_pct=('FG_PCT', 'mean'),
        fg3_pct=('FG3_PCT', 'mean'),
        ft_pct=('FT_PCT', 'mean'),
        efg_pct=('EFG_PCT', 'mean'),
        ts_pct=('TS_PCT', 'mean'),
        tov_pct=('TOV_PCT', 'mean'),
        # Context
        plus_minus=('PLUS_MINUS', 'mean'),
        is_home=('IS_HOME', 'first'),
        # Label (kept for reference)
        win=('WIN', 'first'),
        # Date for sorting
        game_date=('GAME_DATE_DT', 'first'),
        # Season ID for reference
        season_id=('SEASON_ID', 'first'),
    ).reset_index()
    
    # --- OPPONENT INFO (via self-join) ---
    # Each Game_ID has exactly 2 teams. Derive opponent via self-joining.
    print("  Deriving opponent info per game...")
    opp_info = team_games[['Game_ID', 'TEAM_ABBR', 'team_pts']].copy()
    opp_info.columns = ['Game_ID', 'OPP_ABBR', 'opponent_pts']
    
    # For each game, merge to get the OTHER team's info
    team_games = team_games.merge(opp_info, on='Game_ID', how='left')
    # Filter out self-matches (keep only the opponent row)
    team_games = team_games[team_games['TEAM_ABBR'] != team_games['OPP_ABBR']].copy()
    
    # MoV = team's points minus opponent's points (kept for reference)
    team_games['mov'] = team_games['team_pts'] - team_games['opponent_pts']
    
    # --- MISSING STARTER MINUTES (Vectorized) ---
    # Pre-compute "starters" (>25 MPG) per team-season, then check
    # which starters are missing from each game's box score.
    print("  Computing missing_starter_minutes...")
    
    # Step 1: Get season-average minutes per player per team-season
    player_season_avg = player_logs.groupby(
        ['TEAM_ABBR', 'SEASON_ID', 'Player_ID']
    )['MIN_NUMERIC'].mean().reset_index()
    player_season_avg.columns = ['TEAM_ABBR', 'SEASON_ID', 'Player_ID', 'avg_min']
    
    # Step 2: Filter to starters only (>25 MPG)
    starters = player_season_avg[player_season_avg['avg_min'] >= STARTER_MIN_THRESHOLD]
    
    # Step 3: Get set of player IDs who played in each game
    played_per_game = player_logs.groupby(
        ['Game_ID', 'TEAM_ABBR']
    )['Player_ID'].apply(set).reset_index()
    played_per_game.columns = ['Game_ID', 'TEAM_ABBR', 'played_ids']
    
    # Step 4: Merge with team_games to get season_id, then compute missing minutes
    team_games_with_played = team_games.merge(
        played_per_game, on=['Game_ID', 'TEAM_ABBR'], how='left'
    )
    
    def calc_missing_min(row):
        team = row['TEAM_ABBR']
        season = row['season_id']
        played = row.get('played_ids', set()) or set()
        
        team_starters = starters[
            (starters['TEAM_ABBR'] == team) & 
            (starters['SEASON_ID'] == season)
        ]
        
        missing = 0.0
        for _, s in team_starters.iterrows():
            if s['Player_ID'] not in played:
                missing += s['avg_min']
        return missing
    
    team_games['missing_starter_minutes'] = team_games_with_played.apply(calc_missing_min, axis=1)
    
    # --- SCHEDULE CONTEXT (Vectorized) ---
    print("  Computing schedule context (rest_days, b2b, games_last_7)...")
    
    team_games = team_games.sort_values(['TEAM_ABBR', 'game_date']).reset_index(drop=True)
    
    # Rest days: diff between consecutive game dates per team
    team_games['prev_game_date'] = team_games.groupby('TEAM_ABBR')['game_date'].shift(1)
    team_games['rest_days'] = (
        team_games['game_date'] - team_games['prev_game_date']
    ).dt.days.fillna(3).astype(int)  # Default 3 for first game
    
    team_games['is_back_to_back'] = (team_games['rest_days'] == 1).astype(int)
    
    # Games in last 7 days: rolling count per team
    games_last_7 = []
    for team_abbr in team_games['TEAM_ABBR'].unique():
        team_mask = team_games['TEAM_ABBR'] == team_abbr
        team_df = team_games[team_mask]
        dates = team_df['game_date'].values
        
        for i, idx in enumerate(team_df.index):
            current = pd.Timestamp(dates[i])
            seven_ago = current - timedelta(days=7)
            g7 = sum(1 for d in dates[:i] if pd.Timestamp(d) >= seven_ago)
            games_last_7.append((idx, g7))
    
    for idx, val in games_last_7:
        team_games.at[idx, 'games_last_7'] = val
    
    # Drop helper columns
    team_games = team_games.drop(columns=['prev_game_date'], errors='ignore')
    
    # --- ELO RATINGS & OPPONENT WIN% ---
    print("  Computing Elo ratings and opponent win%...")
    team_games, current_elo = compute_elo_and_strength(team_games)
    
    # Drop OPP_ABBR now that Elo computation is done
    team_games = team_games.drop(columns=['OPP_ABBR'], errors='ignore')
    
    print(f"  Built {len(team_games)} team-game rows.")
    return team_games, current_elo


def build_sequences(team_games, lookback=LOOKBACK):
    """
    Build rolling window sequences from team-game features.
    
    For each team, at game index i >= lookback:
        X[n] = features from games [i-lookback, i)  (shape: lookback x 21)
        y[n] = Margin of Victory at game i (continuous float)
    """
    print(f"\n  Building {lookback}-game lookback sequences...")
    
    X_sequences = []
    y_labels = []
    
    for team_abbr in team_games['TEAM_ABBR'].unique():
        team_df = team_games[team_games['TEAM_ABBR'] == team_abbr].copy()
        team_df = team_df.sort_values('game_date').reset_index(drop=True)
        
        if len(team_df) < lookback + 1:
            continue
        
        feature_matrix = team_df[FEATURE_COLUMNS].values
        labels = team_df['win'].values
        
        for i in range(lookback, len(team_df)):
            X_sequences.append(feature_matrix[i - lookback:i])
            y_labels.append(labels[i])
    
    X = np.array(X_sequences, dtype=np.float32)
    y = np.array(y_labels, dtype=np.float32)
    
    print(f"  Generated {len(X)} sequences.")
    return X, y


def main():
    print("=" * 60)
    print("  BUILD SEQUENCES — LSTM Data Generator")
    print("=" * 60)
    
    # 1. Load data
    conn = sqlite3.connect(DB_NAME)
    player_logs = pd.read_sql("SELECT * FROM player_logs", conn)
    conn.close()
    
    print(f"\n  Loaded {len(player_logs)} player-game rows from DB.")
    
    # 2. Fill NaN in raw data (API returns NaN for stats like FT_PCT when 0 FTA)
    nan_count = player_logs.isna().sum().sum()
    if nan_count > 0:
        print(f"  Filling {nan_count} NaN values in raw data...")
        numeric_cols = player_logs.select_dtypes(include=[np.number]).columns
        player_logs[numeric_cols] = player_logs[numeric_cols].fillna(0)
    
    # 3. Ensure advanced metrics exist
    player_logs = compute_advanced_metrics(player_logs)
    
    # 3. Aggregate to team-game level (now also returns current Elo state)
    team_games, current_elo = build_team_game_features(player_logs)
    
    # 4. Build sequences
    X, y = build_sequences(team_games)
    
    print(f"\n  X shape: {X.shape}  (samples, timesteps, features)")
    print(f"  y shape: {y.shape}")
    print(f"  Win rate: {y.mean():.3f}")
    
    # 5. Chronological train/val split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"\n  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    
    # 6. Fit scaler on training data only
    # Reshape to 2D for scaling, then back to 3D
    n_train, timesteps, n_features = X_train.shape
    n_val = X_val.shape[0]
    
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_features)
    scaler.fit(X_train_flat)
    
    X_train_scaled = scaler.transform(X_train_flat).reshape(n_train, timesteps, n_features)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(n_val, timesteps, n_features)
    
    # Sanitize: replace any remaining NaN/inf with 0
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    nan_check = np.isnan(X_train_scaled).sum() + np.isnan(X_val_scaled).sum()
    print(f"\n  NaN check after scaling: {nan_check} (should be 0)")
    
    # 7. Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    np.save('X_train.npy', X_train_scaled)
    np.save('X_val.npy', X_val_scaled)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    print(f"\n  [SAVED] X_train.npy: {X_train_scaled.shape}")
    print(f"  [SAVED] X_val.npy:   {X_val_scaled.shape}")
    print(f"  [SAVED] y_train.npy: {y_train.shape}")
    print(f"  [SAVED] y_val.npy:   {y_val.shape}")
    print(f"  [SAVED] {scaler_path}")
    print(f"\n  Done. Upload .npy files and scaler to Colab for training.")


if __name__ == "__main__":
    main()
