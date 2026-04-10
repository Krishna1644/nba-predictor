"""
predict_tonight.py — BiLSTM+Attention Inference for Tonight's Games.

Loads the trained BiLSTM+Attention model and scaler, queries the DB for each
team's last 10 games, aggregates to team-level features (matching build_sequences.py),
cross-references injuries to calculate missing_starter_minutes, and outputs
win probabilities to final_predictions.csv.
"""

import pandas as pd
import sqlite3
import numpy as np
import os
import datetime
import json
import joblib
import unicodedata

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model

# Import the custom Attention layer so Keras can deserialize the model
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lstm import Attention

# CONFIG
DB_NAME = 'nba_stats.db'
MODELS_DIR = 'models'
HISTORY_DIR = 'history'
LOOKBACK = 10
STARTER_MIN_THRESHOLD = 25

# Must match build_sequences.py exactly
FEATURE_COLUMNS = [
    'team_pts', 'team_reb', 'team_ast', 'team_stl', 'team_blk', 'team_tov',
    'max_pts', 'max_reb', 'max_ast',
    'fg_pct', 'fg3_pct', 'ft_pct', 'efg_pct', 'ts_pct', 'tov_pct',
    'plus_minus', 'is_home',
    'rest_days', 'is_back_to_back', 'games_last_7',
    'missing_starter_minutes',
    'team_elo', 'opp_win_pct', 'opp_pts_allowed_avg'
]


def normalize_name(name):
    """Normalize player name for injury matching."""
    if not isinstance(name, str):
        return ""
    return ''.join(
        c for c in unicodedata.normalize('NFD', name) 
        if unicodedata.category(c) != 'Mn'
    ).lower().strip()


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


def get_team_name(team_id):
    """Get team full name from nba_api."""
    try:
        from nba_api.stats.static import teams
        info = teams.find_team_name_by_id(team_id)
        return info['full_name'] if info else f"Team {team_id}"
    except:
        return f"Team {team_id}"


def get_injured_player_names():
    """Load injury report and return a set of normalized player names."""
    try:
        injuries = pd.read_csv('injuries.csv')
        return set(injuries['Player'].apply(normalize_name))
    except:
        print("  [WARNING] injuries.csv not found. Proceeding without injury data.")
        return set()


def get_schedule_context():
    """Load schedule context (rest_days, b2b, games_last_7) per team."""
    try:
        ctx = pd.read_csv('schedule_context.csv')
        return dict(zip(
            ctx['TEAM_ID'],
            ctx[['rest_days', 'is_back_to_back', 'games_last_7']].to_dict('records')
        ))
    except:
        print("  [WARNING] schedule_context.csv not found. Using defaults.")
        return {}


def build_team_sequence(team_id, rosters_df, conn, injured_names, schedule_ctx, is_home,
                        game_context=None, elo_ratings=None):
    """
    Build the 10-game lookback sequence for a single team.
    Returns a (10, 24) feature matrix, or None if insufficient data.
    """
    # Get player IDs for this team
    team_players = rosters_df[rosters_df['TeamID'] == team_id]['PLAYER_ID'].unique()
    
    if len(team_players) == 0:
        return None
    
    # Fetch recent game logs for all players on this team
    placeholders = ','.join(str(p) for p in team_players)
    query = f"""
        SELECT * FROM player_logs 
        WHERE Player_ID IN ({placeholders})
        ORDER BY GAME_DATE DESC
    """
    player_logs = pd.read_sql(query, conn)
    
    if player_logs.empty:
        return None
    
    # Compute advanced metrics
    player_logs = compute_advanced_metrics(player_logs)
    
    # Parse dates
    player_logs['GAME_DATE_DT'] = pd.to_datetime(
        player_logs['GAME_DATE'], format='mixed'
    )
    player_logs['MIN_NUMERIC'] = pd.to_numeric(player_logs['MIN'], errors='coerce').fillna(0)
    player_logs['IS_HOME'] = player_logs['MATCHUP'].apply(
        lambda x: 1 if 'vs.' in str(x) else 0
    )
    
    # Get unique game IDs sorted by date (most recent first)
    game_dates = player_logs.groupby('Game_ID')['GAME_DATE_DT'].first().sort_values(ascending=False)
    
    # We need exactly LOOKBACK games, but may need more for schedule context calculation
    recent_game_ids = game_dates.head(LOOKBACK + 7).index.tolist()
    
    if len(recent_game_ids) < LOOKBACK:
        return None
    
    # Compute season-average minutes per player (for missing_starter_minutes)
    player_avg_min = player_logs.groupby('Player_ID')['MIN_NUMERIC'].mean()
    starters = player_avg_min[player_avg_min >= STARTER_MIN_THRESHOLD]
    
    # Calculate missing_starter_minutes for tonight
    # Cross-reference injuries with starters
    roster_names = rosters_df[rosters_df['TeamID'] == team_id][['PLAYER_ID', 'PLAYER']]
    tonight_missing_min = 0.0
    
    for _, row in roster_names.iterrows():
        pid = row['PLAYER_ID']
        pname = normalize_name(row['PLAYER'])
        if pname in injured_names and pid in starters.index:
            tonight_missing_min += starters[pid]
    
    # Build team-game features for the last LOOKBACK games
    sequence_rows = []
    lookback_game_ids = recent_game_ids[:LOOKBACK]
    
    # Sort chronologically (oldest first) for the sequence
    lookback_game_ids = list(reversed(lookback_game_ids))
    
    # Compute game dates sorted for schedule context
    all_dates_sorted = game_dates.sort_values(ascending=True)
    dates_list = all_dates_sorted.values
    
    for g_idx, game_id in enumerate(lookback_game_ids):
        game_logs = player_logs[player_logs['Game_ID'] == game_id]
        
        if game_logs.empty:
            continue
        
        # Aggregate
        row = {
            'team_pts': game_logs['PTS'].sum(),
            'team_reb': game_logs['REB'].sum(),
            'team_ast': game_logs['AST'].sum(),
            'team_stl': game_logs['STL'].sum(),
            'team_blk': game_logs['BLK'].sum(),
            'team_tov': game_logs['TOV'].sum(),
            'max_pts': game_logs['PTS'].max(),
            'max_reb': game_logs['REB'].max(),
            'max_ast': game_logs['AST'].max(),
            'fg_pct': game_logs['FG_PCT'].mean(),
            'fg3_pct': game_logs['FG3_PCT'].mean(),
            'ft_pct': game_logs['FT_PCT'].mean(),
            'efg_pct': game_logs['EFG_PCT'].mean(),
            'ts_pct': game_logs['TS_PCT'].mean(),
            'tov_pct': game_logs['TOV_PCT'].mean(),
            'plus_minus': game_logs['PLUS_MINUS'].mean(),
            'is_home': game_logs['IS_HOME'].iloc[0],
        }
        
        # Schedule context for this historical game
        game_date = game_logs['GAME_DATE_DT'].iloc[0]
        game_date_idx = np.searchsorted(dates_list, np.datetime64(game_date))
        
        if game_date_idx > 0:
            prev_date = pd.Timestamp(dates_list[game_date_idx - 1])
            rest = (game_date - prev_date).days
        else:
            rest = 3  # Default for first game
        
        row['rest_days'] = rest
        row['is_back_to_back'] = 1 if rest == 1 else 0
        
        # Games in last 7 days for this historical game
        seven_ago = game_date - pd.Timedelta(days=7)
        g7 = sum(1 for d in dates_list[:game_date_idx] if pd.Timestamp(d) >= seven_ago)
        row['games_last_7'] = g7
        
        # Missing starter minutes for historical games
        played_ids = set(game_logs['Player_ID'].unique())
        hist_missing = 0.0
        for pid, avg_min in starters.items():
            if pid not in played_ids:
                hist_missing += avg_min
        row['missing_starter_minutes'] = hist_missing
        
        # Elo and opponent strength context
        # Get team abbreviation for context lookups
        team_abbr = game_logs['MATCHUP'].iloc[0].split(' ')[0].strip()
        if game_context and (game_id, team_abbr) in game_context:
            ctx = game_context[(game_id, team_abbr)]
            row['team_elo'] = ctx['team_elo']
            row['opp_win_pct'] = ctx['opp_win_pct']
            row['opp_pts_allowed_avg'] = ctx.get('opp_pts_allowed_avg', 105.0)
        elif elo_ratings:
            row['team_elo'] = elo_ratings.get(team_abbr, 1500)
            row['opp_win_pct'] = 0.5
            row['opp_pts_allowed_avg'] = 105.0
        else:
            row['team_elo'] = 1500
            row['opp_win_pct'] = 0.5
            row['opp_pts_allowed_avg'] = 105.0
        
        sequence_rows.append(row)
    
    if len(sequence_rows) < LOOKBACK:
        return None
    
    # Override the most recent game's context with tonight's actual values
    # The last entry in the sequence is the most recent historical game
    # For the PREDICTION game (tonight), we inject tonight's context into
    # the sequence as-is — the model was trained on historical sequences
    # where each timestep has its own context.
    
    # Build feature matrix (LOOKBACK x 24)
    feature_matrix = np.array(
        [[row[col] for col in FEATURE_COLUMNS] for row in sequence_rows],
        dtype=np.float32
    )
    
    return feature_matrix


def main():
    print("--- PREDICTING TONIGHT (LSTM) ---")
    
    # 1. Pre-flight checks
    model_path = os.path.join(MODELS_DIR, 'lstm_model.keras')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    
    if not os.path.exists(model_path):
        print(f"[FAIL] Model not found: {model_path}")
        print("  Train the model in Colab first, then place files in models/")
        return
    if not os.path.exists(scaler_path):
        print(f"[FAIL] Scaler not found: {scaler_path}")
        return
    
    # 2. Load model and scaler
    print("  Loading BiLSTM+Attention model...")
    model = load_model(model_path, custom_objects={'Attention': Attention})
    scaler = joblib.load(scaler_path)
    
    # 3. Load game data
    games = pd.read_csv('todays_games.csv')
    rosters = pd.read_csv('todays_rosters.csv')
    
    if games.empty:
        print("[FAIL] No games in todays_games.csv")
        return
    
    # 4. Load injury, schedule, and strength context
    injured_names = get_injured_player_names()
    schedule_ctx = get_schedule_context()
    
    # Load Elo/strength context for lookback features
    game_context = {}
    elo_ratings = {}
    context_path = os.path.join(MODELS_DIR, 'game_context.pkl')
    elo_path = os.path.join(MODELS_DIR, 'elo_ratings.json')
    try:
        game_context = joblib.load(context_path)
        with open(elo_path) as f:
            elo_ratings = json.load(f)
        print(f"  Loaded Elo context ({len(elo_ratings)} teams, {len(game_context)} game-team pairs)")
    except Exception:
        print("  [WARNING] Elo context files not found. Using defaults.")
    
    conn = sqlite3.connect(DB_NAME)
    
    # 5. Generate predictions for each matchup
    predictions = []
    
    for _, game_row in games.iterrows():
        game_id = game_row['GAME_ID']
        home_id = game_row['HOME_TEAM_ID']
        away_id = game_row['VISITOR_TEAM_ID']
        
        home_name = get_team_name(home_id)
        away_name = get_team_name(away_id)
        
        print(f"\n  {away_name} @ {home_name}")
        
        # Build sequences for both teams
        home_seq = build_team_sequence(
            home_id, rosters, conn, injured_names, schedule_ctx, is_home=1,
            game_context=game_context, elo_ratings=elo_ratings
        )
        away_seq = build_team_sequence(
            away_id, rosters, conn, injured_names, schedule_ctx, is_home=0,
            game_context=game_context, elo_ratings=elo_ratings
        )
        
        if home_seq is None or away_seq is None:
            print(f"    [SKIP] Insufficient data for this matchup.")
            continue
        
        # Scale features
        n_features = home_seq.shape[1]
        home_scaled = scaler.transform(
            home_seq.reshape(-1, n_features)
        ).reshape(1, LOOKBACK, n_features)
        
        away_scaled = scaler.transform(
            away_seq.reshape(-1, n_features)
        ).reshape(1, LOOKBACK, n_features)
        
        # Predict (outputs win probability directly via sigmoid)
        home_raw = model.predict(home_scaled, verbose=0)[0][0]
        away_raw = model.predict(away_scaled, verbose=0)[0][0]
        
        # Normalize probabilities to sum to 1 for the matchup
        total = home_raw + away_raw
        if total > 0:
            home_prob = home_raw / total
            away_prob = away_raw / total
        else:
            home_prob = 0.5
            away_prob = 0.5
        
        # Determine winner and confidence
        if home_prob > away_prob:
            predicted_winner = home_name
            confidence = home_prob
        else:
            predicted_winner = away_name
            confidence = away_prob
        
        print(f"    Home ({home_name}): {home_prob:.1%}")
        print(f"    Away ({away_name}): {away_prob:.1%}")
        print(f"    -> {predicted_winner} ({confidence:.1%})")
        
        predictions.append({
            'GAME_ID': game_id,
            'Home_Team': home_name,
            'Away_Team': away_name,
            'Home_Win_Prob': round(home_prob, 4),
            'Away_Win_Prob': round(away_prob, 4),
            'Predicted_Winner': predicted_winner,
            'Confidence': round(confidence, 4),
        })
    
    conn.close()
    
    # 6. Save predictions
    if predictions:
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        pred_df = pd.DataFrame(predictions)
        
        # Save to dashboard
        pred_df.to_csv('final_predictions.csv', index=False)
        
        # Archive to history
        if not os.path.exists(HISTORY_DIR):
            os.makedirs(HISTORY_DIR)
        pred_df.to_csv(f"{HISTORY_DIR}/preds_{today_str}.csv", index=False)
        
        print(f"\n[SUCCESS] {len(predictions)} predictions generated.")
        print(pred_df[['Home_Team', 'Away_Team', 'Predicted_Winner', 'Confidence']].to_string(index=False))
    else:
        print("\n[FAIL] No predictions generated.")


if __name__ == "__main__":
    main()