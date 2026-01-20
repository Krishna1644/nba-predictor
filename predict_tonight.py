import pandas as pd
import sqlite3
import joblib
import warnings
import sys

# 1. SETUP & CONFIG
# ---------------------------------------------------------
# Fix Unicode Crash: Force standard text output for Windows
sys.stdout.reconfigure(encoding='utf-8')

# Fix Warnings: Shut up the "feature names" warning
warnings.filterwarnings("ignore", category=UserWarning)

print("--- STARTING PREDICTION ENGINE ---")

# Load Resources
try:
    model = joblib.load('nba_points_model.pkl')
    print("[OK] Model loaded.")
except FileNotFoundError:
    print("[ERROR] 'nba_points_model.pkl' not found.")
    exit()

try:
    # Use 'utf-8' encoding to handle names like Dončić correctly
    rosters = pd.read_csv('todays_rosters.csv', encoding='utf-8')
    print(f"[OK] Roster loaded: {len(rosters)} players.")
except FileNotFoundError:
    print("[ERROR] 'todays_rosters.csv' not found.")
    exit()
except UnicodeDecodeError:
    # Fallback if utf-8 fails
    rosters = pd.read_csv('todays_rosters.csv', encoding='latin1')
    print(f"[OK] Roster loaded (latin1 fallback): {len(rosters)} players.")

try:
    conn = sqlite3.connect("nba_stats.db")
    df_stats = pd.read_sql("SELECT * FROM player_logs", conn)
    conn.close()
    print(f"[OK] Database loaded: {len(df_stats)} games.")
except Exception as e:
    print(f"[ERROR] Database Error: {e}")
    exit()

# 2. DATA CLEANING
# ---------------------------------------------------------
print("[...] Cleaning data...")

# Standardize Columns
df_stats.columns = [x.upper() for x in df_stats.columns]
rosters.columns = [x.upper() for x in rosters.columns]

# Force Numbers (Crucial for Math)
cols_to_fix = ['PTS', 'REB', 'AST', 'MIN']
for col in cols_to_fix:
    if col in df_stats.columns:
        df_stats[col] = pd.to_numeric(df_stats[col], errors='coerce')

# Force IDs
df_stats['PLAYER_ID'] = pd.to_numeric(df_stats['PLAYER_ID'], errors='coerce').fillna(0).astype(int)
rosters['PLAYER_ID'] = pd.to_numeric(rosters['PLAYER_ID'], errors='coerce').fillna(0).astype(int)
df_stats['GAME_DATE'] = pd.to_datetime(df_stats['GAME_DATE'])

# 3. PREDICTION LOGIC
# ---------------------------------------------------------
print("[...] Generating predictions...")

predictions = []

def get_last_3_avg(player_id):
    # Filter & Sort
    my_games = df_stats[df_stats['PLAYER_ID'] == player_id].copy()
    my_games = my_games.sort_values(by='GAME_DATE', ascending=False)
    
    if len(my_games) < 3:
        return None
        
    last_3 = my_games.head(3)
    return [last_3['PTS'].mean(), last_3['REB'].mean()]

for index, row in rosters.iterrows():
    pid = row['PLAYER_ID']
    pname = row['PLAYER']
    
    features = get_last_3_avg(pid)
    
    if features:
        # FIX: We put features into a DataFrame to stop the Warning
        feat_df = pd.DataFrame([features], columns=['PTS_L3', 'REB_L3'])
        
        predicted_pts = model.predict(feat_df)[0]
        
        predictions.append({
            'Player': pname,
            'Predicted_PTS': round(predicted_pts, 1),
            'L3_Avg_PTS': round(features[0], 1)
        })

# 4. RESULTS
# ---------------------------------------------------------
if not predictions:
    print("\n[WARNING] No predictions generated.")
else:
    results_df = pd.DataFrame(predictions)
    results_df = results_df.sort_values(by='Predicted_PTS', ascending=False)
    
    # Save FULL data to CSV (keeps special characters intact)
    results_df.to_csv('final_predictions.csv', index=False, encoding='utf-8-sig')
    print("\n[OK] Saved full results to 'final_predictions.csv'")

    # PRINT SAFE VERSION (Removes special chars just for the console display)
    # This prevents the "charmap" crash on Windows
    print("\n" + "="*50)
    print(f"   TOP 20 PREDICTED SCORERS (Total: {len(results_df)})")
    print("="*50)
    
    # Create a temporary copy for display that strips accents
    display_df = results_df.head(20).copy()
    
    # Simple trick: Encode to ASCII, ignore errors, decode back.
    # "Dončić" becomes "Doncic" or "Donic" depending on system, but won't crash.
    display_df['Player'] = display_df['Player'].apply(
        lambda x: x.encode('ascii', 'ignore').decode('ascii')
    )
    
    print(display_df[['Player', 'Predicted_PTS', 'L3_Avg_PTS']].to_string(index=False))