import pandas as pd
import sqlite3
import joblib
import sys
import unicodedata
import re

# 1. SETUP
print("--- PREDICTION ENGINE STARTING ---")

# --- THE FIX: AGGRESSIVE NAME MATCHING ---
def simplify_name(name):
    """
    Turns 'Jimmy Butler III' -> 'jimmy butler'
    Turns 'Tim Hardaway Jr.' -> 'tim hardaway'
    Turns 'Luka Dončić' -> 'luka doncic'
    """
    # 1. Lowercase and strip
    name = str(name).lower().strip()
    
    # 2. Remove Accents (Dončić -> Doncic)
    nfkd_form = unicodedata.normalize('NFKD', name)
    name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    
    # 3. Remove Suffixes (Jr, Sr, II, III, IV) using Regex
    # We look for these patterns at the END of the string
    suffixes = r'\s+(jr\.?|sr\.?|ii|iii|iv|v)$'
    name = re.sub(suffixes, '', name)
    
    # 4. Remove all punctuation (dots, apostrophes)
    name = name.replace('.', '').replace("'", "")
    
    # 5. Collapse multiple spaces to one
    name = " ".join(name.split())
    
    return name.strip()

# Load Resources
try:
    model = joblib.load('nba_points_model.pkl')
    print("[OK] Model loaded.")
except:
    print("[ERROR] Model not found.")
    sys.exit()

try:
    rosters = pd.read_csv('todays_rosters.csv', encoding='utf-8')
    print(f"[OK] Roster loaded: {len(rosters)} players.")
except:
    print("[ERROR] Roster not found.")
    sys.exit()
    
# --- LOAD INJURIES ---
try:
    injuries = pd.read_csv('injuries.csv')
    print(f"[OK] Injury Report loaded: {len(injuries)} records.")
    
    # Create a SET of simplified names for fast lookup
    # This turns ["Jimmy Butler"] into {"jimmy butler"}
    injured_set = set(injuries['Player'].apply(simplify_name))
    
    # DEBUG: Print Jimmy Butler to prove it works
    if "jimmy butler" in injured_set:
        print("   -> Verified: Jimmy Butler is in the injury block list.")
    
except FileNotFoundError:
    print("[WARNING] 'injuries.csv' not found. Assuming everyone is healthy.")
    injured_set = set()

try:
    conn = sqlite3.connect("nba_stats.db")
    df_stats = pd.read_sql("SELECT * FROM player_logs", conn)
    conn.close()
    print(f"[OK] Database loaded: {len(df_stats)} games.")
except Exception as e:
    print(f"[ERROR] Database Error: {e}")
    sys.exit()

# 2. CLEAN DATA
df_stats.columns = [x.upper() for x in df_stats.columns]
rosters.columns = [x.upper() for x in rosters.columns]

for col in ['PTS', 'REB', 'AST', 'MIN']:
    if col in df_stats.columns:
        df_stats[col] = pd.to_numeric(df_stats[col], errors='coerce')

df_stats['PLAYER_ID'] = pd.to_numeric(df_stats['PLAYER_ID'], errors='coerce').fillna(0).astype(int)
df_stats['GAME_DATE'] = pd.to_datetime(df_stats['GAME_DATE'])

# 3. PREDICT
predictions = []

def get_last_3_avg(player_id):
    my_games = df_stats[df_stats['PLAYER_ID'] == player_id].copy()
    my_games = my_games.sort_values(by='GAME_DATE', ascending=False)
    
    if len(my_games) < 3:
        return None
        
    last_3 = my_games.head(3)
    return [last_3['PTS'].mean(), last_3['REB'].mean()]

print("[...] Generating predictions...")

skipped_count = 0
for index, row in rosters.iterrows():
    pid = row['PLAYER_ID']
    pname = row['PLAYER']
    
    # --- ROBUST INJURY FILTER ---
    # Simplify the roster name ("Jimmy Butler III" -> "jimmy butler")
    pname_simple = simplify_name(pname)
    
    if pname_simple in injured_set:
        skipped_count += 1
        # Optional: Print to confirm
        # print(f"🚫 Removing Injured Player: {pname}")
        continue
    
    features = get_last_3_avg(pid)
    
    if features:
        feat_df = pd.DataFrame([features], columns=['PTS_L3', 'REB_L3'])
        predicted_pts = model.predict(feat_df)[0]
        
        predictions.append({
            'Player': pname,
            'Predicted_PTS': round(predicted_pts, 1),
            'L3_Avg_PTS': round(features[0], 1)
        })

print(f"[INFO] Successfully removed {skipped_count} injured players from predictions.")

# 4. SAVE
if not predictions:
    print("[WARNING] No predictions generated.")
else:
    results_df = pd.DataFrame(predictions)
    results_df = results_df.sort_values(by='Predicted_PTS', ascending=False)
    
    results_df.to_csv('final_predictions.csv', index=False, encoding='utf-8-sig')
    print(f"[OK] Predictions saved. {len(results_df)} players active.")