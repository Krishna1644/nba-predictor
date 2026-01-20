import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Load Data (Open the Fridge)
print("Loading data from database...")
conn = sqlite3.connect("nba_stats.db")
df = pd.read_sql("SELECT * FROM player_logs", conn)
conn.close()

# --- FIX: FORCE UPPERCASE COLUMNS ---
# The API gives 'Player_ID', but we want 'PLAYER_ID'. 
# This fixes all casing issues instantly.
df.columns = [x.upper() for x in df.columns]

# 2. Preprocessing
print(f"Columns found: {list(df.columns)}") # Debug print
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

# Sort: Oldest to Newest (Crucial for calculating 'Previous Game' stats)
df = df.sort_values(by=['PLAYER_ID', 'GAME_DATE'], ascending=[True, True])

# 3. Feature Engineering (The Logic)
print("Generating features...")

# Calculate 'Points in Last 3 Games'
# shift(1) means "Don't look at tonight's game, look at the previous one"
df['PTS_L3'] = df.groupby('PLAYER_ID')['PTS'].transform(lambda x: x.shift(1).rolling(window=3).mean())

# Calculate 'Rebounds in Last 3 Games'
df['REB_L3'] = df.groupby('PLAYER_ID')['REB'].transform(lambda x: x.shift(1).rolling(window=3).mean())

# Drop the first 3 games for each player (since they don't have a "Last 3" average yet)
df_clean = df.dropna(subset=['PTS_L3', 'REB_L3', 'PTS'])

print(f"Training data size: {len(df_clean)} games")

# 4. Train the Brain
features = ['PTS_L3', 'REB_L3']
target = 'PTS'

X = df_clean[features]
y = df_clean[target]

if len(X) == 0:
    print("Error: No data available for training. (Did the download finish?)")
    exit()

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"\nModel Report:")
print(f"----------------")
print(f"Mean Absolute Error (MAE): {mae:.2f}") 
print(f"(Model is off by approx {mae:.2f} points per game)")

# 6. Save
joblib.dump(model, 'nba_points_model.pkl')
print("\nModel saved to 'nba_points_model.pkl'")