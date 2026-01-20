from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime
import pandas as pd

# 1. Get today's date in the format the API wants (YYYY-MM-DD)
# Alternatively, you can hardcode a date like '2025-11-15' to test past games
today = datetime.now().strftime('%Y-%m-%d')
print(f"Fetching games for: {today}")

# 2. Call the API
# This pulls the scoreboard data for the specific date
board = scoreboardv2.ScoreboardV2(game_date=today)

# 3. Get the data frames
# The API returns multiple datasets (GameHeader, LineScore, etc.)
# Index 0 is usually the 'GameHeader' which contains the summary
games_df = board.game_header.get_data_frame()

# 4. Clean it up for display
# We only care about GameID, Home Team, and Away Team for now
if not games_df.empty:
    summary = games_df[['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']]
    print(f"\nFound {len(summary)} games:")
    print(summary)
    
    # Save to CSV just to verify we have physical files
    summary.to_csv('todays_games.csv', index=False)
    print("\nSaved to todays_games.csv")
else:
    print("No games scheduled for today.")