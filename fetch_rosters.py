import pandas as pd
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams
import time

# 1. Load the games we found
try:
    games = pd.read_csv('todays_games.csv')
except FileNotFoundError:
    print("Error: 'todays_games.csv' not found. Run get_schedule.py first.")
    exit()

# Get unique team IDs (both home and visitor)
team_ids = pd.concat([games['HOME_TEAM_ID'], games['VISITOR_TEAM_ID']]).unique()

print(f"Found {len(team_ids)} unique teams playing today.")

all_rosters = []

for team_id in team_ids:
    # Get team name for nice display
    team_info = teams.find_team_name_by_id(team_id)
    team_name = team_info['full_name'] if team_info else str(team_id)
    
    print(f"Fetching roster for {team_name} ({team_id})...")
    
    try:
        # Call API for this team's roster
        roster_endpoint = commonteamroster.CommonTeamRoster(team_id=team_id)
        
        # --- FIX IS HERE: use get_data_frames()[0] ---
        roster_frames = roster_endpoint.get_data_frames()
        
        if roster_frames:
            roster = roster_frames[0]
            # Add TeamID so we know who they play for
            roster['TeamID'] = team_id
            all_rosters.append(roster)
            
    except Exception as e:
        print(f"Failed to fetch {team_name}: {e}")
    
    # Sleep to avoid rate limits
    time.sleep(0.6)

# 3. Combine and Save
if all_rosters:
    final_roster_df = pd.concat(all_rosters)
    
    # Select columns - keys might be uppercase
    cols_to_keep = ['TeamID', 'PLAYER', 'PLAYER_ID', 'POSITION']
    # Filter only if columns exist
    existing_cols = [c for c in cols_to_keep if c in final_roster_df.columns]
    
    if existing_cols:
        final_roster_df = final_roster_df[existing_cols]

    final_roster_df.to_csv('todays_rosters.csv', index=False)
    print(f"\nSuccess! Saved {len(final_roster_df)} players to 'todays_rosters.csv'.")
    print(final_roster_df.head())
else:
    print("No rosters found.")