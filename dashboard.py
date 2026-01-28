import streamlit as st
import pandas as pd
from nba_api.stats.static import teams

# 1. SETUP
st.set_page_config(page_title="NBA AI Predictor", layout="wide")
st.title("🏀 NBA Self-Learning Engine")

# 2. LOAD DATA
@st.cache_data
def load_data():
    try:
        preds = pd.read_csv('final_predictions.csv')
        games = pd.read_csv('todays_games.csv')
        rosters = pd.read_csv('todays_rosters.csv')
        
        # Load Injuries
        try:
            injuries = pd.read_csv('injuries.csv')
        except:
            injuries = pd.DataFrame(columns=['Player', 'Injury', 'Status'])
            
        return preds, games, rosters, injuries
    except Exception as e:
        return None, None, None, None

df, games, rosters, injuries = load_data()

if df is None:
    st.error("Data files not found. Run the pipeline first!")
    st.stop()

# --- HOT STREAK LOGIC ---
if not df.empty:
    # If L3 is 25% higher than L10, they are "HOT"
    df['Is_Hot'] = df['L3'] > (df['L10'] * 1.25)
    df['Display_Name'] = df.apply(lambda x: f"{x['Player']} 🔥" if x['Is_Hot'] else x['Player'], axis=1)

# Helper: Get Team Name
def get_team_name(team_id):
    try:
        return teams.find_team_name_by_id(team_id)['full_name']
    except:
        return f"Team {team_id}"

# 3. TABS
tab1, tab2, tab3, tab4 = st.tabs(["Game Matchups", "Brain Monitor", "Full Projections", "Results Validation"])

# --- TAB 1: MATCHUPS & WINNERS ---
with tab1:
    if not games.empty and 'GAME_DATE_EST' in games.columns:
        clean_date = games['GAME_DATE_EST'].iloc[0].split("T")[0]
        st.header(f"Schedule for: {clean_date}")
    else:
        st.header("Tonight's Games")
    
    if games.empty:
        st.warning("No games found.")
    
    # Prepare Dropdown Options
    game_options = ["All Games"]
    game_map = {} # Map "Away @ Home" -> GameID
    
    if not games.empty:
        for idx, row in games.iterrows():
            h_name = get_team_name(row['HOME_TEAM_ID'])
            a_name = get_team_name(row['VISITOR_TEAM_ID'])
            label = f"{a_name} @ {h_name}"
            game_options.append(label)
            game_map[label] = row['GAME_ID']

    # Dropdown
    selected_game = st.selectbox("🏀 Select Matchup:", game_options)

    # Filter Games
    if selected_game != "All Games":
        target_id = game_map[selected_game]
        games = games[games['GAME_ID'] == target_id]

    if games.empty:
        st.warning("No games found.")
    
    # Loop through games
    for index, row in games.iterrows():
        home_id = row['HOME_TEAM_ID']
        away_id = row['VISITOR_TEAM_ID']
        game_time = row.get('GAME_STATUS_TEXT', 'Time TBD') # Get Start Time
        
        home_name = get_team_name(home_id)
        away_name = get_team_name(away_id)
        
        # Get Players for each team (Predictions)
        home_preds = df[df['TeamID'] == home_id].sort_values(by='Predicted_PTS', ascending=False)
        away_preds = df[df['TeamID'] == away_id].sort_values(by='Predicted_PTS', ascending=False)
        
        # INJURY LOGIC
        home_roster_ids = rosters[rosters['TeamID'] == home_id]['PLAYER'].tolist()
        away_roster_ids = rosters[rosters['TeamID'] == away_id]['PLAYER'].tolist()
        
        active_injuries = injuries.copy()
        if not active_injuries.empty:
            import unicodedata
            def normalize_name(name):
                if not isinstance(name, str): return ""
                return ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn').lower().strip()

            active_injuries['Player_Norm'] = active_injuries['Player'].apply(normalize_name)
            
            # Normalize roster names for comparison
            home_roster_norm = [normalize_name(x) for x in home_roster_ids]
            away_roster_norm = [normalize_name(x) for x in away_roster_ids]
            
            home_injuries = active_injuries[active_injuries['Player_Norm'].isin(home_roster_norm)]
            away_injuries = active_injuries[active_injuries['Player_Norm'].isin(away_roster_norm)]
        else:
            home_injuries = pd.DataFrame()
            away_injuries = pd.DataFrame()

        # --- ALGORITHM: SUM OF TOP 8 (ROTATION) ---
        home_score = home_preds.head(8)['Predicted_PTS'].sum()
        away_score = away_preds.head(8)['Predicted_PTS'].sum()
        
        # Determine Winner
        if home_score > away_score:
            winner = home_name
            spread = home_score - away_score
        else:
            winner = away_name
            spread = away_score - home_score

        # DRAW THE UI (Card Style)
        with st.container():
            st.divider()
            
            # Header
            col_head1, col_head2, col_head3 = st.columns([1, 2, 1])
            with col_head2:
                st.markdown(f"<h3 style='text-align: center'>{away_name} @ {home_name}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: #888;'>⏰ {game_time}</p>", unsafe_allow_html=True)
                
                # Winner Banner
                st.info(f"🏆 **Prediction**: {winner} by {spread:.1f} pts")

            col1, col2 = st.columns(2)
            
            # Determine Colors
            if away_score > home_score:
                away_color = "#4CAF50" # Green
                home_color = "#FF5252" # Red
                away_icon = "🏆 "
                home_icon = ""
            else:
                away_color = "#FF5252" # Red
                home_color = "#4CAF50" # Green
                away_icon = ""
                home_icon = "🏆 "

            # --- AWAY TEAM ---
            with col1:
                st.markdown(f"<h3 style='color: {away_color}'>{away_icon}Visiting: {away_name}</h3>", unsafe_allow_html=True)
                st.metric("Predicted Score", f"{int(away_score)}")
                
                st.caption("Active Roster (Top Predictions)")
                if not away_preds.empty:
                    st.dataframe(
                        away_preds[['Display_Name', 'Predicted_PTS', 'L3', 'L10', 'Role']],
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.write("No predictions available.")

                st.caption("🏥 Injury Report")
                if not away_injuries.empty:
                    display_cols = ['Player', 'Injury', 'Status'] 
                    valid_cols = [c for c in display_cols if c in away_injuries.columns]
                    st.dataframe(away_injuries[valid_cols], hide_index=True, use_container_width=True)
                else:
                    st.success("No critical injuries reported.")

            # --- HOME TEAM ---
            with col2:
                st.markdown(f"<h3 style='color: {home_color}'>{home_icon}Home: {home_name}</h3>", unsafe_allow_html=True)
                st.metric("Predicted Score", f"{int(home_score)}")
                
                st.caption("Active Roster (Top Predictions)")
                if not home_preds.empty:
                    st.dataframe(
                        home_preds[['Display_Name', 'Predicted_PTS', 'L3', 'L10', 'Role']],
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.write("No predictions available.")

                st.caption("🏥 Injury Report")
                if not home_injuries.empty:
                    display_cols = ['Player', 'Injury', 'Status']
                    valid_cols = [c for c in display_cols if c in home_injuries.columns]
                    st.dataframe(home_injuries[valid_cols], hide_index=True, use_container_width=True)
                else:
                    st.success("No critical injuries reported.")

# --- TAB 2: BRAIN MONITOR (Original Logic) ---
with tab2:
    st.header("🧠 Brain Monitor")
    st.info("This shows how much the model trusts 'Recent Form' (L3) vs 'Long Term Class' (L10) for tonight's teams.")

    if not df.empty:
        active_teams = df['TeamID'].unique()
        for tid in active_teams:
            team_name = get_team_name(tid)
            
            # Get avg weight for this team's starters
            # [NOTE]: We don't distinguish Home/Away here for simplicity, just show the weight used for THIS prediction
            team_data = df[(df['TeamID'] == tid) & (df['Role'] == 'STARTER')]
            if not team_data.empty:
                w_form = team_data.iloc[0]['Weight_Form']
                st.write(f"**{team_name}** Trust Level:")
                st.progress(w_form, text=f"Form: {int(w_form*100)}% | Class: {int((1-w_form)*100)}%")

# --- TAB 3: FULL PROJECTIONS ---
with tab3:
    st.subheader("All Predictions")
    st.dataframe(
        df[['Display_Name', 'Predicted_PTS', 'L3', 'L10', 'Role', 'Weight_Form']],
        column_config={
            "Weight_Form": st.column_config.ProgressColumn(
                "Trust in Form",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
        },
        hide_index=True,
        use_container_width=True
    )


# --- TAB 4: RESULTS VALIDATION ---
with tab4:
    st.header("✅ Prediction vs Reality")
    
    # Date Selection
    import datetime
    import os
    import sqlite3
    from nba_api.stats.static import players
    
    # Default to yesterday
    default_date = datetime.date.today()
    selected_date = st.date_input("Select Date", default_date)
    date_str = selected_date.strftime("%Y-%m-%d")
    
    history_file = f"history/preds_{date_str}.csv"
    
    if not os.path.exists(history_file):
        st.warning(f"No predictions found for {date_str}. (File: {history_file})")
    else:
        # Load Predictions
        hist_df = pd.read_csv(history_file)
        
        # Load Actuals from DB
        try:
            conn = sqlite3.connect('nba_stats.db')
            # Convert date format if needed. DB uses 'Jan 14, 2026' or similar? 
            # inspect_db output showed 'Jan 14, 2026'. 
            # We need to convert YYYY-MM-DD to 'MMM DD, YYYY'.
            db_date_str = selected_date.strftime("%b %d, %Y")
            
            query = f"SELECT * FROM player_logs WHERE GAME_DATE = '{db_date_str}'"
            actuals_df = pd.read_sql(query, conn)
            conn.close()
            
            if actuals_df.empty:
                # SPECIAL MESSAGE FOR USER REQUEST
                st.info(f"Since you only have history for {date_str} (Today), you won't see much 'Results' for it yet until the games happen.")
            else:
                # Perform Matching
                st.write(f"Found {len(actuals_df)} player records for {db_date_str}.")
                
                # 1. Map Prediction Names -> IDs
                # Get all players for robust matching
                all_players = players.get_players()
                
                import unicodedata
                def normalize(name):
                    if not isinstance(name, str): return ""
                    return ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn').lower().replace('.','').strip()
                
                # Create Map: Normalized Name -> ID
                name_map = {normalize(p['full_name']): p['id'] for p in all_players}
                
                results = []
                
                for _, row in hist_df.iterrows():
                    p_name = row['Player']
                    p_norm = normalize(p_name)
                    p_id = name_map.get(p_norm)
                    
                    if p_id:
                        # Find in actuals
                        match = actuals_df[actuals_df['Player_ID'] == p_id]
                        if not match.empty:
                            actual_pts = match.iloc[0]['PTS']
                            diff = actual_pts - row['Predicted_PTS']
                            results.append({
                                'Player': p_name,
                                'Predicted': row['Predicted_PTS'],
                                'Actual': actual_pts,
                                'Diff': round(diff, 1),
                                'Abs_Diff': abs(diff)
                            })
                
                if results:
                    res_df = pd.DataFrame(results)
                    mae = res_df['Abs_Diff'].mean()
                    
                    st.metric("Mean Absolute Error (MAE)", f"{mae:.2f} pts")
                    
                    st.dataframe(
                        res_df[['Player', 'Predicted', 'Actual', 'Diff']].sort_values(by='Abs_Diff'),
                        use_container_width=True
                    )
                else:
                    st.warning("Could not match any players between Predictions and Results.")

        except Exception as e:
            st.error(f"Error checking results: {str(e)}")


