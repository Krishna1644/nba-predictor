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

# Helper: Get Team Name
def get_team_name(team_id):
    try:
        return teams.find_team_name_by_id(team_id)['full_name']
    except:
        return f"Team {team_id}"

# 3. TABS
tab1, tab2, tab3 = st.tabs(["Game Matchups", "Brain Monitor", "Full Projections"])

# --- TAB 1: MATCHUPS & WINNERS ---
with tab1:
    if not games.empty and 'GAME_DATE_EST' in games.columns:
        clean_date = games['GAME_DATE_EST'].iloc[0].split("T")[0]
        st.header(f"Schedule for: {clean_date}")
    else:
        st.header("Tonight's Games")
    
    if games.empty:
        st.warning("No games found.")
    
    # Loop through games
    for index, row in games.iterrows():
        home_id = row['HOME_TEAM_ID']
        away_id = row['VISITOR_TEAM_ID']
        
        home_name = get_team_name(home_id)
        away_name = get_team_name(away_id)
        
        # Get Players for each team (Predictions)
        home_preds = df[df['TeamID'] == home_id].sort_values(by='Predicted_PTS', ascending=False)
        away_preds = df[df['TeamID'] == away_id].sort_values(by='Predicted_PTS', ascending=False)
        
        # INJURY LOGIC
        # We need to find players in the roster for this team that match names in injuries.csv
        home_roster_ids = rosters[rosters['TeamID'] == home_id]['PLAYER'].tolist()
        away_roster_ids = rosters[rosters['TeamID'] == away_id]['PLAYER'].tolist()
        
        # Filter injuries for this team
        # Normalize names for comparison (strip/lower)
        active_injuries = injuries.copy()
        if not active_injuries.empty:
            active_injuries['Player_Norm'] = active_injuries['Player'].str.lower().str.strip()
            
            home_injuries = active_injuries[active_injuries['Player_Norm'].isin([x.lower().strip() for x in home_roster_ids])]
            away_injuries = active_injuries[active_injuries['Player_Norm'].isin([x.lower().strip() for x in away_roster_ids])]
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
                
                # Winner Banner
                st.info(f"🏆 **Prediction**: {winner} by {spread:.1f} pts")

            col1, col2 = st.columns(2)
            
            # --- AWAY TEAM ---
            with col1:
                st.subheader(f"Visiting: {away_name}")
                st.metric("Predicted Score", f"{int(away_score)}")
                
                st.caption("Active Roster (Top Predictions)")
                if not away_preds.empty:
                    st.dataframe(
                        away_preds[['Player', 'Predicted_PTS', 'L3', 'L10', 'Role']],
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.write("No predictions available.")

                st.caption("🏥 Injury Report")
                if not away_injuries.empty:
                    # Rename columns for display if needed or just show relevant ones
                    # Expected format in injuries.csv: Player, Injury, Status
                    # Make sure to handle missing columns gratefully
                    display_cols = ['Player', 'Injury', 'Status'] 
                    valid_cols = [c for c in display_cols if c in away_injuries.columns]
                    st.dataframe(away_injuries[valid_cols], hide_index=True, use_container_width=True)
                else:
                    st.success("No critical injuries reported.")

            # --- HOME TEAM ---
            with col2:
                st.subheader(f"Home: {home_name}")
                st.metric("Predicted Score", f"{int(home_score)}")
                
                st.caption("Active Roster (Top Predictions)")
                if not home_preds.empty:
                    st.dataframe(
                        home_preds[['Player', 'Predicted_PTS', 'L3', 'L10', 'Role']],
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
            team_data = df[(df['TeamID'] == tid) & (df['Role'] == 'STARTER')]
            if not team_data.empty:
                w_form = team_data.iloc[0]['Weight_Form']
                st.write(f"**{team_name}** Trust Level:")
                st.progress(w_form, text=f"Form: {int(w_form*100)}% | Class: {int((1-w_form)*100)}%")

# --- TAB 3: FULL PROJECTIONS ---
with tab3:
    st.subheader("All Predictions")
    st.dataframe(
        df[['Player', 'Predicted_PTS', 'L3', 'L10', 'Role', 'Weight_Form']],
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