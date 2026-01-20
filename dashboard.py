import streamlit as st
import pandas as pd
from nba_api.stats.static import teams

# 1. SETUP
st.set_page_config(page_title="NBA AI Predictor", layout="wide")
st.title("NBA Performance Prediction Engine")

# 2. LOAD DATA
@st.cache_data
def load_data():
    try:
        preds = pd.read_csv('final_predictions.csv', encoding='utf-8-sig')
        rosters = pd.read_csv('todays_rosters.csv', encoding='utf-8-sig')
        games = pd.read_csv('todays_games.csv', encoding='utf-8-sig')
        
        # Merge Predictions with Roster Info
        merged_df = pd.merge(preds, rosters, left_on='Player', right_on='PLAYER', how='left')
        return merged_df, games
    except FileNotFoundError:
        return None, None

df, games = load_data()

if df is None:
    st.error("CSV files not found. Run the pipeline first!")
    st.stop()

# Helper: Get Team Name
def get_team_name(team_id):
    try:
        return teams.find_team_name_by_id(team_id)['full_name']
    except:
        return f"Team {team_id}"

# 3. TABS
tab1, tab2, tab3 = st.tabs(["Game Matchups & Winners", "Top Scorers", "Value Plays"])

# --- TAB 1: MATCHUPS & WINNERS ---
with tab1:
    # Header Date Logic
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
        
        # Get Players for each team
        home_players = df[df['TeamID'] == home_id].sort_values(by='Predicted_PTS', ascending=False)
        away_players = df[df['TeamID'] == away_id].sort_values(by='Predicted_PTS', ascending=False)
        
        # --- ALGORITHM: SUM OF TOP 8 (ROTATION) ---
        # We only sum the top 8 scorers to simulate a real game rotation
        home_score = home_players.head(8)['Predicted_PTS'].sum()
        away_score = away_players.head(8)['Predicted_PTS'].sum()
        
        # Determine Winner
        if home_score > away_score:
            winner = home_name
            spread = home_score - away_score
            color = "green"
        else:
            winner = away_name
            spread = away_score - home_score
            color = "red" # Just a distinct color for away

        # DRAW THE UI
        st.divider()
        st.subheader(f"{away_name} @ {home_name}")
        
        # Winner Banner
        st.markdown(f"""
        <div style="padding: 10px; border-radius: 5px; background-color: rgba(50, 200, 50, 0.2); border: 1px solid green; margin-bottom: 10px;">
            <h3 style="margin:0; color: #0f5132;"> AI Prediction: {winner} to win</h3>
            <p style="margin:0;">Predicted Score: <b>{away_name} {int(away_score)}</b> - <b>{home_name} {int(home_score)}</b> (Diff: {spread:.1f})</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        fmt = {"Predicted_PTS": "{:.1f}", "L3_Avg_PTS": "{:.1f}"}
        
        with col1:
            st.write(f"**{away_name} Roster**")
            if not away_players.empty:
                st.dataframe(away_players[['Player', 'Predicted_PTS']].style.format(fmt), hide_index=True)
            else:
                st.info("No data")
                
        with col2:
            st.write(f"**{home_name} Roster**")
            if not home_players.empty:
                st.dataframe(home_players[['Player', 'Predicted_PTS']].style.format(fmt), hide_index=True)
            else:
                st.info("No data")

# --- TAB 2: TOP SCORERS ---
with tab2:
    st.subheader("League-Wide Projections")
    num_players = st.slider("Show Top N", 5, 50, 10)
    top_scorers = df.sort_values(by='Predicted_PTS', ascending=False).head(num_players)
    st.bar_chart(top_scorers.set_index('Player')['Predicted_PTS'])
    st.dataframe(top_scorers[['Player', 'Predicted_PTS', 'L3_Avg_PTS']].style.format(fmt), hide_index=True)

# --- TAB 3: VALUE PLAYS ---
with tab3:
    st.subheader("Model vs. Recent Form")
    df['Diff'] = df['Predicted_PTS'] - df['L3_Avg_PTS']
    buy = df.sort_values(by='Diff', ascending=False).head(5)
    sell = df.sort_values(by='Diff', ascending=True).head(5)
    
    c1, c2 = st.columns(2)
    fmt_diff = {"Diff": "{:.1f}", "Predicted_PTS": "{:.1f}"}
    
    with c1:
        st.success("Buy Low (Expect Bounceback)")
        st.dataframe(buy[['Player', 'Diff', 'Predicted_PTS']].style.format(fmt_diff), hide_index=True)
    with c2:
        st.error("Sell High (Expect Regression)")
        st.dataframe(sell[['Player', 'Diff', 'Predicted_PTS']].style.format(fmt_diff), hide_index=True)