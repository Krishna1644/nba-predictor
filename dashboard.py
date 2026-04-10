import streamlit as st
import pandas as pd
import os
import sqlite3
import datetime
import json
import unicodedata
from nba_api.stats.static import teams

try:
    from google import genai
except ImportError:
    genai = None

# 1. SETUP
st.set_page_config(page_title="NBA LSTM Predictor", layout="wide", page_icon="🏀")
st.title("🏀 NBA AI Prediction Engine")

# 2. LOAD DATA
@st.cache_data(ttl=600)  # Clear cache every 10 mins
def load_data():
    try:
        preds = pd.read_csv('final_predictions.csv')
        games = pd.read_csv('todays_games.csv')
        try: rosters = pd.read_csv('todays_rosters.csv')
        except: rosters = pd.DataFrame()
        try: injuries = pd.read_csv('injuries.csv')
        except: injuries = pd.DataFrame()
        try: schedule_ctx = pd.read_csv('schedule_context.csv')
        except: schedule_ctx = pd.DataFrame()
        try:
            with open('models/elo_ratings.json') as f:
                elo = json.load(f)
        except:
            elo = {}
            
        return preds, games, rosters, injuries, schedule_ctx, elo
    except Exception as e:
        return None, None, None, None, None, None

@st.cache_data(ttl=3600)
def load_season_stats():
    if not os.path.exists('nba_stats.db'): return pd.DataFrame()
    conn = sqlite3.connect('nba_stats.db')
    query = """
    WITH team_games AS (
        SELECT 
            TEAM_ABBR,
            Game_ID,
            MAX(CASE WHEN WL = 'W' THEN 1 ELSE 0 END) as Win,
            SUM(PTS) as PTS,
            SUM(REB) as REB,
            SUM(AST) as AST,
            SUM(FGM) as FGM,
            SUM(FGA) as FGA,
            SUM(FG3M) as FG3M,
            SUM(FG3A) as FG3A,
            SUM(FTM) as FTM,
            SUM(FTA) as FTA,
            SUM(TOV) as TOV,
            SUM(PLUS_MINUS)/5.0 as PLUS_MINUS
        FROM player_logs
        WHERE SEASON_ID = '22025'
        GROUP BY TEAM_ABBR, Game_ID
    )
    SELECT 
        TEAM_ABBR,
        COUNT(Game_ID) as Games,
        SUM(Win) as Wins,
        SUM(PTS) as Total_PTS,
        SUM(REB) as Total_REB,
        SUM(AST) as Total_AST,
        SUM(FGM) as Total_FGM,
        SUM(FGA) as Total_FGA,
        SUM(FG3M) as Total_FG3M,
        SUM(FG3A) as Total_FG3A,
        SUM(FTM) as Total_FTM,
        SUM(FTA) as Total_FTA,
        SUM(TOV) as Total_TOV,
        SUM(PLUS_MINUS) as Total_Diff
    FROM team_games
    GROUP BY TEAM_ABBR
    """
    try:
        df = pd.read_sql(query, conn)
    except:
        df = pd.DataFrame()
    finally:
        conn.close()
        
    if df.empty: return df
    
    # Derive Advanced Metrics
    df['Win%'] = (df['Wins'] / df['Games'])
    df['PPG'] = df['Total_PTS'] / df['Games']
    df['RPG'] = df['Total_REB'] / df['Games']
    df['APG'] = df['Total_AST'] / df['Games']
    df['DIFF'] = df['Total_Diff'] / df['Games']
    
    # Efficiency 
    df['eFG%'] = (df['Total_FGM'] + 0.5 * df['Total_FG3M']) / df['Total_FGA']
    df['TS%'] = df['Total_PTS'] / (2 * (df['Total_FGA'] + 0.44 * df['Total_FTA']))
    df['TOV%'] = df['Total_TOV'] / (df['Total_FGA'] + 0.44 * df['Total_FTA'] + df['Total_TOV'])
    
    return df

preds_df, games, rosters, injuries, schedule_ctx, elo_dict = load_data()
season_stats_df = load_season_stats()

if preds_df is None:
    st.error("Data files not found. Run the pipeline first!")
    st.stop()

def get_team_name(team_id):
    try: return teams.find_team_name_by_id(team_id)['full_name']
    except: return f"Team {team_id}"

# Normalization helper for names
def norm_name(name):
    if not isinstance(name, str): return ""
    return ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn').lower().strip()

# 3. LLM SETUP
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# 4. TAB STRUCTURE
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏀 Matchups", "📈 Season Leaderboard", "💬 AI Assistant", "🧠 Model Architecture", "📊 Projections", "✅ Results Validation"
])

# --- TAB 1: MATCHUPS & WINNERS ---
with tab1:
    if not games.empty and 'GAME_DATE_EST' in games.columns:
        clean_date = games['GAME_DATE_EST'].iloc[0].split("T")[0]
        st.header(f"Schedule for: {clean_date}")
    else:
        st.header("Tonight's Games")
    
    if games.empty: st.warning("No games found.")
    
    game_options = ["All Games"]
    game_map = {}
    if not games.empty:
        for idx, row in games.iterrows():
            game_options.append(f"{get_team_name(row['VISITOR_TEAM_ID'])} @ {get_team_name(row['HOME_TEAM_ID'])}")
            game_map[game_options[-1]] = row['GAME_ID']

    selected_game = st.selectbox("🏀 Select Matchup:", game_options)
    display_preds = preds_df[preds_df['GAME_ID'] == game_map[selected_game]] if selected_game != "All Games" else preds_df

    if display_preds.empty: st.warning("No predictions available for this selection.")
    
    for _, pred_row in display_preds.iterrows():
        home_name = pred_row['Home_Team']
        away_name = pred_row['Away_Team']
        st.divider()
        
        col_head1, col_head2, col_head3 = st.columns([1, 2, 1])
        with col_head2:
            st.markdown(f"<h3 style='text-align: center'>{away_name} @ {home_name}</h3>", unsafe_allow_html=True)
            st.info(f"🏆 **{pred_row['Predicted_Winner']}** wins — {pred_row['Confidence']:.1%} confidence")

        col1, col2 = st.columns(2)
        away_won = pred_row['Away_Win_Prob'] > pred_row['Home_Win_Prob']

        def render_team_column(col, name, prob, is_won, is_home):
            with col:
                color = "#4CAF50" if is_won else "#FF5252"
                icon = "🏆 " if is_won else ""
                st.markdown(f"<h4 style='color: {color}'>{icon}{'Home' if is_home else 'Away'}: {name}</h4>", unsafe_allow_html=True)
                st.metric("Win Probability", f"{prob:.1%}")
                st.progress(float(prob))
                
                # Fetch Team ID
                try: 
                    team_id = games[games['GAME_ID'] == pred_row['GAME_ID']].iloc[0]['HOME_TEAM_ID' if is_home else 'VISITOR_TEAM_ID']
                except: return
                
                # Context
                if not schedule_ctx.empty:
                    ctx = schedule_ctx[schedule_ctx['TEAM_ID'] == team_id]
                    if not ctx.empty:
                        st.caption(f"🗓️ Rest: {int(ctx.iloc[0]['rest_days'])} days | B2B: {'Yes' if ctx.iloc[0]['is_back_to_back'] else 'No'} | Games last 7d: {int(ctx.iloc[0]['games_last_7'])}")
                
                # Active Roster List
                if not rosters.empty:
                    team_roster = rosters[rosters['TeamID'] == team_id]
                    if not team_roster.empty:
                        with st.expander(f"👕 Active Players ({len(team_roster)})"):
                            st.dataframe(team_roster[['PLAYER', 'POSITION']], hide_index=True, use_container_width=True)

                # Injuries
                if not injuries.empty:
                    team_injuries = []
                    if not rosters.empty:
                        roster_names = [norm_name(x) for x in team_roster['PLAYER'].tolist()]
                        for _, inj in injuries.iterrows():
                            if norm_name(inj['Player']) in roster_names:
                                team_injuries.append(inj)
                    
                    if team_injuries:
                        st.caption("🏥 Injury Report")
                        st.dataframe(pd.DataFrame(team_injuries)[['Player', 'Injury', 'Status']], hide_index=True)
                    else:
                        st.caption("🏥 No critical injuries reported on active.")

        render_team_column(col1, away_name, pred_row['Away_Win_Prob'], away_won, False)
        render_team_column(col2, home_name, pred_row['Home_Win_Prob'], not away_won, True)

# --- TAB 2: SEASON LEADERBOARD ---
with tab2:
    st.header("📈 2025-26 Season Leaderboard")
    
    with st.expander("📚 Metric Glossary / Legend"):
        st.markdown("""
        * **Elo**: Historical zero-sum power rating algorithm. 1500 is average.
        * **Win%**: Win Percentage for the 2025-26 Season.
        * **PPG / RPG / APG**: Points, Rebounds, and Assists per game.
        * **DIFF**: Average Point Differential per game (Total Points Scored - Total Points Allowed).
        * **eFG%** (Effective Field Goal %): Adjusts standard FG% to account for the fact that 3-point shots are worth 50% more than 2-point shots.
        * **TS%** (True Shooting %): A holistic shooting efficiency metric calculating two-pointers, three-pointers, and free throws combined.
        * **TOV%** (Turnover %): An estimate of turnovers committed per 100 plays. Lower is better. (Color-coded inverse red scale).
        """)
        
    if not season_stats_df.empty:
        # Add Elo if available
        if elo_dict:
            season_stats_df['Elo'] = season_stats_df['TEAM_ABBR'].map(elo_dict).fillna(1500.0)
            
        display_stats = season_stats_df[['TEAM_ABBR', 'Games', 'Wins', 'Win%', 'PPG', 'RPG', 'APG', 'DIFF', 'eFG%', 'TS%', 'TOV%']].copy()
        if 'Elo' in season_stats_df.columns:
            display_stats.insert(1, 'Elo', season_stats_df['Elo'])
            
        st.dataframe(
            display_stats.style.format({
                'Elo': '{:.0f}', 'Win%': '{:.1%}', 'PPG': '{:.1f}', 'RPG': '{:.1f}', 
                'APG': '{:.1f}', 'DIFF': '{:+.1f}', 'eFG%': '{:.1%}', 'TS%': '{:.1%}', 'TOV%': '{:.1%}'
            }).background_gradient(subset=['Win%', 'DIFF', 'eFG%', 'TS%'], cmap='Greens')
              .background_gradient(subset=['TOV%'], cmap='Reds', gmap=-display_stats['TOV%']),
            hide_index=True, use_container_width=True, height=800
        )
    else:
        st.warning("Season stats not generated. Ensure nba_stats.db contains data for exactly SEASON_ID '22025'.")

# --- TAB 3: NBA AI ASSISTANT ---
with tab3:
    st.header("💬 NBA AI Database Assistant")
    
    if not genai or not os.getenv("GEMINI_API_KEY"):
        st.warning("⚠️ **API Config Required:** To use the AI Assistant, please install the library (`pip install google-genai`) and provide your free Gemini API key in the sidebar.")
    else:
        st.info("Ask me anything about tonight's projections, the data structures, injuries, or team statistics!")
        
        # Initialize internal chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        with st.container(height=400):
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if prompt := st.chat_input("Ex: 'Who has the highest projected win probability tonight and why?'"):
            # UI side display
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Build Background Context Payload
            context_payload = "BACKGROUND KNOWLEDGE PROVIDED FROM THE APP:\n"
            context_payload += "--- TONIGHT'S PIPELINE PREDICTIONS ---\n"
            context_payload += preds_df.to_string(index=False) + "\n\n"
            context_payload += "--- KEY INJURIES (TONIGHT) ---\n"
            if not injuries.empty and not rosters.empty:
                roster_team_map = {}
                for _, r in rosters.iterrows():
                    roster_team_map[norm_name(r['PLAYER'])] = get_team_name(r['TeamID'])
                
                mapped_injuries = []
                for _, inj in injuries.iterrows():
                    p_name = norm_name(inj['Player'])
                    t_name = roster_team_map.get(p_name, "Unknown/Free Agent")
                    mapped_injuries.append({'Player': inj['Player'], 'Team': t_name, 'Injury': inj['Injury'], 'Status': inj['Status']})
                context_payload += pd.DataFrame(mapped_injuries).to_string(index=False) + "\n\n"
            else:
                context_payload += injuries.to_string(index=False) + "\n\n"
            context_payload += "--- CURRENT SEASON TEAM STATS (2025-26) ---\n"
            if not season_stats_df.empty:
                context_payload += season_stats_df[['TEAM_ABBR', 'Win%', 'PPG', 'DIFF', 'eFG%', 'TS%', 'TOV%']].to_string(index=False) + "\n"

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                try:
                    client = genai.Client()
                    
                    models_to_try = [
                        'gemini-2.5-flash',
                        'gemini-2.0-flash-lite',
                        'gemini-flash-latest',
                        'gemini-pro-latest'
                    ]
                    
                    response = None
                    last_error = None
                    
                    for model_name in models_to_try:
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=prompt,
                                config=genai.types.GenerateContentConfig(
                                    system_instruction="You are a brilliant NBA data analyst answering user questions. Use the provided contextual data to directly inform your answers. If the data is provided, use it. Output in clean markdown. \n" + context_payload,
                                    temperature=0.4
                                )
                            )
                            break # Success!
                        except Exception as e:
                            last_error = str(e)
                            continue
                            
                    if response:
                        full_response = response.text
                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    else:
                        message_placeholder.error(f"Error connecting to Gemini. Exhausted all fallback models. Last error: {last_error}")
                except Exception as e:
                    message_placeholder.error(f"Failed to initialize Gemini Client: {str(e)}")


# --- TAB 4: MODEL ARCHITECTURE ---
with tab4:
    st.header("🧠 BiLSTM + Attention Architecture")
    st.markdown("""
    ### Deep Learning Topology
    ```
    Input:   (10 games × 24 features)
        ↓
    BiLSTM:  128 units (64 per direction), return_sequences=True
    Dropout: 0.3
        ↓
    BiLSTM:  64 units (32 per direction), return_sequences=True
    Dropout: 0.3
        ↓
    Attention: Learns which games in the rolling window matter most
        ↓
    Dense:   16 units, ReLU
    Dropout: 0.3
        ↓
    Output:  1 unit, Sigmoid → Win Probability
    ```
    """)
    st.markdown("### Feature Matrix Definition")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Performance (9)**\n- PTS, REB, AST, STL, BLK, TOV\n- Max PTS, REB, AST (Absence Proxy)")
    c2.markdown("**Efficiency/Strength (11)**\n- FG%, 3P%, FT%, eFG%, TS%, TOV%\n- Plus/Minus, Home Flag\n- Team Elo, Opp Win%, Opp Def Rtg")
    c3.markdown("**Situational (4)**\n- Rest Days, Back-to-Back, Games in 7d\n- Missing Starter Minutes")
    
# --- TAB 5: PROJECTIONS ---
with tab5:
    st.header("📊 Full Projections Table")
    if not preds_df.empty:
        d = preds_df.copy()
        d['Home_Win_Prob'] = d['Home_Win_Prob'].apply(lambda x: f"{x:.1%}")
        d['Away_Win_Prob'] = d['Away_Win_Prob'].apply(lambda x: f"{x:.1%}")
        d['Confidence'] = d['Confidence'].apply(lambda x: f"{x:.1%}")
        st.dataframe(d[['Home_Team', 'Away_Team', 'Home_Win_Prob', 'Away_Win_Prob', 'Predicted_Winner', 'Confidence']], hide_index=True, use_container_width=True)

# --- TAB 6: RESULTS VALIDATION ---
with tab6:
    st.header("✅ Prediction vs Reality Engine")
    selected_date = st.date_input("Select Historical Date", datetime.date.today())
    history_file = f"history/preds_{selected_date.strftime('%Y-%m-%d')}.csv"
    
    if not os.path.exists(history_file): st.warning("No predictions found for this date.")
    else:
        hist_df = pd.read_csv(history_file)
        if 'Predicted_Winner' in hist_df.columns:
            try:
                conn = sqlite3.connect('nba_stats.db')
                actuals = pd.read_sql(f"SELECT DISTINCT Game_ID, WL, MATCHUP FROM player_logs WHERE GAME_DATE = '{selected_date.strftime('%b %d, %Y')}'", conn)
                conn.close()
                
                if actuals.empty: st.info("No game results in dataset for this date yet.")
                else:
                    results, correct, total = [], 0, 0
                    for _, pr in hist_df.iterrows():
                        hr = actuals[(actuals['Game_ID'] == str(pr.get('GAME_ID', ''))) & (actuals['MATCHUP'].str.contains('vs.', na=False))]
                        if not hr.empty:
                            actual_winner = pr['Home_Team'] if hr.iloc[0]['WL'] == 'W' else pr['Away_Team']
                            if pr['Predicted_Winner'] == actual_winner: correct += 1
                            total += 1
                            results.append({'Matchup': f"{pr['Away_Team']} @ {pr['Home_Team']}", 'Predicted': pr['Predicted_Winner'], 'Actual': actual_winner, 'Conf': f"{pr['Confidence']:.1%}", 'Result': '✅' if pr['Predicted_Winner'] == actual_winner else '❌'})
                    
                    if results:
                        c1, c2 = st.columns(2)
                        c1.metric("Games Evaluated", total)
                        c2.metric("Neural Net Accuracy", f"{(correct/total):.1%}")
                        st.dataframe(pd.DataFrame(results), hide_index=True, use_container_width=True)
            except Exception as e: st.error(f"Error checking DB results: {e}")
        else: st.info("Legacy prediction format detected. Validation off.")
