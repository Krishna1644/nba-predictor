"""
Streamlit UI for the production logistic matchup model.
Reads precomputed CSVs from data/live/ (fallback: repo root). Does not retrain.
"""

from __future__ import annotations

import datetime
import json
import os
import sqlite3
import unicodedata
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from nba_api.stats.static import teams

try:
    from google import genai
except ImportError:
    genai = None

st.set_page_config(page_title="NBA Win Probability", layout="wide")

REST_CAP = 7
SEASON_LABEL = {
    "22024": "2024-25",
    "22025": "2025-26",
    "22026": "2026-27",
}


def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _norm_game_id(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")
    return s.replace("", "0")


def _norm_name(name) -> str:
    if not isinstance(name, str):
        return ""
    return "".join(
        c
        for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    ).lower().strip()


@st.cache_data(ttl=600)
def load_model_meta():
    path = _first_existing(
        "models/logistic_feature_columns.json",
        "artifacts/feature_selection.json",
    )
    meta = {}
    if path:
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
    metrics_path = "artifacts/baseline_metrics.json"
    report = None
    if os.path.exists(metrics_path):
        rows = json.load(open(metrics_path, encoding="utf-8"))
        for r in rows:
            if "logistic_efficiency" in r.get("model", "") and "test_2025-26" in r.get(
                "model", ""
            ):
                report = r
                break
            if report is None and "test_2025-26" in r.get("model", "") and "logistic" in r.get(
                "model", ""
            ):
                report = r
    return meta, report


@st.cache_data(ttl=600)
def load_data():
    preds_path = _first_existing(
        "data/live/final_predictions.csv", "final_predictions.csv"
    )
    games_path = _first_existing("data/live/todays_games.csv", "todays_games.csv")
    if not preds_path or not games_path:
        return None

    preds = pd.read_csv(preds_path)
    games = pd.read_csv(games_path)
    if "GAME_ID" in preds.columns:
        preds["GAME_ID"] = _norm_game_id(preds["GAME_ID"])
    if "GAME_ID" in games.columns:
        games["GAME_ID"] = _norm_game_id(games["GAME_ID"])

    def _csv(*names):
        p = _first_existing(*names)
        if not p:
            return pd.DataFrame()
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()

    rosters = _csv("data/live/todays_rosters.csv", "todays_rosters.csv")
    injuries = _csv("data/live/injuries.csv", "injuries.csv")
    schedule_ctx = _csv("data/live/schedule_context.csv", "schedule_context.csv")
    try:
        with open("models/elo_ratings.json", encoding="utf-8") as f:
            elo = json.load(f)
    except Exception:
        elo = {}

    return {
        "preds": preds,
        "games": games,
        "rosters": rosters,
        "injuries": injuries,
        "schedule_ctx": schedule_ctx,
        "elo": elo,
        "preds_path": preds_path,
        "games_path": games_path,
        "preds_mtime": datetime.datetime.fromtimestamp(os.path.getmtime(preds_path)),
    }


@st.cache_data(ttl=3600)
def load_season_stats():
    db = _first_existing("nba_stats.db")
    if not db:
        # Unpack slim gz if present (local / Cloud without full DB)
        gz = _first_existing("data/nba_stats_live.db.gz")
        if gz:
            import gzip
            import shutil

            with gzip.open(gz, "rb") as f_in, open("nba_stats.db", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            db = "nba_stats.db"
        else:
            return pd.DataFrame(), None

    conn = sqlite3.connect(db)
    season_ids = (
        pd.read_sql(
            "SELECT DISTINCT SEASON_ID FROM player_logs ORDER BY SEASON_ID DESC", conn
        )["SEASON_ID"]
        .astype(str)
        .tolist()
    )
    season_id = next(
        (s for s in ("22026", "22025", "22024") if s in season_ids),
        season_ids[0] if season_ids else None,
    )
    if not season_id:
        conn.close()
        return pd.DataFrame(), None

    query = """
    WITH team_games AS (
        SELECT
            TEAM_ABBR, Game_ID,
            MAX(CASE WHEN WL = 'W' THEN 1 ELSE 0 END) AS Win,
            SUM(PTS) AS PTS, SUM(REB) AS REB, SUM(AST) AS AST,
            SUM(FGM) AS FGM, SUM(FGA) AS FGA,
            SUM(FG3M) AS FG3M, SUM(FG3A) AS FG3A,
            SUM(FTM) AS FTM, SUM(FTA) AS FTA,
            SUM(TOV) AS TOV, SUM(PLUS_MINUS)/5.0 AS PLUS_MINUS
        FROM player_logs
        WHERE SEASON_ID = ?
        GROUP BY TEAM_ABBR, Game_ID
    )
    SELECT
        TEAM_ABBR, COUNT(Game_ID) AS Games, SUM(Win) AS Wins,
        SUM(PTS) AS Total_PTS, SUM(REB) AS Total_REB, SUM(AST) AS Total_AST,
        SUM(FGM) AS Total_FGM, SUM(FGA) AS Total_FGA,
        SUM(FG3M) AS Total_FG3M, SUM(FG3A) AS Total_FG3A,
        SUM(FTM) AS Total_FTM, SUM(FTA) AS Total_FTA,
        SUM(TOV) AS Total_TOV, SUM(PLUS_MINUS) AS Total_Diff
    FROM team_games
    GROUP BY TEAM_ABBR
    """
    try:
        df = pd.read_sql(query, conn, params=(season_id,))
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return df, season_id

    df["Win%"] = df["Wins"] / df["Games"]
    df["PPG"] = df["Total_PTS"] / df["Games"]
    df["RPG"] = df["Total_REB"] / df["Games"]
    df["APG"] = df["Total_AST"] / df["Games"]
    df["DIFF"] = df["Total_Diff"] / df["Games"]
    df["eFG%"] = (df["Total_FGM"] + 0.5 * df["Total_FG3M"]) / df["Total_FGA"]
    df["TS%"] = df["Total_PTS"] / (2 * (df["Total_FGA"] + 0.44 * df["Total_FTA"]))
    df["TOV%"] = df["Total_TOV"] / (
        df["Total_FGA"] + 0.44 * df["Total_FTA"] + df["Total_TOV"]
    )
    return df, season_id


def get_team_name(team_id):
    try:
        return teams.find_team_name_by_id(int(team_id))["full_name"]
    except Exception:
        return f"Team {team_id}"


def schedule_date_str(games: pd.DataFrame) -> str | None:
    if games.empty or "GAME_DATE_EST" not in games.columns:
        return None
    raw = str(games["GAME_DATE_EST"].iloc[0])
    return raw.split("T")[0]


# --- load ---
meta, report = load_model_meta()
bundle = load_data()
season_stats_df, season_id = load_season_stats()
season_name = SEASON_LABEL.get(str(season_id), str(season_id) if season_id else "—")

absence = meta.get("absence_feature", "efficiency")
calib = meta.get("calibration", "none")
value_key = meta.get("value_key", "impact_eff")

st.title("NBA Win Probability")
st.caption(
    f"Production: logistic · absence={absence} · calib={calib} · "
    f"train ≤2024-25 · report 2025-26 · live 2026-27"
)

with st.sidebar:
    st.header("Data")
    st.markdown(
        "Predictions are **precomputed** (`data/live/`). "
        "This app never retrains. See `data/live/MANUAL.md` to drop in CSVs by hand."
    )
    if bundle:
        st.write(f"Preds file: `{bundle['preds_path']}`")
        st.write(f"Updated: {bundle['preds_mtime'].strftime('%Y-%m-%d %H:%M')}")
    st.divider()
    st.markdown(
        f"**Model**\n\n"
        f"- Absence: `{absence}` (`{value_key}`)\n"
        f"- Calibration: `{calib}`\n"
        f"- Leaderboard season: **{season_name}**"
    )
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

if bundle is None:
    st.error(
        "Missing `data/live/final_predictions.csv` or `todays_games.csv`. "
        "Drop files in `data/live/` (see MANUAL.md) or run `python run_pipeline.py`."
    )
    st.stop()

preds_df = bundle["preds"]
games = bundle["games"]
rosters = bundle["rosters"]
injuries = bundle["injuries"]
schedule_ctx = bundle["schedule_ctx"]
elo_dict = bundle["elo"]

slate = schedule_date_str(games)
today_et = datetime.datetime.now(ZoneInfo("America/New_York")).date().isoformat()
if slate and slate != today_et:
    st.warning(
        f"Showing slate for **{slate}**, not today ({today_et} ET). "
        "Live CSVs look stale or incomplete. Re-run `python run_pipeline.py` "
        "(or `--source espn` on cloud) — details in `data/live/MANUAL.md`."
    )

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Matchups",
        "Season",
        "Assistant",
        "Model",
        "Projections",
        "Validation",
    ]
)

# --- Matchups ---
with tab1:
    st.subheader(f"Slate · {slate or 'unknown date'}")
    if games.empty:
        st.warning("No games in `todays_games.csv`.")

    game_options = ["All games"]
    game_map = {}
    for _, row in games.iterrows():
        label = f"{get_team_name(row['VISITOR_TEAM_ID'])} @ {get_team_name(row['HOME_TEAM_ID'])}"
        game_options.append(label)
        game_map[label] = row["GAME_ID"]

    selected = st.selectbox("Matchup", game_options)
    display = (
        preds_df
        if selected == "All games"
        else preds_df[preds_df["GAME_ID"] == game_map[selected]]
    )
    if display.empty:
        st.warning("No predictions for this selection.")

    for _, pred in display.iterrows():
        home, away = pred["Home_Team"], pred["Away_Team"]
        st.divider()
        st.markdown(f"### {away} @ {home}")
        st.info(
            f"**{pred['Predicted_Winner']}** · "
            f"{float(pred['Confidence']):.0%} confidence · "
            f"P(home)={float(pred['Home_Win_Prob']):.1%}"
        )
        if "Model" in pred.index and pd.notna(pred.get("Model")):
            st.caption(f"Model tag: {pred['Model']}")

        c_away, c_home = st.columns(2)
        away_fav = float(pred["Away_Win_Prob"]) > float(pred["Home_Win_Prob"])

        def team_panel(col, name, prob, favored, is_home):
            with col:
                st.markdown(f"**{'Home' if is_home else 'Away'} · {name}**")
                st.metric("Win prob", f"{float(prob):.1%}")
                st.progress(min(max(float(prob), 0.0), 1.0))
                try:
                    gid = pred["GAME_ID"]
                    g = games[games["GAME_ID"] == gid].iloc[0]
                    tid = int(g["HOME_TEAM_ID" if is_home else "VISITOR_TEAM_ID"])
                except Exception:
                    return

                if not schedule_ctx.empty:
                    ctx = schedule_ctx[schedule_ctx["TEAM_ID"] == tid]
                    if not ctx.empty:
                        rest = int(min(int(ctx.iloc[0]["rest_days"]), REST_CAP))
                        b2b = "yes" if int(ctx.iloc[0]["is_back_to_back"]) else "no"
                        g7 = int(ctx.iloc[0]["games_last_7"])
                        st.caption(f"Rest {rest}d (cap {REST_CAP}) · B2B {b2b} · last-7 {g7}")

                deficit_col = (
                    "Home_Player_Deficit" if is_home else "Away_Player_Deficit"
                )
                if deficit_col in pred.index and pd.notna(pred.get(deficit_col)):
                    st.caption(f"Efficiency player deficit: {float(pred[deficit_col]):.1f}")

                if not rosters.empty:
                    tr = rosters[rosters["TeamID"] == tid]
                    if not tr.empty:
                        with st.expander(f"Roster ({len(tr)})"):
                            cols = [c for c in ("PLAYER", "POSITION") if c in tr.columns]
                            st.dataframe(tr[cols], hide_index=True, use_container_width=True)

                if not injuries.empty and not rosters.empty:
                    names = {_norm_name(x) for x in tr["PLAYER"].tolist()} if not tr.empty else set()
                    rows = [
                        inj
                        for _, inj in injuries.iterrows()
                        if _norm_name(inj.get("Player", "")) in names
                    ]
                    if rows:
                        st.caption("Injuries (display) — model uses Status=`Out` only")
                        st.dataframe(
                            pd.DataFrame(rows)[["Player", "Injury", "Status"]],
                            hide_index=True,
                        )

        team_panel(c_away, away, pred["Away_Win_Prob"], away_fav, False)
        team_panel(c_home, home, pred["Home_Win_Prob"], not away_fav, True)

# --- Season ---
with tab2:
    st.subheader(f"Team table · {season_name}")
    st.caption("From box scores in DB / slim live DB. Elo from `models/elo_ratings.json`.")
    if season_stats_df.empty:
        st.warning(
            "No season aggregates. Unpack `data/nba_stats_live.db.gz` or add `nba_stats.db` "
            "(include 2026-27 when available)."
        )
    else:
        df = season_stats_df.copy()
        if elo_dict:
            df["Elo"] = df["TEAM_ABBR"].map(elo_dict).fillna(1500.0)
        cols = ["TEAM_ABBR", "Games", "Wins", "Win%", "PPG", "RPG", "APG", "DIFF", "eFG%", "TS%", "TOV%"]
        if "Elo" in df.columns:
            cols.insert(1, "Elo")
        show = df[cols].sort_values("Win%", ascending=False)
        st.dataframe(
            show.style.format(
                {
                    "Elo": "{:.0f}",
                    "Win%": "{:.1%}",
                    "PPG": "{:.1f}",
                    "RPG": "{:.1f}",
                    "APG": "{:.1f}",
                    "DIFF": "{:+.1f}",
                    "eFG%": "{:.1%}",
                    "TS%": "{:.1%}",
                    "TOV%": "{:.1%}",
                }
            ),
            hide_index=True,
            use_container_width=True,
            height=720,
        )

# --- Assistant ---
with tab3:
    st.subheader("Ask about tonight's slate")
    if not genai or not os.getenv("GEMINI_API_KEY"):
        st.info("Optional: set `GEMINI_API_KEY` in Streamlit secrets for the assistant.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt := st.chat_input("e.g. Who is favored tonight and why?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            ctx = (
                f"Model: logistic absence={absence} calib={calib}\n"
                f"Predictions:\n{preds_df.to_string(index=False)}\n"
            )
            with st.chat_message("assistant"):
                ph = st.empty()
                try:
                    client = genai.Client()
                    resp = None
                    err = None
                    for model_name in (
                        "gemini-2.5-flash",
                        "gemini-2.5-flash-lite",
                        "gemini-2.0-flash",
                    ):
                        try:
                            resp = client.models.generate_content(
                                model=model_name,
                                contents=prompt,
                                config=genai.types.GenerateContentConfig(
                                    system_instruction=(
                                        "NBA analyst. Use only provided context. Markdown.\n"
                                        + ctx
                                    ),
                                    temperature=0.3,
                                ),
                            )
                            break
                        except Exception as e:
                            err = str(e)
                    if resp:
                        ph.markdown(resp.text)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": resp.text}
                        )
                    else:
                        ph.error(err or "Gemini failed")
                except Exception as e:
                    ph.error(str(e))

# --- Model ---
with tab4:
    st.subheader("Model card")
    if report:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("2025-26 accuracy", f"{report['accuracy']:.3f}")
        c2.metric("Log loss", f"{report['log_loss']:.4f}")
        c3.metric("Brier", f"{report['brier']:.4f}")
        c4.metric("AUC", f"{report['auc']:.3f}")
        st.caption(f"Report row: `{report['model']}` · n={report['n']} · not used for selection")

    st.markdown(
        f"""
**Shipped:** `StandardScaler` → logistic → calib=`{calib}`  
**Absence value:** `{absence}` / `{value_key}`  
**Train:** 2015-16 → 2024-25 (VAL = last 20% of those dates)  
**Report:** 2025-26 (selection-free)  
**Live:** 2026-27  

**Features:** Elo diff, rest / B2B / games-in-7 (rest capped at 7), rolling form diffs,  
efficiency-weighted player deficit diff (`Out` only at serve; DNP proxy in train).

**Not production:** BiLSTM (held for comparison only).  
**Not automatic:** nightly fetch refreshes features; coefficients change only on deliberate retrain.
"""
    )
    formula = meta.get(
        "efficiency_formula",
        "PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*OREB + 0.3*DREB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV",
    )
    st.code(formula, language="text")
    cols = meta.get("columns") or meta.get("selected_columns")
    if cols:
        st.write("Feature columns:", ", ".join(f"`{c}`" for c in cols))

# --- Projections ---
with tab5:
    st.subheader("All projections")
    d = preds_df.copy()
    for c in ("Home_Win_Prob", "Away_Win_Prob", "Confidence"):
        if c in d.columns:
            d[c] = d[c].map(lambda x: f"{float(x):.1%}")
    keep = [
        c
        for c in (
            "Home_Team",
            "Away_Team",
            "Home_Win_Prob",
            "Away_Win_Prob",
            "Predicted_Winner",
            "Confidence",
            "Home_Player_Deficit",
            "Away_Player_Deficit",
            "Model",
        )
        if c in d.columns
    ]
    st.dataframe(d[keep], hide_index=True, use_container_width=True)

# --- Validation ---
with tab6:
    st.subheader("Historical preds vs results")
    selected_date = st.date_input("Date", datetime.date.today())
    history_file = f"history/preds_{selected_date.strftime('%Y-%m-%d')}.csv"
    if not os.path.exists(history_file):
        st.warning(f"No file `{history_file}`.")
    else:
        hist_df = pd.read_csv(history_file)
        if "Predicted_Winner" not in hist_df.columns:
            st.info("Legacy format — skip.")
        elif not os.path.exists("nba_stats.db"):
            st.info("Need `nba_stats.db` (or unpacked slim DB) to score actuals.")
        else:
            try:
                conn = sqlite3.connect("nba_stats.db")
                # GAME_DATE in DB is often 'YYYY-MM-DD' after modern backfill
                day = selected_date.isoformat()
                actuals = pd.read_sql(
                    "SELECT DISTINCT Game_ID, WL, MATCHUP, GAME_DATE FROM player_logs "
                    "WHERE GAME_DATE LIKE ?",
                    conn,
                    params=(f"%{day}%",),
                )
                if actuals.empty:
                    # try old NBA format
                    actuals = pd.read_sql(
                        "SELECT DISTINCT Game_ID, WL, MATCHUP, GAME_DATE FROM player_logs "
                        "WHERE GAME_DATE = ?",
                        conn,
                        params=(selected_date.strftime("%b %d, %Y"),),
                    )
                conn.close()
                if actuals.empty:
                    st.info("No results in DB for that date.")
                else:
                    actuals["Game_ID"] = _norm_game_id(actuals["Game_ID"])
                    hist_df = hist_df.copy()
                    hist_df["GAME_ID"] = _norm_game_id(hist_df["GAME_ID"])
                    results, correct, total = [], 0, 0
                    for _, pr in hist_df.iterrows():
                        hr = actuals[
                            (actuals["Game_ID"] == pr["GAME_ID"])
                            & (actuals["MATCHUP"].astype(str).str.contains("vs.", na=False))
                        ]
                        if hr.empty:
                            continue
                        actual_winner = (
                            pr["Home_Team"] if hr.iloc[0]["WL"] == "W" else pr["Away_Team"]
                        )
                        ok = pr["Predicted_Winner"] == actual_winner
                        correct += int(ok)
                        total += 1
                        results.append(
                            {
                                "Matchup": f"{pr['Away_Team']} @ {pr['Home_Team']}",
                                "Predicted": pr["Predicted_Winner"],
                                "Actual": actual_winner,
                                "Conf": f"{float(pr['Confidence']):.0%}",
                                "Hit": "yes" if ok else "no",
                            }
                        )
                    if results:
                        a, b = st.columns(2)
                        a.metric("Games", total)
                        b.metric("Accuracy", f"{correct / total:.0%}")
                        st.dataframe(
                            pd.DataFrame(results),
                            hide_index=True,
                            use_container_width=True,
                        )
            except Exception as e:
                st.error(str(e))
