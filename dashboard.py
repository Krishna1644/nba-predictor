"""
Streamlit UI for the production logistic matchup model.
Reads precomputed CSVs from data/live/ (fallback: repo root). Does not retrain.
"""

from __future__ import annotations

import datetime
import json
import os
import unicodedata
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

from nba_ml.history_validation import (
    fetch_espn_standings,
    grade_preds,
    list_pred_dates,
    load_preds,
    summarize_season,
)
from nba_ml.team_ids import NBA_ID_TO_ABBR

st.set_page_config(page_title="NBA Win Probability", layout="wide")

REST_CAP = 7
SEASON_LABEL = {
    "22024": "2024-25",
    "22025": "2025-26",
    "22026": "2026-27",
}
# ESPN/CDN logos by NBA abbreviation (GSW, NYK, …)
LOGO_URL = "https://a.espncdn.com/i/teamlogos/nba/500/{abbr}.png"
ABBR_FOR_LOGO = {
    "NOP": "no",
    "NYK": "ny",
    "GSW": "gs",
    "SAS": "sa",
    "UTA": "utah",
    "WAS": "wsh",
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


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Streamlit/pyarrow rejects frames with duplicate column labels."""
    if df is None or df.empty:
        return df
    if not df.columns.duplicated().any():
        return df
    return df.loc[:, ~df.columns.duplicated()].copy()


def logo_url_for_team_name(name: str) -> str | None:
    # Reverse lookup from full name
    for tid, full in __import__("nba_ml.team_ids", fromlist=["NBA_ID_TO_NAME"]).NBA_ID_TO_NAME.items():
        if full == name:
            abbr = NBA_ID_TO_ABBR.get(tid)
            if not abbr:
                return None
            slug = ABBR_FOR_LOGO.get(abbr, abbr.lower())
            return LOGO_URL.format(abbr=slug)
    return None


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
    # Prefer feature_selection calibration_selected when present
    if "calibration_selected" in meta and "calibration" not in meta:
        meta["calibration"] = meta["calibration_selected"]
    if "selected_absence_feature" in meta and "absence_feature" not in meta:
        meta["absence_feature"] = meta["selected_absence_feature"]
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
    coefs = {}
    coef_path = "models/logistic_coefficients.json"
    if os.path.exists(coef_path):
        with open(coef_path, encoding="utf-8") as f:
            coefs = json.load(f)
    return meta, report, coefs


@st.cache_data(ttl=600)
def load_data():
    preds_path = _first_existing(
        "data/live/final_predictions.csv", "final_predictions.csv"
    )
    games_path = _first_existing("data/live/todays_games.csv", "todays_games.csv")
    if not preds_path or not games_path:
        return None

    preds = _dedupe_columns(pd.read_csv(preds_path))
    games = _dedupe_columns(pd.read_csv(games_path))
    if "GAME_ID" in preds.columns:
        preds["GAME_ID"] = _norm_game_id(preds["GAME_ID"])
    if "GAME_ID" in games.columns:
        games["GAME_ID"] = _norm_game_id(games["GAME_ID"])
    if "Confidence" in preds.columns:
        if "Win_Probability" not in preds.columns:
            preds["Win_Probability"] = preds["Confidence"]
        preds = preds.drop(columns=["Confidence"])

    def _csv(*names):
        p = _first_existing(*names)
        if not p:
            return pd.DataFrame()
        try:
            return _dedupe_columns(pd.read_csv(p))
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
        "preds_mtime": datetime.datetime.fromtimestamp(os.path.getmtime(preds_path)),
    }


@st.cache_data(ttl=1800)
def load_standings_and_elo(elo_dict: dict):
    standings = fetch_espn_standings()
    if standings.empty:
        return standings
    # Map ESPN abbr → our Elo keys
    from nba_ml.team_ids import nba_abbr_from_espn

    def elo_for(row):
        abbr = nba_abbr_from_espn(row.get("Abbr")) or row.get("Abbr")
        return float(elo_dict.get(abbr, 1500.0))

    standings = standings.copy()
    standings["Elo"] = standings.apply(elo_for, axis=1)
    standings["Team"] = standings["Team"].fillna("")
    return standings.sort_values(["Conference", "Win%"], ascending=[True, False])


def schedule_date_str(games: pd.DataFrame) -> str | None:
    if games.empty or "GAME_DATE_EST" not in games.columns:
        return None
    raw = str(games["GAME_DATE_EST"].iloc[0])
    return raw.split("T")[0]


def win_prob_col(pred) -> float:
    if "Win_Probability" in pred.index and pd.notna(pred.get("Win_Probability")):
        return float(pred["Win_Probability"])
    return float(pred["Confidence"])


# --- load ---
meta, report, coefs = load_model_meta()
bundle = load_data()

absence = meta.get("absence_feature", meta.get("selected_absence_feature", "efficiency"))
calib = meta.get("calibration", meta.get("calibration_selected", "none"))
value_key = meta.get("value_key", "impact_eff")

st.title("NBA Win Probability")
st.caption(
    f"Production: logistic · absence={absence} · calib={calib} · "
    f"train ≤2024-25 · report 2025-26 · live 2026-27"
)

with st.sidebar:
    st.header("Tonight")
    if bundle:
        st.write(f"Updated {bundle['preds_mtime'].strftime('%Y-%m-%d %H:%M')}")
        st.caption("Predictions are precomputed. This app never retrains.")
    st.divider()
    st.markdown(
        f"**Model**\n\n"
        f"- Absence: `{absence}`\n"
        f"- Calibration: `{calib}`\n"
        f"- Live season: **2026-27**"
    )

if bundle is None:
    st.error("Missing live predictions. Run the daily pipeline or drop CSVs in data/live/.")
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
    st.warning(f"Showing slate for **{slate}**, not today ({today_et} ET).")

tab_picks, tab_season, tab_model, tab_val = st.tabs(
    ["Tonight's picks", "Season", "Model", "Validation"]
)

# Green pick badge (works in dark theme)
st.markdown(
    """
<style>
.pick-winner {
  display: inline-block;
  background: rgba(34, 197, 94, 0.18);
  color: #4ADE80;
  border: 1px solid rgba(74, 222, 128, 0.55);
  border-radius: 8px;
  padding: 0.35rem 0.75rem;
  font-weight: 700;
  font-size: 1.05rem;
  line-height: 1.4;
}
.pick-winner span.prob {
  color: #86EFAC;
  font-weight: 600;
  margin-left: 0.35rem;
}
.pick-side-label {
  color: #4ADE80;
  font-weight: 700;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- Tonight's picks (Matchups + Projections) ---
with tab_picks:
    st.subheader(f"Tonight's picks · {slate or 'unknown date'}")

    # Compact card grid
    for _, pred in preds_df.iterrows():
        home, away = pred["Home_Team"], pred["Away_Team"]
        home_p = float(pred["Home_Win_Prob"])
        away_p = float(pred["Away_Win_Prob"])
        win_p = win_prob_col(pred)
        winner = str(pred["Predicted_Winner"])
        reason = pred.get("Reason") if "Reason" in pred.index else None
        if not isinstance(reason, str) or not reason:
            reason = "See Model tab for feature methodology."

        with st.container(border=True):
            c_logo_a, c_mid, c_logo_h = st.columns([1, 3, 1])
            la = logo_url_for_team_name(away)
            lh = logo_url_for_team_name(home)
            with c_logo_a:
                if la:
                    st.image(la, width=56)
                if winner == away:
                    st.markdown(
                        f'<p class="pick-side-label">{away}</p>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption(away)
            with c_logo_h:
                if lh:
                    st.image(lh, width=56)
                if winner == home:
                    st.markdown(
                        f'<p class="pick-side-label">{home}</p>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption(home)
            with c_mid:
                st.markdown(
                    f'<div class="pick-winner">{winner}'
                    f'<span class="prob">· {win_p:.0%} win probability</span></div>',
                    unsafe_allow_html=True,
                )
                st.progress(
                    min(max(home_p, 0.0), 1.0),
                    text=f"Home {home_p:.0%}  ·  Away {away_p:.0%}",
                )
                st.caption(reason)

            # Detail metrics
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Home win prob", f"{home_p:.1%}")
            d2.metric("Away win prob", f"{away_p:.1%}")
            hd = float(pred["Home_Player_Deficit"]) if "Home_Player_Deficit" in pred.index else None
            ad = float(pred["Away_Player_Deficit"]) if "Away_Player_Deficit" in pred.index else None
            if hd is not None:
                d3.metric(
                    "Home deficit",
                    f"{hd:.1f}",
                    help="Efficiency-weighted value of Status=Out rotation players (higher = more missing production).",
                )
            if ad is not None:
                d4.metric(
                    "Away deficit",
                    f"{ad:.1f}",
                    help="Efficiency-weighted value of Status=Out rotation players (higher = more missing production).",
                )

            # Rest line from preds or schedule_ctx
            if "Home_Rest" in pred.index:
                st.caption(
                    f"Rest {int(pred['Away_Rest'])}d / {int(pred['Home_Rest'])}d · "
                    f"B2B {'yes' if int(pred.get('Away_B2B', 0)) else 'no'} / "
                    f"{'yes' if int(pred.get('Home_B2B', 0)) else 'no'} · "
                    f"last-7 {int(pred.get('Away_G7', 0))} / {int(pred.get('Home_G7', 0))}"
                )
            elif not schedule_ctx.empty:
                try:
                    gid = pred["GAME_ID"]
                    g = games[games["GAME_ID"] == gid].iloc[0]
                    hid, aid = int(g["HOME_TEAM_ID"]), int(g["VISITOR_TEAM_ID"])
                    hc = schedule_ctx[schedule_ctx["TEAM_ID"] == hid]
                    ac = schedule_ctx[schedule_ctx["TEAM_ID"] == aid]
                    if not hc.empty and not ac.empty:
                        st.caption(
                            f"Rest {int(ac.iloc[0]['rest_days'])}d / {int(hc.iloc[0]['rest_days'])}d "
                            f"(cap {REST_CAP}) · "
                            f"B2B {'yes' if int(ac.iloc[0]['is_back_to_back']) else 'no'} / "
                            f"{'yes' if int(hc.iloc[0]['is_back_to_back']) else 'no'} · "
                            f"last-7 {int(ac.iloc[0]['games_last_7'])} / {int(hc.iloc[0]['games_last_7'])}"
                        )
                except Exception:
                    pass

            with st.expander("Injuries & roster"):
                try:
                    gid = pred["GAME_ID"]
                    g = games[games["GAME_ID"] == gid].iloc[0]
                    for tid, label in (
                        (int(g["VISITOR_TEAM_ID"]), away),
                        (int(g["HOME_TEAM_ID"]), home),
                    ):
                        st.markdown(f"**{label}**")
                        if not injuries.empty:
                            if "TeamID" in injuries.columns:
                                ti = injuries[injuries["TeamID"] == tid]
                            elif not rosters.empty:
                                names = {
                                    _norm_name(x)
                                    for x in rosters.loc[
                                        rosters["TeamID"] == tid, "PLAYER"
                                    ].tolist()
                                }
                                ti = injuries[
                                    injuries["Player"].map(_norm_name).isin(names)
                                ]
                            else:
                                ti = pd.DataFrame()
                            if not ti.empty:
                                ti = _dedupe_columns(ti)
                                cols = [
                                    c
                                    for c in ("Player", "Injury", "Status")
                                    if c in ti.columns
                                ]
                                st.dataframe(
                                    ti.loc[:, cols],
                                    hide_index=True,
                                    use_container_width=True,
                                )
                            else:
                                st.caption("No listed injuries for this team.")
                except Exception as e:
                    st.caption(str(e))

    st.divider()
    st.markdown("#### All projections")
    show = _dedupe_columns(preds_df.copy())
    if "Confidence" in show.columns and "Win_Probability" in show.columns:
        show = show.drop(columns=["Confidence"])
    elif "Confidence" in show.columns:
        show = show.rename(columns={"Confidence": "Win_Probability"})
    for c in ("Home_Win_Prob", "Away_Win_Prob", "Win_Probability"):
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").map(
                lambda x: f"{float(x):.1%}" if pd.notna(x) else ""
            )
    keep = []
    for c in (
        "Away_Team",
        "Home_Team",
        "Away_Win_Prob",
        "Home_Win_Prob",
        "Predicted_Winner",
        "Win_Probability",
        "Home_Player_Deficit",
        "Away_Player_Deficit",
        "Reason",
    ):
        if c in show.columns and c not in keep:
            keep.append(c)
    table = show.loc[:, keep].copy()
    if "Predicted_Winner" in table.columns:
        def _green_winner(col):
            if col.name != "Predicted_Winner":
                return [""] * len(col)
            return [
                "background-color: rgba(34,197,94,0.22); color: #4ADE80; font-weight: 700"
                for _ in col
            ]

        st.dataframe(
            table.style.apply(_green_winner, axis=0),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.dataframe(table, hide_index=True, use_container_width=True)
    st.download_button(
        "Download CSV",
        data=preds_df.to_csv(index=False).encode("utf-8"),
        file_name=f"projections_{slate or 'slate'}.csv",
        mime="text/csv",
    )

# --- Season ---
with tab_season:
    st.subheader("2026-27 standings")
    standings = load_standings_and_elo(elo_dict)
    if standings.empty:
        st.warning("Could not load ESPN standings.")
    else:
        for conf in ("East", "West", "Eastern Conference", "Western Conference"):
            block = standings[standings["Conference"].astype(str).str.contains(conf[:4], case=False, na=False)]
            if block.empty:
                continue
            st.markdown(f"**{conf}**")
            cols = [c for c in ("Team", "W", "L", "Win%", "Elo") if c in block.columns]
            view = _dedupe_columns(block.loc[:, cols].copy())
            st.dataframe(
                view.style.format({"Win%": "{:.1%}", "Elo": "{:.0f}"}),
                hide_index=True,
                use_container_width=True,
            )

# --- Model ---
with tab_model:
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

**Methodology:** Features are pre-game only. Absence uses a shared deficit function in
training and serving. Nightly Actions refresh the slate via ESPN; coefficients change
only on a deliberate retrain. BiLSTM is research-only and not production.
"""
    )
    formula = meta.get(
        "efficiency_formula",
        "PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*OREB + 0.3*DREB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV",
    )
    st.code(formula, language="text")

    if coefs:
        st.markdown("#### Coefficients")
        coef_df = (
            pd.DataFrame({"feature": list(coefs.keys()), "coef": list(coefs.values())})
            .sort_values("coef", key=lambda s: s.abs(), ascending=False)
        )
        st.bar_chart(coef_df.set_index("feature")["coef"], horizontal=True)

    cal_path = "artifacts/calibration_holdout.png"
    if os.path.exists(cal_path):
        st.markdown("#### Holdout calibration")
        st.image(cal_path, use_container_width=True)

# --- Validation ---
with tab_val:
    st.subheader("Predictions vs final scores")
    dates = list_pred_dates()
    if not dates:
        st.info("No graded history files yet under history/preds_*.csv.")
    else:
        # Default to latest date that has ESPN finals
        default_idx = len(dates) - 1
        default_date = dates[default_idx]
        for d in reversed(dates):
            g = grade_preds(load_preds(d), d)
            if not g.empty:
                default_date = d
                break
        selected = st.selectbox(
            "Date",
            options=list(reversed(dates)),
            index=list(reversed(dates)).index(default_date),
        )
        day_preds = load_preds(selected)
        graded = grade_preds(day_preds, selected)
        if graded.empty:
            st.info("No final scores available for that date yet (or GAME_ID/name mismatch).")
        else:
            a, b, c = st.columns(3)
            a.metric("Games", len(graded))
            b.metric("Accuracy", f"{graded['Hit'].mean():.0%}")
            y = graded["home_win"].astype(int).values
            p = graded["Home_Win_Prob"].astype(float).clip(1e-6, 1 - 1e-6).values
            from sklearn.metrics import log_loss

            c.metric("Log loss", f"{log_loss(y, p, labels=[0, 1]):.3f}")
            view = graded[
                ["Matchup", "Predicted_Winner", "Actual_Winner", "Home_Win_Prob", "Hit"]
            ].copy()
            view["Home_Win_Prob"] = view["Home_Win_Prob"].map(lambda x: f"{float(x):.0%}")
            view["Hit"] = view["Hit"].map(lambda x: "yes" if int(x) else "no")
            st.dataframe(view, hide_index=True, use_container_width=True)

        st.markdown("#### Rolling 2026-27 (history with results)")
        # Prefer dates in the live season window
        live_dates = [d for d in dates if d >= "2026-10-01"]
        all_g, metrics = summarize_season(live_dates or dates)
        if not metrics:
            st.caption("Not enough completed games in history to score yet.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("N", metrics["n"])
            m2.metric("Accuracy", f"{metrics['accuracy']:.1%}")
            m3.metric("Log loss", f"{metrics['log_loss']:.3f}")
            m4.metric("Brier", f"{metrics['brier']:.3f}")
            b1, b2 = st.columns(2)
            b1.metric("Home-always acc", f"{metrics['home_always_acc']:.1%}")
            b2.metric("Home-always log loss (p=0.55)", f"{metrics['home_always_ll']:.3f}")

            # Simple calibration table
            if len(all_g) >= 5:
                bins = pd.cut(all_g["Home_Win_Prob"], bins=np.linspace(0, 1, 6), include_lowest=True)
                cal = (
                    all_g.assign(bin=bins)
                    .groupby("bin", observed=False)
                    .agg(pred=("Home_Win_Prob", "mean"), actual=("home_win", "mean"), n=("home_win", "size"))
                    .dropna()
                )
                st.markdown("##### Calibration (binned)")
                st.dataframe(
                    cal.style.format({"pred": "{:.2f}", "actual": "{:.2f}"}),
                    use_container_width=True,
                )
                try:
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.plot([0, 1], [0, 1], "--", color="gray")
                    ax.scatter(cal["pred"], cal["actual"], s=cal["n"] * 8)
                    ax.set_xlabel("Predicted P(home)")
                    ax.set_ylabel("Actual home win rate")
                    ax.set_title("Live calibration")
                    st.pyplot(fig, clear_figure=True)
                except Exception:
                    pass
