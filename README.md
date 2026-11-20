# NBA Win Probability

Predicts regular-season NBA game winners with **calibrated home-win probabilities**.

**Production model:** logistic regression on leak-safe matchup features  
**Not production:** BiLSTM + Attention (kept for comparison; it loses the holdout)

| Model (2024-25 holdout, n=1225) | Accuracy | Log loss | Brier | AUC |
| --- | ---: | ---: | ---: | ---: |
| Elo only | 0.642 | 0.6280 | 0.2193 | 0.715 |
| **Logistic (shipped)** | **0.669** | **0.5975** | **0.2066** | **0.739** |
| BiLSTM (renormalized) | 0.647 | 0.6443 | 0.2262 | 0.700 |

Selection rule: lowest log loss on a held-out season. Calibration plot: `artifacts/calibration_holdout.png`.

---

## Why this project looks the way it does

Earlier versions had real methodology bugs:

1. **Starter minutes used full-season averages** (future leakage).
2. **Train/val split was a row cut after concatenating teams**, not a time split.
3. **Home/away LSTM scores were forced to sum to 1** instead of modeling the matchup jointly.

Those are fixed. Features are pre-game only. Evaluation is a **season holdout** (train through 2023-24, test 2024-25). The deep model had to beat Elo and logistic — it did not, so the README and dashboard tell that story instead of hiding it.

---

## Model card

| | |
| --- | --- |
| **Task** | P(home team wins) for NBA regular-season games |
| **Data** | `nba_api` LeagueGameLog player box scores, seasons 2015-16 → 2025-26 in SQLite |
| **Grain** | One row per game (home vs away) |
| **Label** | `home_win` |
| **Features** | Elo diff, rest / B2B / games-in-7 diffs, missing-starter-minutes diff (injury + prior MPG), rolling form diffs, home court |
| **Train window** | Seasons ≤ 2023-24 (with chronological val on last 20% of train dates) |
| **Test window** | 2024-25 full season |
| **Live season** | 2025-26 reserved for daily inference only |
| **Metrics** | Log loss, Brier, AUC, accuracy, calibration curve |
| **Known limits** | No market odds; injury report name matching is fuzzy; Elo is a simple K=20 system |

### Leakage controls

- Starter status = expanding mean minutes from **prior games only** (≥3 games, ≥25 MPG).
- Elo / opponent strength / rest are **pre-game**.
- Rolling form uses `shift(1)` so the current game’s box score is never a feature for itself.

---

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

# If nba_stats.db is missing, backfill once (slow; hits nba_api):
python backfill_history.py

python build_datasets.py
python train_baseline.py
python evaluate_holdout.py          # writes artifacts/holdout_metrics.*

# Daily predictions (needs todays_*.csv from the fetch scripts):
python run_pipeline.py
streamlit run dashboard.py
```

Optional LSTM comparison (needs TensorFlow):

```bash
python train_lstm.py --epochs 40
python evaluate_holdout.py
```

---

## Repo layout

| Path | Role |
| --- | --- |
| `nba_ml/features.py` | Leak-safe team-game + matchup features |
| `nba_ml/metrics.py` | Log loss / Brier / calibration helpers |
| `build_datasets.py` | Build features + season splits |
| `train_baseline.py` | Elo + logistic |
| `train_lstm.py` | BiLSTM + Attention (research / comparison) |
| `evaluate_holdout.py` | Holdout leaderboard |
| `predict_matchup.py` | Production inference |
| `predict_tonight.py` | Legacy LSTM inference (optional) |
| `dashboard.py` | Streamlit UI |
| `artifacts/` | Holdout metrics, calibration plots, preds |
| `_archive/` | Old heuristic weight tuner (retired) |

---

## Data note (BigData / 1946–2026 dump)

A separate Kaggle dump (`nba-data-1946-2026`) has 1.5M+ player-gamelog rows back to the 1940s. **This repo does not use it.** Reasons:

- Schema differs (no clean `MATCHUP`/`WL` without joining `dim_games`).
- Pre-3PT / different-pace eras hurt a modern win model more than they help.
- The local `nba_stats.db` already covers **2015-16 → 2025-26**, which is enough for Elo warm-up + a full held-out season.

Using 1960s box scores here would be portfolio theater, not better science.

---

## Deploy

This app is **Streamlit + sklearn**, not a Vercel serverless app. Free options that fit:

- [Streamlit Community Cloud](https://streamlit.io/cloud) (best match)
- Render / Railway free tier

Vercel would require rewriting to FastAPI/static; TensorFlow does not belong on free serverless even if you keep the LSTM around for research.

---

## Disclaimer

For research / portfolio demonstration only. Not betting advice.
