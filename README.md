# NBA Win Probability

Predicts regular-season NBA game winners with **calibrated home-win probabilities**.

**Production model:** logistic (VAL-selected calib = none this round), **efficiency-weighted player deficit**  
**Not production:** BiLSTM + Attention (comparison only)

### 2025-26 report (n=1225) — not used for selection

| Model | Accuracy | Log loss | Brier | AUC |
| --- | ---: | ---: | ---: | ---: |
| Elo only | 0.660 | 0.6174 | 0.2136 | 0.726 |
| **Logistic efficiency (shipped)** | **0.685** | **0.5913** | **0.2032** | **0.742** |
| Counting impact (not selected) | 0.684 | 0.5920 | 0.2034 | 0.742 |
| Minutes deficit (not selected) | 0.681 | 0.5946 | 0.2043 | 0.740 |

### Validation selection notes

| Absence value | VAL log loss | VAL Brier |
| --- | ---: | ---: |
| **Efficiency (shipped)** | **0.6008** | **0.2074** |
| Counting | 0.6016 | 0.2077 |
| Minutes | 0.6044 | 0.2089 |

- Paired bootstrap Δll(efficiency − counting): **−0.0008 [−0.0013, −0.0003]** (1000 resamples).
- Calibration on VAL: **none** edged Platt/sigmoid; isotonic worse. Re-checked each retrain.
- Train/val pool: seasons **2015-16 → 2024-25** (`SEASON_ID <= 22024`).
- **2025-26 is report-only** (never used to pick features / calib).
- **Live:** 2026-27 (`fetch_*` + `run_pipeline.py`).
- Replacement-at-position: **skipped** (no reliable position in box logs).

**Efficiency formula (shipped):**  
`PTS + 0.4·FGM − 0.7·FGA − 0.4·(FTA−FTM) + 0.7·OREB + 0.3·DREB + STL + 0.7·AST + 0.7·BLK − 0.4·PF − TOV`  
(pre-game blend `w = gp/(gp+10)`).

Calibration plot: `artifacts/calibration_holdout.png`. Details: `artifacts/feature_selection.json`.

---

## Why this project looks the way it does

Earlier versions had real methodology bugs:

1. **Starter minutes used full-season averages** (future leakage).
2. **Train/val split was a row cut after concatenating teams**, not a time split.
3. **Home/away LSTM scores were forced to sum to 1** instead of modeling the matchup jointly.

Those are fixed. Features are pre-game only. Evaluation is a **season holdout**. The deep model had to beat Elo and logistic — it did not, so the README and dashboard tell that story instead of hiding it.

---

## Model card

| | |
| --- | --- |
| **Task** | P(home team wins) for NBA regular-season games |
| **Data** | `nba_api` LeagueGameLog player box scores in SQLite |
| **Grain** | One row per game (home vs away) |
| **Label** | `home_win` |
| **Features** | Elo diff, rest / B2B / games-in-7 diffs (rest capped at 7), efficiency player deficit diff, rolling form diffs |
| **Train window** | Seasons 2015-16 → 2024-25 (chronological val = last 20% of those dates) |
| **Report test** | 2025-26 full season (selection-free) |
| **Live** | 2026-27 |
| **Metrics** | Log loss, Brier, AUC, accuracy, calibration curve |
| **Known limits** | Train uses box-score DNP as Out proxy; serve uses Status=`Out`. No lineup role reassignment. No position-based replacement. No market odds. |

### Leakage controls

- Player priors = expanding means from **games played only**, blended with prior season via `gp/(gp+10)`.
- Elo / opponent strength / rest are **pre-game**; rest_days capped at 7.
- Rolling form uses `shift(1)` so the current game’s box score is never a feature for itself.

**Retrain vs daily refresh:** scoring games or running the nightly pipeline does **not** change logistic weights. Nightly runs refresh schedule / injuries / priors from new box scores. Weights change only when you deliberately run `train_baseline.py`.

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

# Daily predictions:
python run_pipeline.py
streamlit run dashboard.py
```

Optional LSTM comparison (needs TensorFlow):

```bash
python train_lstm.py --epochs 40
python evaluate_holdout.py
```

---

## Auto-update the website (daily)

The Streamlit app reads **precomputed CSVs** under `data/live/` (fallback: repo root). It does not retrain.

1. Deploy the app on [Streamlit Community Cloud](https://streamlit.io/cloud) from this repo (`dashboard.py`).
2. Enable GitHub Action `.github/workflows/daily_predict.yml` (cron ~14:00 UTC / workflow_dispatch).
3. Slim DB is **committed** as `data/nba_stats_live.db.gz` (~3MB; last 2–3 seasons only). Refresh with:
   ```bash
   python scripts/export_live_db.py
   ```
   Full `nba_stats.db` stays local for training — daily Actions never need 2015–23.
4. Streamlit Cloud redeploys when Actions commits `data/live/`; cache TTL is 10 minutes.

Local equivalent:

```bash
python run_pipeline.py   # writes final_predictions.csv and data/live/
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
| `run_pipeline.py` | Daily fetch + predict |
| `dashboard.py` | Streamlit UI |
| `data/live/` | Tracked nightly outputs for the website |
| `artifacts/` | Holdout metrics, calibration plots, preds |
| `_archive/` | Old heuristic weight tuner (retired) |

---

## Data note (BigData / 1946–2026 dump)

A separate Kaggle dump (`nba-data-1946-2026`) has 1.5M+ player-gamelog rows back to the 1940s. **This repo does not use it.** Local `nba_stats.db` covers modern seasons needed for Elo warm-up + holdout.

---

## Deploy

This app is **Streamlit + sklearn**, not a Vercel serverless app:

- [Streamlit Community Cloud](https://streamlit.io/cloud) (best match)
- Render / Railway free tier

---

## Disclaimer

For research / portfolio demonstration only. Not betting advice.
