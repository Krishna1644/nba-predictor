# NBA Win Probability

Portfolio project: **predict regular-season NBA winners** with honest evaluation, leak-safe features, and a live Streamlit dashboard that refreshes daily.

**Live app:** deploy `dashboard.py` on [Streamlit Community Cloud](https://streamlit.io/cloud) from this repo (`main`).  
**Production model:** logistic regression + **efficiency-weighted player deficit** (VAL-selected). Calibration = **none** (raw logistic beat Platt/isotonic on VAL).  
**Not production:** BiLSTM + Attention — kept as a comparison that did **not** win.

---

## Results at a glance

### 2025-26 report holdout (n=1225) — never used for model selection

| Model | Accuracy | Log loss | Brier | AUC |
| --- | ---: | ---: | ---: | ---: |
| Elo only | 0.660 | 0.6174 | 0.2136 | 0.726 |
| **Logistic efficiency (shipped)** | **0.685** | **0.5913** | **0.2032** | **0.742** |
| Counting impact (not selected) | 0.684 | 0.5920 | 0.2034 | 0.742 |
| Minutes deficit (not selected) | 0.681 | 0.5946 | 0.2043 | 0.740 |

### What was chosen on validation only

| Choice | VAL log loss | Notes |
| --- | ---: | --- |
| **Efficiency deficit (shipped)** | **0.6008** | Beat counting (0.6016) and minutes (0.6044) |
| Counting | 0.6016 | |
| Minutes | 0.6044 | |
| Calibration **none** | **0.60080** | Platt/sigmoid 0.60086; isotonic worse |

- Paired bootstrap Δll(efficiency − counting): **−0.0008 [−0.0013, −0.0003]** (1000 resamples).
- Train/val: seasons **2015-16 → 2024-25**. Chronological VAL = last 20% of those dates.
- **2025-26 is report-only** (frozen; not used to pick features or calibration).
- **Live season:** 2026-27.

Plots / JSON: `artifacts/calibration_holdout.png`, `artifacts/feature_selection.json`, `artifacts/baseline_metrics.md`.

---

## What recruiters should notice

1. **Leakage was fixed, not papered over.** Early versions used season-long starter minutes and a non-time split. Those bugs are documented below; features are pre-game only now.
2. **The deep model lost.** BiLSTM + Attention was trained and evaluated on the same holdout. Logistic + Elo won, so production is logistic — the README says so.
3. **Selection vs report are separated.** Feature/calibration choices use VAL only; 2025-26 metrics are a one-shot report.
4. **Serving matches training.** Absence uses one shared function (`compute_team_absence_deficit`) for DNP-in-train and `Status=Out`-at-serve.
5. **Cloud engineering is real.** `stats.nba.com` blocks GitHub Actions IPs. Live extract uses a shared interface with **ESPN** in CI and **nba_api** (home IP) locally.

---

## Model card

| | |
| --- | --- |
| **Task** | P(home team wins) for NBA regular-season games |
| **Data** | Player box scores → SQLite (`nba_api` history; ESPN for live slate in cloud) |
| **Grain** | One row per game (home vs away) |
| **Label** | `home_win` |
| **Features** | Elo diff; rest / B2B / games-in-7 diffs (rest capped at 7); rolling form diffs; efficiency player-deficit diff |
| **Absence value (shipped)** | Efficiency composite, blended prior `w = gp/(gp+10)` |
| **Train window** | 2015-16 → 2024-25 |
| **Report test** | 2025-26 full season |
| **Live** | 2026-27 |
| **Metrics** | Log loss, Brier, AUC, accuracy, calibration |
| **Known limits** | Train DNP ≈ Out proxy; serve requires Status=`Out`. No position replacement. No market odds. Roster/injury name matching can miss two-way / new players until DB catches up. |

**Efficiency formula:**  
`PTS + 0.4·FGM − 0.7·FGA − 0.4·(FTA−FTM) + 0.7·OREB + 0.3·DREB + STL + 0.7·AST + 0.7·BLK − 0.4·PF − TOV`

### Leakage controls

- Player priors = expanding means from **games played only**, blended with prior season via `gp/(gp+10)`.
- Elo / rest / form are **pre-game**; rest days capped at 7.
- Rolling form uses `shift(1)` so tonight’s box score is never a feature for itself.

**Retrain vs daily refresh:** the nightly job does **not** change coefficients. It refreshes schedule, injuries, and predictions. Weights change only when you run `train_baseline.py`.

---

## Why this project looks the way it does

Earlier versions had real methodology bugs:

1. **Starter minutes used full-season averages** (future leakage).
2. **Train/val was a row cut after concatenating teams**, not a time split.
3. **Home/away LSTM scores were forced to sum to 1** instead of modeling the matchup jointly.

Those are fixed. The dashboard and this README report the logistic win over LSTM instead of hiding it.

---

## Live pipeline (local + GitHub Actions)

```text
ESPN or nba_api  →  data/live/*.csv  →  logistic predict  →  Streamlit
                         ↕
              data/injury_snapshots/YYYY-MM-DD.csv
```

| Where | Source | What it does |
| --- | --- | --- |
| Your PC | `nba_api` → ESPN fallback | Full extract + optional player-log refresh |
| GitHub Actions | **ESPN only** (`--source espn --skip-stats`) | Games + injuries; rest from ESPN recent finals; rosters/priors from slim DB |

- Workflow: `.github/workflows/daily_predict.yml` — cron **14:00 UTC** (~9 AM ET) or **Run workflow**.
- Shared providers: `nba_ml/live_sources.py` (`NbaApiSource` / `EspnSource`).
- Slim DB for CI: `data/nba_stats_live.db.gz` (export with `scripts/export_live_db.py`).

Dashboard tabs: **Tonight's picks** · **Season** (2026-27 standings + Elo) · **Model** · **Validation** (history vs final scores).

---

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

# One-time history (slow; needs nba_api / home IP):
python backfill_history.py
python build_datasets.py
python train_baseline.py
python evaluate_holdout.py

# Daily (local):
python run_pipeline.py
streamlit run dashboard.py

# Same path Actions uses:
python run_pipeline.py --source espn --skip-stats
```

Optional LSTM comparison (TensorFlow via `requirements-train.txt`):

```bash
python train_lstm.py --epochs 40
python evaluate_holdout.py
```

Tests:

```bash
pytest tests/ -q
```

---

## Repo layout

| Path | Role |
| --- | --- |
| `nba_ml/features.py` | Leak-safe features + shared absence deficit |
| `nba_ml/player_priors.py` | Pre-game priors, injury → player_id matching |
| `nba_ml/live_sources.py` | Live extract interface (nba_api / ESPN) |
| `nba_ml/history_validation.py` | Grade `history/preds_*.csv` vs finals |
| `build_datasets.py` / `train_baseline.py` | Dataset + shipped logistic |
| `predict_matchup.py` | Production inference |
| `scripts/fetch_live.py` | CLI live extract → `data/live/` |
| `run_pipeline.py` | Extract + predict |
| `dashboard.py` | Streamlit UI (dark theme) |
| `data/live/` | Nightly CSVs for the site |
| `data/injury_snapshots/` | Append-only injury history |
| `artifacts/` | Holdout metrics, calibration, selection JSON |
| `.github/workflows/daily_predict.yml` | ESPN extract → predict → commit |

---

## Deploy

- **Streamlit Community Cloud** from `main` (`dashboard.py`) — best fit.
- Render / Railway also work (Streamlit + sklearn, not serverless).

After push, Cloud redeploys; live CSVs update when Actions commits `data/live/`.

---

## Disclaimer

Research / portfolio demonstration only. **Not betting advice.**
