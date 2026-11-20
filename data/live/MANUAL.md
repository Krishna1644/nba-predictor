# Live data drop-in

Daily extract writes these files (Actions uses **ESPN**; home machines can use `nba_api`).

## Required

| File | Columns (minimum) |
| --- | --- |
| `todays_games.csv` | `GAME_DATE_EST,GAME_ID,HOME_TEAM_ID,VISITOR_TEAM_ID,GAME_STATUS_TEXT` |
| `final_predictions.csv` | `GAME_ID,Home_Team,Away_Team,Home_Win_Prob,Away_Win_Prob,Predicted_Winner,Confidence` |

Optional but better:

| File | Notes |
| --- | --- |
| `todays_rosters.csv` | `TeamID,PLAYER_ID,PLAYER,POSITION` |
| `injuries.csv` | `Player,Injury,Status` (`Out` only counts in the model) |
| `schedule_context.csv` | `TEAM_ID,rest_days,is_back_to_back,games_last_7` (rest capped at 7 in model) |

Injury history (append-only): `data/injury_snapshots/YYYY-MM-DD.csv`.

## Commands

```bash
# Local (home IP): nba_api first, ESPN fallback + optional stats refresh
python scripts/fetch_live.py
python run_pipeline.py

# Cloud / GitHub Actions (ESPN extract — no stats.nba.com)
python run_pipeline.py --source espn --skip-stats

# Predict only from already-written CSVs
python run_pipeline.py --predict-only
```

Team IDs are NBA numeric IDs (e.g. Lakers `1610612747`). `GAME_ID` may be NBA or ESPN; the dashboard normalizes them.
