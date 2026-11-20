"""
Build leak-safe team-game features, matchup rows, and LSTM sequences.

Splits (by target-game season):
  train/val pool: seasons <= 22023 (through 2023-24)
  report test:    22024 (2024-25) — metrics only after val selection
  frozen eval:    22025 (2025-26) — saved, never used for selection
  live:           22026 (2026-27) when present

Within the train pool, validation is the chronologically last 20% of
*target game dates* (true time split), not a row-index cut after team concat.
"""

from __future__ import annotations

import os
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from nba_ml.features import (
    FEATURE_COLUMNS,
    FROZEN_EVAL_SEASON,
    LOOKBACK,
    LIVE_SEASON,
    MATCHUP_FEATURE_COLUMNS,
    TEST_SEASON,
    TRAIN_SEASON_MAX,
    build_matchup_frame,
    build_sequences,
    build_team_game_features,
    compute_advanced_metrics,
)

DB_NAME = "nba_stats.db"
MODELS_DIR = "models"
DATA_DIR = "artifacts"


def load_player_logs(db_name: str = DB_NAME) -> pd.DataFrame:
    conn = sqlite3.connect(db_name)
    logs = pd.read_sql("SELECT * FROM player_logs", conn)
    conn.close()
    nan_count = logs.isna().sum().sum()
    if nan_count:
        numeric_cols = logs.select_dtypes(include=[np.number]).columns
        logs[numeric_cols] = logs[numeric_cols].fillna(0)
    return compute_advanced_metrics(logs)


def season_masks(meta: pd.DataFrame):
    train_pool = meta["season_id"].astype(str) <= TRAIN_SEASON_MAX
    test = meta["season_id"].astype(str) == TEST_SEASON
    frozen = meta["season_id"].astype(str) == FROZEN_EVAL_SEASON
    live = meta["season_id"].astype(str) == LIVE_SEASON
    return train_pool, test, frozen, live


def chronological_train_val(meta: pd.DataFrame, train_pool_mask: np.ndarray, val_frac=0.2):
    """Split train-pool rows by target game_date (not by array position)."""
    pool = meta.loc[train_pool_mask].copy()
    dates = np.sort(pool["game_date"].unique())
    cut = dates[int(len(dates) * (1 - val_frac))]
    train_idx = pool.index[pool["game_date"] < cut]
    val_idx = pool.index[pool["game_date"] >= cut]
    return train_idx, val_idx, cut


def main():
    print("=" * 60)
    print("  BUILD DATASETS — leak-safe + season holdout")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    logs = load_player_logs()
    print(f"\n  Loaded {len(logs)} player-game rows.")

    team_games, _ = build_team_game_features(logs, models_dir=MODELS_DIR)
    team_games.to_pickle(os.path.join(DATA_DIR, "team_games.pkl"))

    matchups = build_matchup_frame(team_games)
    matchups.to_pickle(os.path.join(DATA_DIR, "matchups.pkl"))
    print(f"  Matchup rows: {len(matchups)}")

    # --- Matchup splits ---
    m_train_pool = matchups["season_id"].astype(str) <= TRAIN_SEASON_MAX
    m_test = matchups["season_id"].astype(str) == TEST_SEASON
    m_frozen = matchups["season_id"].astype(str) == FROZEN_EVAL_SEASON
    m_dates = np.sort(matchups.loc[m_train_pool, "game_date"].unique())
    m_cut = m_dates[int(len(m_dates) * 0.8)]
    matchups_train = matchups.loc[m_train_pool & (matchups["game_date"] < m_cut)].copy()
    matchups_val = matchups.loc[m_train_pool & (matchups["game_date"] >= m_cut)].copy()
    matchups_test = matchups.loc[m_test].copy()
    matchups_frozen = matchups.loc[m_frozen].copy()

    for name, df in [
        ("matchups_train", matchups_train),
        ("matchups_val", matchups_val),
        ("matchups_test", matchups_test),
        ("matchups_frozen_2025_26", matchups_frozen),
    ]:
        path = os.path.join(DATA_DIR, f"{name}.pkl")
        df.to_pickle(path)
        print(f"  [SAVED] {path} ({len(df)} rows)")

    # Feature matrices for baseline
    def xy(df):
        X = df[MATCHUP_FEATURE_COLUMNS].astype(float).fillna(0).values
        y = df["home_win"].astype(int).values
        return X, y

    X_tr, y_tr = xy(matchups_train)
    X_va, y_va = xy(matchups_val)
    X_te, y_te = xy(matchups_test)
    np.savez(
        os.path.join(DATA_DIR, "matchup_arrays.npz"),
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        X_test=X_te,
        y_test=y_te,
        feature_names=np.array(MATCHUP_FEATURE_COLUMNS),
    )

    # --- LSTM sequences ---
    X, y, meta = build_sequences(team_games, lookback=LOOKBACK)
    print(f"\n  Sequences: {X.shape}, win rate={y.mean():.3f}")

    train_pool, test_mask, frozen_mask, live_mask = season_masks(meta)
    train_idx, val_idx, date_cut = chronological_train_val(meta, train_pool.values)
    test_idx = meta.index[test_mask]

    cut_ts = pd.Timestamp(date_cut)
    print(f"  Train dates < {cut_ts.date()} | Val from {cut_ts.date()}")
    print(f"  Train={len(train_idx)} Val={len(val_idx)} Test(2024-25)={len(test_idx)}")
    print(
        f"  Frozen(2025-26) excluded: {int(frozen_mask.sum())} | "
        f"Live(2026-27) rows: {int(live_mask.sum())}"
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    scaler = StandardScaler()
    n_features = X_train.shape[2]
    X_train_s = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val_s = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_s = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    X_train_s = np.nan_to_num(X_train_s)
    X_val_s = np.nan_to_num(X_val_s)
    X_test_s = np.nan_to_num(X_test_s)

    np.save("X_train.npy", X_train_s)
    np.save("X_val.npy", X_val_s)
    np.save("X_test.npy", X_test_s)
    np.save("y_train.npy", y_train)
    np.save("y_val.npy", y_val)
    np.save("y_test.npy", y_test)
    meta.loc[train_idx].to_pickle(os.path.join(DATA_DIR, "seq_meta_train.pkl"))
    meta.loc[val_idx].to_pickle(os.path.join(DATA_DIR, "seq_meta_val.pkl"))
    meta.loc[test_idx].to_pickle(os.path.join(DATA_DIR, "seq_meta_test.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(MODELS_DIR, "feature_columns.pkl"))

    print("\n  [SAVED] X_*.npy / y_*.npy + scaler.pkl")
    print("  Done.")


if __name__ == "__main__":
    main()
