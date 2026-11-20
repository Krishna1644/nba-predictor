"""
Train Elo-only and logistic-regression matchup baselines.

Elo-only: P(home win) from Elo difference (+100 home-court Elo points).
Logistic: sklearn LogisticRegression on home-minus-away pre-game features.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_ml.features import MATCHUP_FEATURE_COLUMNS
from nba_ml.metrics import elo_win_prob, evaluate_probs, save_calibration_plot, write_metrics_table

DATA_DIR = "artifacts"
MODELS_DIR = "models"


def main():
    print("=" * 60)
    print("  TRAIN BASELINES — Elo + Logistic")
    print("=" * 60)

    matchups_train = pd.read_pickle(os.path.join(DATA_DIR, "matchups_train.pkl"))
    matchups_val = pd.read_pickle(os.path.join(DATA_DIR, "matchups_val.pkl"))
    matchups_test = pd.read_pickle(os.path.join(DATA_DIR, "matchups_test.pkl"))

    def pack(df):
        X = df[MATCHUP_FEATURE_COLUMNS].astype(float).fillna(0)
        y = df["home_win"].astype(int)
        return X, y

    X_tr, y_tr = pack(matchups_train)
    X_va, y_va = pack(matchups_val)
    X_te, y_te = pack(matchups_test)

    # Elo-only (no fitting beyond home-advantage constant)
    elo_tr = elo_win_prob(matchups_train["elo_diff"].values)
    elo_va = elo_win_prob(matchups_val["elo_diff"].values)
    elo_te = elo_win_prob(matchups_test["elo_diff"].values)

    # Logistic on matchup features
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipe.fit(X_tr, y_tr)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipe, os.path.join(MODELS_DIR, "logistic_baseline.pkl"))

    log_tr = pipe.predict_proba(X_tr)[:, 1]
    log_va = pipe.predict_proba(X_va)[:, 1]
    log_te = pipe.predict_proba(X_te)[:, 1]

    rows = [
        evaluate_probs(y_tr, elo_tr, "elo_only/train"),
        evaluate_probs(y_va, elo_va, "elo_only/val"),
        evaluate_probs(y_te, elo_te, "elo_only/test_2024-25"),
        evaluate_probs(y_tr, log_tr, "logistic/train"),
        evaluate_probs(y_va, log_va, "logistic/val"),
        evaluate_probs(y_te, log_te, "logistic/test_2024-25"),
    ]

    # Coefficient inspection
    coefs = dict(
        zip(
            MATCHUP_FEATURE_COLUMNS,
            pipe.named_steps["clf"].coef_.ravel().tolist(),
        )
    )
    with open(os.path.join(MODELS_DIR, "logistic_coefficients.json"), "w", encoding="utf-8") as f:
        json.dump(coefs, f, indent=2)

    write_metrics_table(
        rows,
        os.path.join(DATA_DIR, "baseline_metrics.json"),
        os.path.join(DATA_DIR, "baseline_metrics.md"),
    )
    save_calibration_plot(
        y_te,
        {"Elo only": elo_te, "Logistic": log_te},
        os.path.join(DATA_DIR, "calibration_baselines.png"),
    )

    # Persist test predictions for later LSTM comparison
    out = matchups_test[
        ["Game_ID", "game_date", "home_team", "away_team", "home_win", "elo_diff"]
    ].copy()
    out["elo_prob"] = elo_te
    out["logistic_prob"] = log_te
    out.to_csv(os.path.join(DATA_DIR, "holdout_baseline_preds.csv"), index=False)

    print("\nHoldout (2024-25):")
    for r in rows:
        if "test" in r["model"]:
            print(
                f"  {r['model']}: acc={r['accuracy']:.3f}  "
                f"logloss={r['log_loss']:.4f}  brier={r['brier']:.4f}  auc={r['auc']:.3f}"
            )
    print(f"\n  [SAVED] models/logistic_baseline.pkl")
    print(f"  [SAVED] artifacts/baseline_metrics.md")


if __name__ == "__main__":
    main()
