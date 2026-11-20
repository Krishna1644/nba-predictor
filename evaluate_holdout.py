"""
Compare Elo, logistic, and LSTM on the 2025-26 holdout season.

LSTM matchup probability: predict home and away sequences independently,
then renormalize (same as predict_tonight.py) so we evaluate the shipped rule.
Also reports home-only sigmoid for reference.
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from nba_ml.metrics import (
    elo_win_prob,
    evaluate_probs,
    save_calibration_plot,
    write_metrics_table,
)

DATA_DIR = "artifacts"
MODELS_DIR = "models"


def lstm_matchup_probs(model, X_test, meta_test: pd.DataFrame):
    """Build per-game home-win probs from team-level sequence preds."""
    raw = model.predict(X_test, verbose=0).ravel()
    meta = meta_test.reset_index(drop=True).copy()
    meta["raw_p"] = raw

    rows = []
    for gid, g in meta.groupby("Game_ID"):
        home = g[g["is_home"] == 1]
        away = g[g["is_home"] == 0]
        if home.empty or away.empty:
            continue
        home_raw = float(home.iloc[0]["raw_p"])
        away_raw = float(away.iloc[0]["raw_p"])
        total = home_raw + away_raw
        if total > 0:
            home_p = home_raw / total
        else:
            home_p = 0.5
        # Label: home team win from meta (need actual). Use home row's y via index.
        rows.append(
            {
                "Game_ID": gid,
                "lstm_renorm": home_p,
                "lstm_home_only": home_raw,
                "home_idx": int(home.index[0]),
            }
        )
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("  HOLDOUT EVAL — 2025-26")
    print("=" * 60)

    baseline = pd.read_csv(os.path.join(DATA_DIR, "holdout_baseline_preds.csv"))
    y = baseline["home_win"].astype(int).values

    rows = [
        evaluate_probs(y, baseline["elo_prob"].values, "elo_only"),
        evaluate_probs(y, baseline["logistic_prob"].values, "logistic"),
    ]
    prob_plot = {
        "Elo only": baseline["elo_prob"].values,
        "Logistic": baseline["logistic_prob"].values,
    }

    model_path = os.path.join(MODELS_DIR, "lstm_model.keras")
    if os.path.exists(model_path) and os.path.exists("X_test.npy"):
        import tensorflow as tf
        from train_lstm import Attention

        X_test = np.load("X_test.npy")
        meta_test = pd.read_pickle(os.path.join(DATA_DIR, "seq_meta_test.pkl"))
        # Align y for sequences is team-level; matchup merge below
        model = tf.keras.models.load_model(
            model_path, custom_objects={"Attention": Attention}
        )
        lstm_df = lstm_matchup_probs(model, X_test, meta_test)
        def norm_gid(s):
            return s.astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")

        baseline = baseline.copy()
        baseline["Game_ID"] = norm_gid(baseline["Game_ID"])
        lstm_df["Game_ID"] = norm_gid(lstm_df["Game_ID"])
        merged = baseline.merge(lstm_df, on="Game_ID", how="inner")
        print(f"  LSTM matched games: {len(merged)} / {len(baseline)}")
        if len(merged) > 50:
            y_m = merged["home_win"].astype(int).values
            rows.append(
                evaluate_probs(y_m, merged["lstm_renorm"].values, "lstm_renorm")
            )
            rows.append(
                evaluate_probs(y_m, merged["lstm_home_only"].values, "lstm_home_only")
            )
            prob_plot["LSTM (renorm)"] = merged["lstm_renorm"].values
            # Recompute elo/logistic on same matched subset for fair plot
            prob_plot = {
                "Elo only": merged["elo_prob"].values,
                "Logistic": merged["logistic_prob"].values,
                "LSTM (renorm)": merged["lstm_renorm"].values,
            }
            rows = [
                evaluate_probs(y_m, merged["elo_prob"].values, "elo_only"),
                evaluate_probs(y_m, merged["logistic_prob"].values, "logistic"),
                evaluate_probs(y_m, merged["lstm_renorm"].values, "lstm_renorm"),
                evaluate_probs(y_m, merged["lstm_home_only"].values, "lstm_home_only"),
            ]
            merged.to_csv(
                os.path.join(DATA_DIR, "holdout_all_preds.csv"), index=False
            )
    else:
        print("  [SKIP] LSTM model or X_test.npy not found — baseline-only report.")

    write_metrics_table(
        rows,
        os.path.join(DATA_DIR, "holdout_metrics.json"),
        os.path.join(DATA_DIR, "holdout_metrics.md"),
    )
    save_calibration_plot(
        # use last y aligned to prob_plot lengths
        y if "LSTM (renorm)" not in prob_plot else merged["home_win"].astype(int).values,
        prob_plot,
        os.path.join(DATA_DIR, "calibration_holdout.png"),
    )

    print("\nResults:")
    for r in rows:
        print(
            f"  {r['model']:<16} acc={r['accuracy']:.3f}  "
            f"logloss={r['log_loss']:.4f}  brier={r['brier']:.4f}  auc={r['auc']:.3f}"
        )

    # Winner declaration
    scored = [r for r in rows if r["model"] in ("elo_only", "logistic", "lstm_renorm")]
    if scored:
        best = min(scored, key=lambda r: r["log_loss"])
        print(f"\n  >>> Best by log loss: {best['model']}")
        with open(os.path.join(DATA_DIR, "model_selection.txt"), "w", encoding="utf-8") as f:
            f.write(
                f"best_model={best['model']}\n"
                f"selection_metric=log_loss\n"
                f"log_loss={best['log_loss']:.6f}\n"
                f"brier={best['brier']:.6f}\n"
                f"accuracy={best['accuracy']:.6f}\n"
            )


if __name__ == "__main__":
    main()
