"""
Train Elo + logistic matchup models (Phase 1).

Production absence family = player deficit (not legacy starter).
Val picks impact composite vs minutes-based deficit.
Calibration (isotonic / Platt) chosen on VAL only; report on 2024-25.
2025-26 never touched.
"""

from __future__ import annotations

import inspect
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_ml.features import (
    ABSENCE_FEATURE_MINUTES,
    ABSENCE_FEATURE_PLAYER,
    MATCHUP_CORE_FEATURES,
)
from nba_ml.metrics import (
    elo_win_prob,
    evaluate_probs,
    save_calibration_plot,
    write_metrics_table,
)

DATA_DIR = "artifacts"
MODELS_DIR = "models"
N_BOOT = 1000
RNG = np.random.default_rng(42)


def make_pipe():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def pack(df: pd.DataFrame, cols: list[str]):
    X = df[cols].astype(float).fillna(0)
    y = df["home_win"].astype(int)
    return X, y


def paired_bootstrap_logloss_delta(y, p_a, p_b, n_boot=N_BOOT):
    """Delta = logloss(a) - logloss(b). Negative => a better."""
    y = np.asarray(y).astype(int)
    p_a = np.clip(np.asarray(p_a, float), 1e-6, 1 - 1e-6)
    p_b = np.clip(np.asarray(p_b, float), 1e-6, 1 - 1e-6)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        la = log_loss(y[idx], p_a[idx], labels=[0, 1])
        lb = log_loss(y[idx], p_b[idx], labels=[0, 1])
        deltas.append(la - lb)
    deltas = np.asarray(deltas)
    return {
        "delta_mean": float(deltas.mean()),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "n_boot": int(n_boot),
        "definition": "delta = logloss(impact) - logloss(minutes); negative => impact better",
    }


def main():
    print("=" * 60)
    print("  PHASE 1 — logistic + impact deficit + calibration")
    print("=" * 60)

    matchups_train = pd.read_pickle(os.path.join(DATA_DIR, "matchups_train.pkl"))
    matchups_val = pd.read_pickle(os.path.join(DATA_DIR, "matchups_val.pkl"))
    matchups_test = pd.read_pickle(os.path.join(DATA_DIR, "matchups_test.pkl"))

    if "season_id" in matchups_test.columns:
        if matchups_test["season_id"].astype(str).eq("22025").any():
            raise RuntimeError("matchups_test contains frozen 2025-26")

    candidates = {
        "impact": MATCHUP_CORE_FEATURES + [ABSENCE_FEATURE_PLAYER],
        "minutes": MATCHUP_CORE_FEATURES + [ABSENCE_FEATURE_MINUTES],
    }

    val_rows = []
    fitted = {}
    val_probs = {}
    for name, cols in candidates.items():
        X_tr, y_tr = pack(matchups_train, cols)
        X_va, y_va = pack(matchups_val, cols)
        pipe = make_pipe()
        pipe.fit(X_tr, y_tr)
        fitted[name] = (pipe, cols)
        va_prob = pipe.predict_proba(X_va)[:, 1]
        val_probs[name] = va_prob
        metrics = evaluate_probs(y_va, va_prob, f"logistic_{name}/val")
        val_rows.append(metrics)
        print(
            f"  VAL {name}: logloss={metrics['log_loss']:.4f}  "
            f"brier={metrics['brier']:.4f}  acc={metrics['accuracy']:.3f}"
        )

    y_va = matchups_val["home_win"].astype(int).values
    boot = paired_bootstrap_logloss_delta(
        y_va, val_probs["impact"], val_probs["minutes"]
    )
    print(
        f"\n  Bootstrap VAL Δll(impact-minutes): "
        f"{boot['delta_mean']:.4f} "
        f"[{boot['ci95_low']:.4f}, {boot['ci95_high']:.4f}]"
    )

    # Prefer impact when CI includes 0 (noise) — train/serve consistency story
    ci_includes_zero = boot["ci95_low"] <= 0 <= boot["ci95_high"]
    if ci_includes_zero:
        best_name = "impact"
        selection_reason = (
            "forced_impact_ci_includes_zero_train_serve_consistency"
        )
    else:
        best_name = min(
            candidates.keys(),
            key=lambda n: next(r["log_loss"] for r in val_rows if n in r["model"]),
        )
        selection_reason = "val_logloss_winner"

    best_cols = candidates[best_name]
    print(f"  >>> Selected: {best_name} ({selection_reason})")

    # Calibration on VAL for the selected raw model
    X_tr, y_tr = pack(matchups_train, best_cols)
    X_va, y_va_s = pack(matchups_val, best_cols)
    base = make_pipe()
    base.fit(X_tr, y_tr)

    calib_results = []
    calibrated = {"none": base}
    raw_va = base.predict_proba(X_va)[:, 1]
    calib_results.append(
        {
            "method": "none",
            **{k: evaluate_probs(y_va_s, raw_va, "x")[k] for k in ("log_loss", "brier", "accuracy", "auc")},
        }
    )

    for method in ("isotonic", "sigmoid"):  # sigmoid = Platt
        cal = CalibratedClassifierCV(make_pipe(), method=method, cv=3)
        cal.fit(X_tr, y_tr)
        p = cal.predict_proba(X_va)[:, 1]
        calibrated[method] = cal
        m = evaluate_probs(y_va_s, p, "x")
        calib_results.append(
            {
                "method": method,
                "log_loss": m["log_loss"],
                "brier": m["brier"],
                "accuracy": m["accuracy"],
                "auc": m["auc"],
            }
        )
        print(
            f"  VAL calib[{method}]: logloss={m['log_loss']:.4f}  brier={m['brier']:.4f}"
        )

    best_calib = min(calib_results, key=lambda r: r["log_loss"])["method"]
    print(f"  >>> Calibration selected on VAL: {best_calib}")

    # Deploy: refit on train+val, then wrap calibration fitted on train+val
    # using the chosen method (isotonic/platt need a holdout — use cv on train+val)
    train_val = pd.concat([matchups_train, matchups_val], ignore_index=True)
    X_tv, y_tv = pack(train_val, best_cols)
    if best_calib == "none":
        deploy = make_pipe()
        deploy.fit(X_tv, y_tv)
    else:
        deploy = CalibratedClassifierCV(
            make_pipe(), method=best_calib, cv=3
        )
        deploy.fit(X_tv, y_tv)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(deploy, os.path.join(MODELS_DIR, "logistic_baseline.pkl"))
    with open(os.path.join(MODELS_DIR, "logistic_feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "absence_feature": best_name,
                "calibration": best_calib,
                "columns": best_cols,
                "impact_formula": "PTS + 1.5*AST + REB + STL + BLK - TOV",
                "replacement_adjustment": "skipped_no_reliable_position_data",
            },
            f,
            indent=2,
        )

    # Coefficients from underlying logistic when possible
    try:
        clf = deploy
        if hasattr(deploy, "calibrated_classifiers_"):
            # average coefs across CV folds
            coefs_list = []
            for c in deploy.calibrated_classifiers_:
                est = c.estimator
                if hasattr(est, "named_steps"):
                    coefs_list.append(est.named_steps["clf"].coef_.ravel())
            coef = np.mean(coefs_list, axis=0)
        else:
            coef = deploy.named_steps["clf"].coef_.ravel()
        coefs = dict(zip(best_cols, coef.tolist()))
    except Exception:
        coefs = {}
    with open(os.path.join(MODELS_DIR, "logistic_coefficients.json"), "w", encoding="utf-8") as f:
        json.dump(coefs, f, indent=2)

    # Report 2024-25
    X_te, y_te = pack(matchups_test, best_cols)
    log_te = deploy.predict_proba(X_te)[:, 1]
    elo_te = elo_win_prob(matchups_test["elo_diff"].values)

    # Also report the other absence variant (uncalibrated refit) for transparency
    report_rows = [
        evaluate_probs(y_te, elo_te, "elo_only/test_2024-25"),
        evaluate_probs(y_te, log_te, f"logistic_{best_name}_{best_calib}/test_2024-25"),
    ]
    for name, cols in candidates.items():
        if name == best_name:
            continue
        alt = make_pipe()
        alt.fit(*pack(train_val, cols))
        report_rows.append(
            evaluate_probs(
                y_te,
                alt.predict_proba(pack(matchups_test, cols)[0])[:, 1],
                f"logistic_{name}/test_2024-25_not_selected",
            )
        )
    report_rows.extend(val_rows)

    write_metrics_table(
        report_rows,
        os.path.join(DATA_DIR, "baseline_metrics.json"),
        os.path.join(DATA_DIR, "baseline_metrics.md"),
    )
    save_calibration_plot(
        y_te,
        {
            "Elo only": elo_te,
            f"Logistic {best_name}+{best_calib}": log_te,
        },
        os.path.join(DATA_DIR, "calibration_holdout.png"),
    )

    out = matchups_test[
        ["Game_ID", "game_date", "home_team", "away_team", "home_win", "elo_diff"]
    ].copy()
    out["elo_prob"] = elo_te
    out["logistic_prob"] = log_te
    out.to_csv(os.path.join(DATA_DIR, "holdout_baseline_preds.csv"), index=False)

    # Guardrail metadata for tests
    pipe_src = inspect.getsource(make_pipe)

    selection = {
        "selection_split": "validation",
        "selection_metric": "log_loss",
        "production_absence_family": "player_deficit",
        "selected_absence_feature": best_name,
        "selection_reason": selection_reason,
        "selected_columns": best_cols,
        "calibration_selected": best_calib,
        "calibration_val": calib_results,
        "bootstrap_val_logloss_delta_impact_minus_minutes": boot,
        "validation_metrics": val_rows,
        "report_test_season": "2024-25",
        "frozen_season": "2025-26",
        "impact_formula": "PTS + 1.5*AST + REB + STL + BLK - TOV",
        "replacement_adjustment": "skipped_no_reliable_position_data",
        "guards": {
            "class_weight_kwarg_present": "class_weight=" in pipe_src,
            "home_court_in_core_features": "home_court" in MATCHUP_CORE_FEATURES,
        },
        "notes": (
            "Legacy missing_starter not in production candidates. "
            "If bootstrap CI for impact-minutes includes 0, ship impact for "
            "train/serve consistency. Replacement-at-position skipped. "
            "2025-26 not evaluated."
        ),
    }
    with open(os.path.join(DATA_DIR, "feature_selection.json"), "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    te = report_rows[1]
    with open(os.path.join(DATA_DIR, "model_selection.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"best_model=logistic_{best_name}_{best_calib}\n"
            f"selection_split=validation\n"
            f"test_log_loss={te['log_loss']:.6f}\n"
            f"test_brier={te['brier']:.6f}\n"
            f"test_accuracy={te['accuracy']:.6f}\n"
            f"bootstrap_delta_mean={boot['delta_mean']:.6f}\n"
            f"bootstrap_ci95=[{boot['ci95_low']:.6f}, {boot['ci95_high']:.6f}]\n"
            f"frozen_eval_season=2025-26\n"
        )

    print("\nReport test (2024-25):")
    for r in report_rows:
        if "test_2024-25" in r["model"]:
            print(
                f"  {r['model']}: acc={r['accuracy']:.3f}  "
                f"logloss={r['log_loss']:.4f}  brier={r['brier']:.4f}  auc={r['auc']:.3f}"
            )
    print("  [NOTE] 2025-26 left untouched.")


if __name__ == "__main__":
    main()
