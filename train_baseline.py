"""
Train Elo + logistic. VAL picks among counting impact / efficiency impact / minutes.
Calibrate on VAL; report 2025-26 only (never used for selection).
Train window: seasons <= 2024-25. Live serve: 2026-27.
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
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_ml.features import (
    ABSENCE_FEATURE_EFF,
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

VALUE_KEY_BY_NAME = {
    "counting": "impact",
    "efficiency": "impact_eff",
    "minutes": "minutes",
}


def make_pipe():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )


def pack(df: pd.DataFrame, cols: list[str]):
    return df[cols].astype(float).fillna(0), df["home_win"].astype(int)


def paired_bootstrap_logloss_delta(y, p_a, p_b, label_a="a", label_b="b", n_boot=N_BOOT):
    y = np.asarray(y).astype(int)
    p_a = np.clip(np.asarray(p_a, float), 1e-6, 1 - 1e-6)
    p_b = np.clip(np.asarray(p_b, float), 1e-6, 1 - 1e-6)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        deltas.append(
            log_loss(y[idx], p_a[idx], labels=[0, 1])
            - log_loss(y[idx], p_b[idx], labels=[0, 1])
        )
    deltas = np.asarray(deltas)
    return {
        "delta_mean": float(deltas.mean()),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "n_boot": int(n_boot),
        "definition": f"delta = logloss({label_a}) - logloss({label_b}); negative => {label_a} better",
    }


def main():
    print("=" * 60)
    print("  VAL pick: counting vs efficiency vs minutes impact")
    print("=" * 60)

    matchups_train = pd.read_pickle(os.path.join(DATA_DIR, "matchups_train.pkl"))
    matchups_val = pd.read_pickle(os.path.join(DATA_DIR, "matchups_val.pkl"))
    matchups_test = pd.read_pickle(os.path.join(DATA_DIR, "matchups_test.pkl"))

    if matchups_test["season_id"].astype(str).ne("22025").any():
        raise RuntimeError("matchups_test must be report-only 2025-26 (season_id=22025)")
    if (matchups_train["season_id"].astype(str) > "22024").any() or (
        matchups_val["season_id"].astype(str) > "22024"
    ).any():
        raise RuntimeError("train/val must not include 2025-26+")

    candidates = {
        "counting": MATCHUP_CORE_FEATURES + [ABSENCE_FEATURE_PLAYER],
        "efficiency": MATCHUP_CORE_FEATURES + [ABSENCE_FEATURE_EFF],
        "minutes": MATCHUP_CORE_FEATURES + [ABSENCE_FEATURE_MINUTES],
    }

    val_rows, fitted, val_probs = [], {}, {}
    y_va = matchups_val["home_win"].astype(int).values
    for name, cols in candidates.items():
        pipe = make_pipe()
        pipe.fit(*pack(matchups_train, cols))
        fitted[name] = (pipe, cols)
        p = pipe.predict_proba(pack(matchups_val, cols)[0])[:, 1]
        val_probs[name] = p
        m = evaluate_probs(y_va, p, f"logistic_{name}/val")
        val_rows.append(m)
        print(
            f"  VAL {name}: logloss={m['log_loss']:.4f}  "
            f"brier={m['brier']:.4f}  acc={m['accuracy']:.3f}"
        )

    boot_eff_vs_count = paired_bootstrap_logloss_delta(
        y_va, val_probs["efficiency"], val_probs["counting"], "efficiency", "counting"
    )
    boot_count_vs_min = paired_bootstrap_logloss_delta(
        y_va, val_probs["counting"], val_probs["minutes"], "counting", "minutes"
    )
    print(
        f"\n  Bootstrap Δll(eff-counting): {boot_eff_vs_count['delta_mean']:.4f} "
        f"[{boot_eff_vs_count['ci95_low']:.4f}, {boot_eff_vs_count['ci95_high']:.4f}]"
    )

    best_name = min(
        candidates.keys(),
        key=lambda n: next(r["log_loss"] for r in val_rows if n in r["model"]),
    )
    best_cols = candidates[best_name]
    print(f"  >>> Selected on VAL log loss: {best_name}")

    # Calibration on VAL
    X_tr, y_tr = pack(matchups_train, best_cols)
    X_va, y_va_s = pack(matchups_val, best_cols)
    base = make_pipe()
    base.fit(X_tr, y_tr)
    calib_results = []
    raw_va = base.predict_proba(X_va)[:, 1]
    calib_results.append(
        {"method": "none", **{k: evaluate_probs(y_va_s, raw_va, "x")[k]
         for k in ("log_loss", "brier", "accuracy", "auc")}}
    )
    for method in ("isotonic", "sigmoid"):
        cal = CalibratedClassifierCV(make_pipe(), method=method, cv=3)
        cal.fit(X_tr, y_tr)
        m = evaluate_probs(y_va_s, cal.predict_proba(X_va)[:, 1], "x")
        calib_results.append(
            {"method": method, "log_loss": m["log_loss"], "brier": m["brier"],
             "accuracy": m["accuracy"], "auc": m["auc"]}
        )
        print(f"  VAL calib[{method}]: logloss={m['log_loss']:.4f}  brier={m['brier']:.4f}")

    best_calib = min(calib_results, key=lambda r: r["log_loss"])["method"]
    print(f"  >>> Calibration: {best_calib}")

    train_val = pd.concat([matchups_train, matchups_val], ignore_index=True)
    X_tv, y_tv = pack(train_val, best_cols)
    if best_calib == "none":
        deploy = make_pipe()
        deploy.fit(X_tv, y_tv)
    else:
        deploy = CalibratedClassifierCV(make_pipe(), method=best_calib, cv=3)
        deploy.fit(X_tv, y_tv)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(deploy, os.path.join(MODELS_DIR, "logistic_baseline.pkl"))
    with open(os.path.join(MODELS_DIR, "logistic_feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "absence_feature": best_name,
                "value_key": VALUE_KEY_BY_NAME[best_name],
                "calibration": best_calib,
                "columns": best_cols,
                "counting_formula": "PTS + 1.5*AST + REB + STL + BLK - TOV",
                "efficiency_formula": (
                    "PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*OREB + 0.3*DREB "
                    "+ STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV"
                ),
                "replacement_adjustment": "skipped_no_reliable_position_data",
            },
            f,
            indent=2,
        )

    try:
        if hasattr(deploy, "calibrated_classifiers_"):
            coef = np.mean(
                [c.estimator.named_steps["clf"].coef_.ravel()
                 for c in deploy.calibrated_classifiers_],
                axis=0,
            )
        else:
            coef = deploy.named_steps["clf"].coef_.ravel()
        coefs = dict(zip(best_cols, coef.tolist()))
    except Exception:
        coefs = {}
    with open(os.path.join(MODELS_DIR, "logistic_coefficients.json"), "w", encoding="utf-8") as f:
        json.dump(coefs, f, indent=2)

    X_te, y_te = pack(matchups_test, best_cols)
    log_te = deploy.predict_proba(X_te)[:, 1]
    elo_te = elo_win_prob(matchups_test["elo_diff"].values)

    report_rows = [
        evaluate_probs(y_te, elo_te, "elo_only/test_2025-26"),
        evaluate_probs(y_te, log_te, f"logistic_{best_name}_{best_calib}/test_2025-26"),
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
                f"logistic_{name}/test_2025-26_not_selected",
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
        {"Elo only": elo_te, f"Logistic {best_name}+{best_calib}": log_te},
        os.path.join(DATA_DIR, "calibration_holdout.png"),
    )

    out = matchups_test[
        ["Game_ID", "game_date", "home_team", "away_team", "home_win", "elo_diff"]
    ].copy()
    out["elo_prob"] = elo_te
    out["logistic_prob"] = log_te
    out.to_csv(os.path.join(DATA_DIR, "holdout_baseline_preds.csv"), index=False)

    pipe_src = inspect.getsource(make_pipe)
    selection = {
        "selection_split": "validation",
        "selection_metric": "log_loss",
        "selected_absence_feature": best_name,
        "value_key": VALUE_KEY_BY_NAME[best_name],
        "selected_columns": best_cols,
        "calibration_selected": best_calib,
        "calibration_val": calib_results,
        "bootstrap_val_logloss_delta_efficiency_minus_counting": boot_eff_vs_count,
        "bootstrap_val_logloss_delta_counting_minus_minutes": boot_count_vs_min,
        "validation_metrics": val_rows,
        "train_season_max": "2024-25",
        "report_test_season": "2025-26",
        "live_season": "2026-27",
        "selection_note": (
            "Absence feature + calibration chosen on VAL only; "
            "2025-26 is report-only and was not used for selection."
        ),
        "counting_formula": "PTS + 1.5*AST + REB + STL + BLK - TOV",
        "efficiency_formula": (
            "PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*OREB + 0.3*DREB "
            "+ STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV"
        ),
        "replacement_adjustment": "skipped_no_reliable_position_data",
        "guards": {
            "class_weight_kwarg_present": "class_weight=" in pipe_src,
            "home_court_in_core_features": "home_court" in MATCHUP_CORE_FEATURES,
        },
    }
    with open(os.path.join(DATA_DIR, "feature_selection.json"), "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    te = report_rows[1]
    with open(os.path.join(DATA_DIR, "model_selection.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"best_model=logistic_{best_name}_{best_calib}\n"
            f"value_key={VALUE_KEY_BY_NAME[best_name]}\n"
            f"test_log_loss={te['log_loss']:.6f}\n"
            f"test_brier={te['brier']:.6f}\n"
            f"test_accuracy={te['accuracy']:.6f}\n"
            f"boot_eff_minus_counting={boot_eff_vs_count['delta_mean']:.6f} "
            f"[{boot_eff_vs_count['ci95_low']:.6f}, {boot_eff_vs_count['ci95_high']:.6f}]\n"
            f"report_test_season=2025-26\n"
            f"live_season=2026-27\n"
        )

    print("\nReport test (2025-26):")
    for r in report_rows:
        if "test_2025-26" in r["model"]:
            print(
                f"  {r['model']}: acc={r['accuracy']:.3f}  "
                f"logloss={r['log_loss']:.4f}  brier={r['brier']:.4f}  auc={r['auc']:.3f}"
            )


if __name__ == "__main__":
    main()
