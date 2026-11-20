"""Holdout evaluation metrics for win-probability models."""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def evaluate_probs(y_true, y_prob, name: str) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    y_hat = (y_prob >= 0.5).astype(int)
    out = {
        "model": name,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "base_rate": float(y_true.mean()),
    }
    return out


def elo_win_prob(elo_diff: np.ndarray, home_advantage: float = 100.0) -> np.ndarray:
    """Standard Elo expected score with optional home-advantage Elo points."""
    adj = np.asarray(elo_diff, dtype=float) + home_advantage
    return 1.0 / (1.0 + 10.0 ** (-adj / 400.0))


def save_calibration_plot(y_true, prob_dict: dict, out_path: str, n_bins: int = 10):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    for name, probs in prob_dict.items():
        frac_pos, mean_pred = calibration_curve(
            y_true, np.clip(probs, 1e-6, 1 - 1e-6), n_bins=n_bins, strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.set_xlabel("Predicted home-win probability")
    ax.set_ylabel("Observed home-win rate")
    ax.set_title("Calibration (2024-25 holdout)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_metrics_table(rows: list[dict], out_json: str, out_md: str):
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    headers = ["model", "n", "accuracy", "log_loss", "brier", "auc", "base_rate"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["model"]),
                    str(r["n"]),
                    f"{r['accuracy']:.3f}",
                    f"{r['log_loss']:.4f}",
                    f"{r['brier']:.4f}",
                    f"{r['auc']:.3f}",
                    f"{r['base_rate']:.3f}",
                ]
            )
            + " |"
        )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
