"""
Absence / player-deficit helpers.

The single public deficit calculator used by training and serving is
`nba_ml.features.compute_team_absence_deficit` (re-exported here for tests).
"""

from __future__ import annotations

import unicodedata
from typing import Iterable

import numpy as np

PRIOR_STRENGTH = 10  # w = gp / (gp + 10)
ROTATION_MIN_THRESHOLD = 10.0
MAX_CONSECUTIVE_MISSES = 10  # miss 10+ straight → stop counting


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return "".join(
        c
        for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    ).lower().strip()


def is_exact_out_status(status: str) -> bool:
    """Serving rule: Status must be exactly 'Out' (after strip)."""
    return str(status).strip() == "Out"


def canonicalize_injury_status(raw: str) -> str:
    """
    Map messy feed strings to a canonical label.
    Only 'Out' counts for deficit; Q/D/P are preserved as themselves.
    """
    s = str(raw).strip()
    if s == "Out":
        return "Out"
    low = s.lower()
    if any(
        x in low
        for x in ("questionable", "doubtful", "probable", "available", "game time", "gtd")
    ):
        if "questionable" in low:
            return "Questionable"
        if "doubtful" in low:
            return "Doubtful"
        if "probable" in low or "available" in low:
            return "Probable"
        return "Questionable"
    if "out" in low:
        return "Out"
    return s


def blend_prior(gp_current: int, curr_avg: float, prev_avg: float | None) -> float:
    """weight_current = gp / (gp + 10). No fixed 70/30."""
    n = max(int(gp_current), 0)
    w = n / (n + PRIOR_STRENGTH)
    if prev_avg is None or (isinstance(prev_avg, float) and np.isnan(prev_avg)):
        return float(curr_avg) if n > 0 else 0.0
    if n <= 0:
        return float(prev_avg)
    return float(w * curr_avg + (1.0 - w) * prev_avg)


def impact_from_avgs(pts: float, reb: float, ast: float) -> float:
    return float(pts) + 0.5 * float(reb) + 0.5 * float(ast)


def season_id_to_prev(season_id: str) -> str:
    s = str(season_id)
    if len(s) == 5 and s[0] == "2" and s[1:].isdigit():
        return f"2{int(s[1:]) - 1:04d}"
    return s
