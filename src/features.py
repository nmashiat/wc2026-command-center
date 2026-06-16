"""
features.py
-----------
Elo rating engine + leakage-free chronological feature builder for
international football match outcome prediction.

The core idea: walk every match in date order exactly once. For each match we
first READ the current state of both teams (their Elo, recent form, rolling
goals, days of rest) to produce a feature row, and only THEN update that state
with the actual result. Because features are always computed from information
available strictly *before* kickoff, there is no target leakage by construction.
"""

from collections import deque
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Elo configuration (World-Football-Elo style)
# ----------------------------------------------------------------------------
START_ELO = 1500.0
HOME_ADV = 65.0          # Elo points added to the home side's expectation (non-neutral)
FORM_WINDOW = 5          # matches used for recent-form points average
GOALS_WINDOW = 10        # matches used for rolling goals for / against

FEATURES = [
    "elo_diff", "home_elo", "away_elo", "neutral", "is_friendly", "is_competitive",
    "form_pts_diff", "home_form", "away_form",
    "gf_diff", "ga_diff", "home_gf", "home_ga", "away_gf", "away_ga",
    "rest_diff", "home_rest", "away_rest", "home_logn", "away_logn",
]


def k_weight(tournament: str) -> float:
    """Match importance -> base Elo K factor."""
    t = str(tournament).lower()
    if "world cup" in t and "qualif" not in t:
        return 60.0
    if "confederations" in t:
        return 50.0
    if any(x in t for x in ("euro", "copa", "african", "asian cup", "gold cup", "nations league")) \
            and "qualif" not in t:
        return 50.0
    if "qualif" in t:
        return 40.0
    if "friendly" in t:
        return 20.0
    return 30.0


def _goal_mult(margin: int) -> float:
    """Goal-difference multiplier so blowouts move Elo more."""
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8.0


def _expected(home_elo: float, away_elo: float, neutral: bool) -> float:
    ha = 0.0 if neutral else HOME_ADV
    return 1.0 / (1.0 + 10 ** ((away_elo - home_elo - ha) / 400.0))


def new_state() -> dict:
    """Container for all rolling per-team information."""
    return dict(elo={}, last_date={}, form={}, gf={}, ga={}, n={})


def _ctx_flags(tournament: str):
    t = str(tournament).lower()
    is_friendly = int("friendly" in t)
    is_competitive = int(any(x in t for x in (
        "qualif", "world cup", "euro", "copa", "nations league",
        "asian cup", "african", "gold cup")))
    return is_friendly, is_competitive


def row_features(state: dict, home: str, away: str, neutral: bool,
                 tournament: str, date) -> dict:
    """Build a single feature row from the CURRENT state (no update)."""
    he = state["elo"].get(home, START_ELO)
    ae = state["elo"].get(away, START_ELO)
    hf = np.mean(state["form"][home]) if state["form"].get(home) else np.nan
    af = np.mean(state["form"][away]) if state["form"].get(away) else np.nan
    hgf = np.mean(state["gf"][home]) if state["gf"].get(home) else np.nan
    hga = np.mean(state["ga"][home]) if state["ga"].get(home) else np.nan
    agf = np.mean(state["gf"][away]) if state["gf"].get(away) else np.nan
    aga = np.mean(state["ga"][away]) if state["ga"].get(away) else np.nan
    hrest = (date - state["last_date"][home]).days if home in state["last_date"] else np.nan
    arest = (date - state["last_date"][away]).days if away in state["last_date"] else np.nan
    is_friendly, is_competitive = _ctx_flags(tournament)

    sub = lambda x, y: (x - y) if not (np.isnan(x) or np.isnan(y)) else np.nan
    return dict(
        neutral=int(neutral), is_friendly=is_friendly, is_competitive=is_competitive,
        elo_diff=he - ae, home_elo=he, away_elo=ae,
        form_pts_diff=sub(hf, af), home_form=hf, away_form=af,
        gf_diff=sub(hgf, agf), ga_diff=sub(hga, aga),
        home_gf=hgf, home_ga=hga, away_gf=agf, away_ga=aga,
        rest_diff=sub(hrest, arest), home_rest=hrest, away_rest=arest,
        home_logn=np.log1p(state["n"].get(home, 0)),
        away_logn=np.log1p(state["n"].get(away, 0)),
    )


def update_state(state: dict, home: str, away: str, hs: int, as_: int,
                 neutral: bool, tournament: str, date) -> None:
    """Apply one match result to the rolling state (Elo, form, goals, rest)."""
    he = state["elo"].get(home, START_ELO)
    ae = state["elo"].get(away, START_ELO)
    margin = abs(hs - as_)
    s = 1.0 if hs > as_ else 0.0 if hs < as_ else 0.5
    e = _expected(he, ae, neutral)
    k = k_weight(tournament) * _goal_mult(margin)
    state["elo"][home] = he + k * (s - e)
    state["elo"][away] = ae + k * ((1 - s) - (1 - e))

    hp = 3 if s == 1 else 1 if s == 0.5 else 0
    ap = 3 if s == 0 else 1 if s == 0.5 else 0
    state["form"].setdefault(home, deque(maxlen=FORM_WINDOW)).append(hp)
    state["form"].setdefault(away, deque(maxlen=FORM_WINDOW)).append(ap)
    state["gf"].setdefault(home, deque(maxlen=GOALS_WINDOW)).append(hs)
    state["ga"].setdefault(home, deque(maxlen=GOALS_WINDOW)).append(as_)
    state["gf"].setdefault(away, deque(maxlen=GOALS_WINDOW)).append(as_)
    state["ga"].setdefault(away, deque(maxlen=GOALS_WINDOW)).append(hs)
    state["last_date"][home] = date
    state["last_date"][away] = date
    state["n"][home] = state["n"].get(home, 0) + 1
    state["n"][away] = state["n"].get(away, 0) + 1


def build(played: pd.DataFrame):
    """
    Single chronological pass over played matches.
    Returns (features_df, final_state). The final_state holds the up-to-date
    ratings/form for every team and is what predict.py uses for future fixtures.
    """
    played = played.sort_values("date").reset_index(drop=True)
    state = new_state()
    rows = []
    for r in played.itertuples(index=False):
        feat = row_features(state, r.home_team, r.away_team, bool(r.neutral),
                            r.tournament, r.date)
        feat["date"] = r.date
        feat["target"] = ("home_win" if r.home_score > r.away_score
                          else "away_win" if r.home_score < r.away_score else "draw")
        rows.append(feat)
        update_state(state, r.home_team, r.away_team, int(r.home_score),
                     int(r.away_score), bool(r.neutral), r.tournament, r.date)
    return pd.DataFrame(rows), state
