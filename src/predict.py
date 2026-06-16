"""
predict.py — ML section of the Command Center (live-aware).

Loads the pre-tournament blend model, then walks every played match in date
order. For each World Cup 2026 match it:
  * predicts the outcome from team state BEFORE kickoff (no leakage),
  * records the actual result and whether the model's pick was correct,
  * flags upsets (the Elo favorite failed to win).
After the played matches it predicts the genuinely UNPLAYED fixtures.

Because matches with a score drop out of "upcoming" automatically, the dashboard
stays current the moment you refresh results.csv and re-run.

Exports:
  outputs/ml.json             {results, fixtures, scorecard}
  outputs/predictions.csv     upcoming fixtures
  outputs/wc2026_results.csv  played matches, graded

Run: python src/predict.py
"""

import os
# Limit BLAS/OpenMP threads BEFORE numpy/sklearn import — prevents thread
# oversubscription that can hang the per-fixture prediction loop on some machines.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import joblib
import numpy as np
import pandas as pd

import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "results.csv")
OUT = os.path.join(ROOT, "outputs")
ART = os.path.join(OUT, "model.joblib")
LAB = ["away_win", "draw", "home_win"]
ALIAS = {"Turkey": "Türkiye", "Bosnia and Herzegovina": "Bosnia & Herzegovina",
         "Korea Republic": "South Korea", "Czech Republic": "Czechia"}

# A 2026 World Cup match = exact tournament name AND on/after the opening day.
# "contains('World Cup')" would wrongly include "FIFA World Cup qualification".
WC_START = pd.Timestamp("2026-06-11")


def WC(tournament, date):
    return str(tournament).strip() == "FIFA World Cup" and date >= WC_START


def _align(p, classes):
    return p[:, [list(classes).index(c) for c in LAB]]


def predict_proba(hgb, logf, X):
    return (0.5 * _align(hgb.predict_proba(X), hgb.classes_) +
            0.5 * _align(logf.predict_proba(X), logf.classes_))[0]


def main():
    art = joblib.load(ART)
    hgb, logf, feats = art["hgb"], art["logistic"], art["features"]

    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date")
    df["home_team"] = df.home_team.replace(ALIAS)
    df["away_team"] = df.away_team.replace(ALIAS)
    played = df.dropna(subset=["home_score"]).copy()
    played["home_score"] = played.home_score.astype(int)
    played["away_score"] = played.away_score.astype(int)

    state = F.new_state()
    results = []
    for r in played.itertuples(index=False):
        if WC(r.tournament, r.date):
            feat = F.row_features(state, r.home_team, r.away_team, bool(r.neutral),
                                  r.tournament, r.date)
            p = predict_proba(hgb, logf, pd.DataFrame([feat])[feats])
            ph, pdr, pa = p[LAB.index("home_win")], p[LAB.index("draw")], p[LAB.index("away_win")]
            actual = ("home_win" if r.home_score > r.away_score
                      else "away_win" if r.home_score < r.away_score else "draw")
            pick = LAB[int(np.argmax(p))]
            he = state["elo"].get(r.home_team, 1500.0)
            ae = state["elo"].get(r.away_team, 1500.0)
            favorite = r.home_team if he >= ae else r.away_team
            fav_won = ((favorite == r.home_team and actual == "home_win") or
                       (favorite == r.away_team and actual == "away_win"))
            results.append(dict(
                date=str(r.date.date()), home=r.home_team, away=r.away_team,
                home_score=int(r.home_score), away_score=int(r.away_score),
                actual=actual, p_home=round(float(ph), 3), p_draw=round(float(pdr), 3),
                p_away=round(float(pa), 3), pick=pick, correct=bool(pick == actual),
                favorite=favorite, upset=bool(not fav_won),
                p_actual=round(float(p[LAB.index(actual)]), 3),
            ))
        F.update_state(state, r.home_team, r.away_team, int(r.home_score),
                       int(r.away_score), bool(r.neutral), r.tournament, r.date)

    fut = df[df.home_score.isna() &
             (df.tournament.astype(str).str.strip() == "FIFA World Cup") &
             (df.date >= WC_START)].copy()
    fixtures = []
    for r in fut.itertuples(index=False):
        feat = F.row_features(state, r.home_team, r.away_team, bool(r.neutral),
                              r.tournament, r.date)
        p = predict_proba(hgb, logf, pd.DataFrame([feat])[feats])
        ph, pdr, pa = p[LAB.index("home_win")], p[LAB.index("draw")], p[LAB.index("away_win")]
        he = state["elo"].get(r.home_team, 1500.0); ae = state["elo"].get(r.away_team, 1500.0)
        favorite, fav_p = (r.home_team, ph) if he >= ae else (r.away_team, pa)
        gap = abs(he - ae)
        fixtures.append(dict(
            date=str(r.date.date()), home=r.home_team, away=r.away_team,
            p_home=round(float(ph), 3), p_draw=round(float(pdr), 3), p_away=round(float(pa), 3),
            most_likely=LAB[int(np.argmax(p))], favorite=favorite,
            favorite_win_prob=round(float(fav_p), 3), elo_gap=int(round(gap)),
            gap_class="clear favorite" if gap >= 100 else "lean" if gap >= 40 else "coin flip",
            upset_risk=round(float(1 - fav_p), 3),
        ))

    scorecard = {"graded": 0}
    if results:
        rs = pd.DataFrame(results)
        from sklearn.metrics import log_loss
        P = rs[["p_away", "p_draw", "p_home"]].values
        P = P / P.sum(axis=1, keepdims=True)  # undo rounding drift so rows sum to 1
        scorecard = dict(
            graded=len(rs), correct=int(rs.correct.sum()),
            accuracy=round(float(rs.correct.mean()), 3),
            log_loss=round(float(log_loss(rs.actual, P, labels=LAB)), 3),
            upsets=int(rs.upset.sum()),
        )

    pd.DataFrame(fixtures).to_csv(os.path.join(OUT, "predictions.csv"), index=False)
    pd.DataFrame(results).to_csv(os.path.join(OUT, "wc2026_results.csv"), index=False)
    with open(os.path.join(OUT, "ml.json"), "w") as f:
        json.dump(dict(results=results, fixtures=fixtures, scorecard=scorecard), f, indent=2)

    print(f"Graded {scorecard['graded']} played WC matches | {len(fixtures)} upcoming fixtures")
    if results:
        print(f"  model called {scorecard['correct']}/{scorecard['graded']} correct "
              f"({scorecard['accuracy']:.0%}), log-loss {scorecard['log_loss']}, "
              f"{scorecard['upsets']} upsets")
        print("\nMost recent results (pick vs actual):")
        for r in results[-6:]:
            mark = "OK " if r["correct"] else "XX "
            print(f"  {mark}{r['date']}  {r['home']} {r['home_score']}-{r['away_score']} {r['away']}"
                  f"  - model picked {r['pick']}")


if __name__ == "__main__":
    main()
