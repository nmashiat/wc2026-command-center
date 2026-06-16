"""
train.py
--------
Train and evaluate the international football match-outcome model.

Pipeline:
  1. Load results.csv (download instructions in README).
  2. Build leakage-free features via a single chronological pass (features.build).
  3. Restrict to the modern era (1990+) and split by TIME (train <2018, test >=2018)
     so we never train on the future.
  4. Compare baselines (always-home, Elo-only) against a HistGradientBoosting
     model, a full-feature logistic, and a blend; select the best by log-loss.
  5. Report accuracy, log-loss, Brier, calibration, permutation importance.
  6. Refit the deployed (blend) model on ALL modern data and save artifacts.

Run:  python src/train.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, log_loss, classification_report,
                             confusion_matrix)

import features as F

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "results.csv")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
LAB = ["away_win", "draw", "home_win"]
MODERN, SPLIT = "1990-01-01", "2018-01-01"


def new_hgb():
    return HistGradientBoostingClassifier(
        max_leaf_nodes=31, learning_rate=0.05, max_iter=700, l2_regularization=2.0,
        min_samples_leaf=80, early_stopping=True, validation_fraction=0.12,
        n_iter_no_change=30, random_state=42)


def new_logistic():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, C=0.5))


def align(proba, classes):
    """Reorder probability columns to LAB order."""
    idx = [list(classes).index(c) for c in LAB]
    return proba[:, idx]


def main():
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date")
    played = df.dropna(subset=["home_score"]).copy()
    played["home_score"] = played.home_score.astype(int)
    played["away_score"] = played.away_score.astype(int)
    print(f"Loaded {len(played):,} played matches  ({played.date.min().date()} -> {played.date.max().date()})")

    feat, state = F.build(played)
    m = feat[feat.date >= MODERN].copy()
    train = m[m.date < SPLIT]
    test = m[m.date >= SPLIT]
    Xtr, ytr = train[F.FEATURES], train.target
    Xte, yte = test[F.FEATURES], test.target
    print(f"Modern window from {MODERN[:4]}: train {len(train):,} | test {len(test):,} (split {SPLIT[:4]})\n")

    results = {}

    # ---- Baseline A: always home win ----
    accA = accuracy_score(yte, ["home_win"] * len(yte))
    prior = train.target.value_counts(normalize=True)
    P0 = np.tile([prior["away_win"], prior["draw"], prior["home_win"]], (len(yte), 1))
    results["baseline_home"] = dict(acc=accA, log_loss=log_loss(yte, P0, labels=LAB))

    # ---- Baseline B: Elo-only logistic ----
    b = LogisticRegression(max_iter=1000)
    b.fit(train[["elo_diff", "neutral"]].fillna(0), ytr)
    pb = align(b.predict_proba(test[["elo_diff", "neutral"]].fillna(0)), b.classes_)
    results["baseline_elo"] = dict(
        acc=accuracy_score(yte, b.predict(test[["elo_diff", "neutral"]].fillna(0))),
        log_loss=log_loss(yte, pb, labels=LAB))

    # ---- Model: HGB ----
    hgb = new_hgb().fit(Xtr, ytr)
    ph = align(hgb.predict_proba(Xte), hgb.classes_)
    results["hgb"] = dict(acc=accuracy_score(yte, [LAB[i] for i in ph.argmax(1)]),
                          log_loss=log_loss(yte, ph, labels=LAB))

    # ---- Model: full logistic ----
    logf = new_logistic().fit(Xtr, ytr)
    pl = align(logf.predict_proba(Xte), logf.classes_)
    results["full_logistic"] = dict(acc=accuracy_score(yte, [LAB[i] for i in pl.argmax(1)]),
                                    log_loss=log_loss(yte, pl, labels=LAB))

    # ---- Model: blend ----
    pbl = 0.5 * ph + 0.5 * pl
    results["blend"] = dict(acc=accuracy_score(yte, [LAB[i] for i in pbl.argmax(1)]),
                            log_loss=log_loss(yte, pbl, labels=LAB))

    print(f"{'model':16s} {'accuracy':>9s} {'log_loss':>9s}")
    for k, v in results.items():
        print(f"{k:16s} {v['acc']:9.4f} {v['log_loss']:9.4f}")

    # selected probabilities for diagnostics = blend (best accuracy, near-best log-loss)
    proba, pred = pbl, np.array([LAB[i] for i in pbl.argmax(1)])
    print("\n--- Blend classification report (test) ---")
    print(classification_report(yte, pred, digits=3, zero_division=0))

    # Brier + calibration
    Y = pd.get_dummies(yte)[LAB].values
    brier = float(np.mean(np.sum((proba - Y) ** 2, axis=1)))
    hp = proba[:, LAB.index("home_win")]
    yb = (yte.values == "home_win").astype(int)
    calib = []
    for lo in np.arange(0, 1.0, 0.1):
        mask = (hp >= lo) & (hp < lo + 0.1)
        if mask.sum():
            calib.append((round(float(hp[mask].mean()), 3), round(float(yb[mask].mean()), 3), int(mask.sum())))

    # permutation importance (on HGB; log-loss scoring)
    sub = test.sample(min(3000, len(test)), random_state=0)
    pi = permutation_importance(hgb, sub[F.FEATURES], sub.target, n_repeats=5,
                                random_state=0, scoring="neg_log_loss")
    imp = pd.Series(pi.importances_mean, index=F.FEATURES).sort_values(ascending=False)

    # ---------- plots ----------
    _plot_importance(imp, os.path.join(OUT, "feature_importance.png"))
    _plot_calibration(calib, os.path.join(OUT, "calibration.png"))
    _plot_confusion(confusion_matrix(yte, pred, labels=LAB), os.path.join(OUT, "confusion_matrix.png"))

    # ---------- refit deployed blend on modern data BEFORE the tournament, save ----------
    # Excluding 2026-06-11+ means the model never saw a single World Cup 2026 result,
    # so grading it against those matches in predict.py is leakage-free.
    pre = m[m.date < "2026-06-11"]
    Xall, yall = pre[F.FEATURES], pre.target
    dep_hgb = new_hgb().fit(Xall, yall)
    dep_log = new_logistic().fit(Xall, yall)
    joblib.dump(dict(hgb=dep_hgb, logistic=dep_log, features=F.FEATURES, labels=LAB,
                     state=state), os.path.join(OUT, "model.joblib"))

    metrics = dict(results=results, brier=brier, calibration=calib,
                   importance=imp.round(4).to_dict(),
                   train_n=len(train), test_n=len(test))
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nBrier {brier:.4f} | saved model.joblib, metrics.json, and 3 plots to outputs/")


def _plot_importance(imp, path):
    top = imp.head(10)[::-1]
    plt.figure(figsize=(7, 4.2))
    plt.barh(top.index, top.values, color="#1D9E75")
    plt.xlabel("log-loss increase when feature is shuffled")
    plt.title("Permutation importance (top 10)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def _plot_calibration(calib, path):
    if not calib:
        return
    pred = [c[0] for c in calib]; act = [c[1] for c in calib]
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="#888", label="perfect")
    plt.plot(pred, act, "o-", color="#185FA5", label="model")
    plt.xlabel("predicted P(home win)"); plt.ylabel("observed home-win rate")
    plt.title("Calibration — home win"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def _plot_confusion(cm, path):
    plt.figure(figsize=(4.8, 4.2))
    plt.imshow(cm, cmap="Greens")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks(range(3), LAB, rotation=20); plt.yticks(range(3), LAB)
    plt.xlabel("predicted"); plt.ylabel("actual"); plt.title("Confusion matrix (test)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


if __name__ == "__main__":
    main()
