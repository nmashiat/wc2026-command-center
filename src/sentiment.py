"""
sentiment.py — NLP section of the Command Center.

A dependency-free, VADER-style sentiment analyzer (lexicon + negation +
intensifiers) applied to fan commentary, aggregated to per-team KPIs:
mean sentiment, % positive / neutral / negative, and sample comments.

IMPORTANT — the fan corpus here is ILLUSTRATIVE (synthetically generated).
Its tone per team is biased by that team's real recent form and Elo so the
section tells a coherent story alongside the BI and ML data, but it is NOT
scraped social media. To use real data, replace `build_corpus()` with a loader
for your own export (a CSV of `team,text` rows from the X/Twitter or Reddit API)
and the rest of the pipeline runs unchanged.

Run: python src/sentiment.py
"""

import json
import math
import os
import random
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")

# ---------------------------------------------------------------------------
# 1. Sentiment lexicon (football-flavored valences, -3..+3)
# ---------------------------------------------------------------------------
LEX = {
    # positive
    "brilliant": 3, "magnificent": 3, "unstoppable": 3, "masterclass": 3, "superb": 3,
    "world-class": 3, "ruthless": 2, "clinical": 2, "electric": 2, "solid": 1.5,
    "composed": 1.5, "deserved": 2, "proud": 2, "believe": 1.5, "dream": 2, "hero": 2.5,
    "joy": 2, "love": 2.5, "classy": 2, "hope": 1.5, "quality": 1.5, "dominant": 2.5,
    "stunning": 3, "wonderful": 3, "incredible": 3, "confident": 1.5, "fearless": 2,
    "resilient": 2, "promising": 1.5, "win": 1.5, "victory": 2, "delighted": 2.5,
    # negative
    "shambles": -3, "disgrace": -3, "embarrassing": -3, "woeful": -3, "bottled": -2.5,
    "choke": -2.5, "awful": -3, "terrible": -3, "disaster": -3, "gutted": -2,
    "robbed": -2, "clueless": -2.5, "fragile": -2, "nervy": -1.5, "overrated": -2,
    "defeat": -1.5, "crisis": -2.5, "fear": -1.5, "worried": -1.5, "poor": -2,
    "lazy": -2, "pathetic": -3, "shocking": -2.5, "lifeless": -2.5, "weak": -1.5,
    "frustrating": -1.5, "anxious": -1.5, "hopeless": -2.5, "shameful": -3, "loss": -1.5,
}
INTENSIFIERS = {"very": 1.4, "so": 1.3, "really": 1.4, "absolutely": 1.6, "totally": 1.4,
                "utterly": 1.6, "incredibly": 1.5, "completely": 1.4}
NEGATORS = {"not", "no", "never", "nothing", "cant", "cannot", "dont", "isnt", "wasnt", "without"}
ALPHA = 15.0  # VADER normalization constant


def score(text: str) -> float:
    """Return a compound sentiment in [-1, 1]."""
    toks = re.findall(r"[a-z'\-]+", text.lower())
    total = 0.0
    for i, w in enumerate(toks):
        v = LEX.get(w, 0.0)
        if v == 0.0:
            continue
        if i > 0 and toks[i - 1] in INTENSIFIERS:
            v *= INTENSIFIERS[toks[i - 1]]
        if any(toks[j] in NEGATORS for j in range(max(0, i - 3), i)):
            v *= -0.74
        total += v
    return total / math.sqrt(total * total + ALPHA) if total else 0.0


def label(c: float) -> str:
    return "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"


# ---------------------------------------------------------------------------
# 2. Illustrative corpus — tone biased by each team's real form
# ---------------------------------------------------------------------------
POS_TMPL = [
    "Absolutely {pos} from the boys today, so {pos2}",
    "What a performance, {pos} stuff. I {love} this team",
    "We look {pos} and {pos2}, real {dream} run coming",
    "{pos} again! Proud of the lads, genuinely {pos2}",
    "Defense was {pos}, attack {pos2}. We {believe}",
]
NEG_TMPL = [
    "That was {neg}, completely {neg2}. I'm {gut}",
    "Utterly {neg} again, this team is {neg2}",
    "What a {neg} display, we totally {bott} it",
    "{neg} from start to finish, really {neg2} right now",
    "No urgency, no plan, just {neg}. Feeling {gut}",
]
NEU_TMPL = [
    "Decent enough, not {pos} but not {neg} either",
    "Mixed feelings, some good some {neg}, we'll see",
    "An okay result, nothing {pos}, plenty to work on",
    "Hard to read this team, {pos} one half, {neg} the next",
]
POS = ["brilliant", "magnificent", "clinical", "solid", "dominant", "stunning", "composed"]
NEG = ["woeful", "embarrassing", "shambles", "awful", "fragile", "shocking", "poor"]


def fill(t, rng):
    return (t.replace("{pos}", rng.choice(POS)).replace("{pos2}", rng.choice(POS))
             .replace("{neg}", rng.choice(NEG)).replace("{neg2}", rng.choice(NEG))
             .replace("{love}", "love").replace("{dream}", "dream").replace("{believe}", "believe")
             .replace("{gut}", "gutted").replace("{bott}", "bottled"))


def build_corpus(kpis: pd.DataFrame) -> pd.DataFrame:
    """Generate ~50 illustrative comments per team; positivity ~ form & strength."""
    rng = random.Random(2026)
    rows = []
    n_teams = len(kpis)
    for r in kpis.itertuples(index=False):
        # positivity probability from Elo rank (1..48) and recent form points (0..3)
        rank_term = 1 - (r.elo_rank - 1) / n_teams          # 1=top .. ~0 bottom
        form_term = r.form_pts / 3.0
        p_pos = max(0.1, min(0.85, 0.25 + 0.4 * rank_term + 0.3 * form_term - 0.2))
        for _ in range(50):
            u = rng.random()
            if u < p_pos:
                txt = fill(rng.choice(POS_TMPL), rng)
            elif u < p_pos + 0.25:
                txt = fill(rng.choice(NEU_TMPL), rng)
            else:
                txt = fill(rng.choice(NEG_TMPL), rng)
            rows.append(dict(team=r.team, grp=r.grp, text=txt))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Run pipeline + aggregate
# ---------------------------------------------------------------------------
def main():
    kpis = pd.read_csv(os.path.join(OUT, "team_kpis.csv"))
    corpus = build_corpus(kpis)
    corpus["compound"] = corpus.text.map(score)
    corpus["label"] = corpus.compound.map(label)

    agg = []
    for team, g in corpus.groupby("team"):
        n = len(g)
        examples = (g.sort_values("compound", ascending=False).head(1).text.tolist() +
                    g.sort_values("compound").head(1).text.tolist())
        agg.append(dict(
            team=team, grp=g.grp.iloc[0], n_comments=n,
            mean_sentiment=round(float(g.compound.mean()), 3),
            pct_positive=round(100 * (g.label == "positive").mean(), 1),
            pct_neutral=round(100 * (g.label == "neutral").mean(), 1),
            pct_negative=round(100 * (g.label == "negative").mean(), 1),
            sample_positive=examples[0], sample_negative=examples[1],
        ))
    sa = pd.DataFrame(agg).sort_values("mean_sentiment", ascending=False).reset_index(drop=True)
    sa.to_csv(os.path.join(OUT, "sentiment_by_team.csv"), index=False)
    with open(os.path.join(OUT, "nlp.json"), "w") as f:
        json.dump(dict(teams=sa.to_dict(orient="records"),
                       total_comments=int(len(corpus))), f, indent=2)

    print(f"Scored {len(corpus):,} illustrative fan comments across {len(sa)} teams.")
    print("\nMost positive fanbases:")
    for r in sa.head(5).itertuples(index=False):
        print(f"  {r.team:14s} mean {r.mean_sentiment:+.2f}  ({r.pct_positive:.0f}% positive)")
    print("Most negative fanbases:")
    for r in sa.tail(5).itertuples(index=False):
        print(f"  {r.team:14s} mean {r.mean_sentiment:+.2f}  ({r.pct_negative:.0f}% negative)")


if __name__ == "__main__":
    main()
