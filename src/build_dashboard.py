"""
build_dashboard.py — assemble the single data blob the web Command Center reads,
and inject it into command_center.html.

Merges:
  outputs/bi.json   (KPIs, standings, team KPIs, goals)
  outputs/ml.json   (fixture predictions + upset risk)
  outputs/nlp.json  (per-team sentiment)
into one DATA object, then writes a self-contained command_center.html.

Run (after etl.py, predict.py, sentiment.py): python src/build_dashboard.py
"""

import json
import os
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def main():
    bi, ml, nlp = load("bi.json"), load("ml.json"), load("nlp.json")

    # index standings + sentiment by team
    st = {r["team"]: r for r in bi["standings"]}
    se = {r["team"]: r for r in nlp["teams"]}

    teams = []
    for t in bi["team_kpis"]:
        name = t["team"]
        s = st.get(name, {})
        sent = se.get(name, {})
        teams.append({
            **t,
            "played": int(s.get("played", 0)), "won": int(s.get("won", 0)),
            "drawn": int(s.get("drawn", 0)), "lost": int(s.get("lost", 0)),
            "gf_wc": int(s.get("gf", 0)), "ga_wc": int(s.get("ga", 0)),
            "gd_wc": int(s.get("gd", 0)), "points": int(s.get("points", 0)),
            "sentiment": sent.get("mean_sentiment", 0.0),
            "pct_positive": sent.get("pct_positive", 0.0),
            "pct_negative": sent.get("pct_negative", 0.0),
        })

    data = {
        "meta": {"generated": str(date.today()),
                 "fan_comments": nlp.get("total_comments", 0)},
        "summary": bi["summary"],
        "teams": teams,
        "goals_by_day": bi["goals_by_day"],
        "biggest_wins": bi["biggest_wins"],
        "fixtures": ml["fixtures"],
        "results": ml.get("results", []),
        "scorecard": ml.get("scorecard", {"graded": 0}),
        "sentiment": nlp["teams"],
    }

    tpl = os.path.join(HERE, "command_center.template.html")
    with open(tpl) as f:
        html = f.read()
    html = html.replace("/*__DATA__*/null", json.dumps(data))
    dest = os.path.join(OUT, "command_center.html")
    with open(dest, "w") as f:
        f.write(html)
    print(f"Built {dest}  ({len(html):,} bytes, {len(teams)} teams, {len(ml['fixtures'])} fixtures)")


if __name__ == "__main__":
    main()
