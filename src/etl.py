"""
etl.py — data collection -> cleaning -> SQL -> KPI marts for the BI section.

Steps:
  1. Load the raw international results, clean types, derive outcome/goals.
  2. Tag the 48 World Cup 2026 teams with their group + confederation.
  3. Load everything into a SQLite star schema (schema.sql).
  4. Run the analytical queries in queries.sql and export each as a KPI table.
  5. Compute per-team KPIs (current Elo + last-10 form) and write bi.json,
     the data blob the web Command Center reads.

Run: python src/etl.py
"""

import json
import os
import re
import sqlite3

import numpy as np
import pandas as pd

import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "results.csv")
SQLDIR = os.path.join(ROOT, "sql")
OUT = os.path.join(ROOT, "outputs")
DB = os.path.join(OUT, "wc2026.db")
os.makedirs(OUT, exist_ok=True)

# ---- WC2026 groups + confederations (final draw, Dec 2025) ----
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia & Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
CONF = {  # confederation per team
    **{t: "CONCACAF" for t in ["Mexico", "Canada", "United States", "Haiti", "Curaçao", "Panama"]},
    **{t: "CONMEBOL" for t in ["Brazil", "Paraguay", "Ecuador", "Uruguay", "Argentina", "Colombia"]},
    **{t: "UEFA" for t in ["Czechia", "Bosnia & Herzegovina", "Switzerland", "Scotland", "Türkiye",
                           "Germany", "Netherlands", "Sweden", "Belgium", "Spain", "France",
                           "Norway", "Austria", "Portugal", "England", "Croatia"]},
    **{t: "CAF" for t in ["South Africa", "Morocco", "Ivory Coast", "Tunisia", "Egypt", "Cape Verde",
                          "Senegal", "Algeria", "DR Congo", "Ghana"]},
    **{t: "AFC" for t in ["South Korea", "Qatar", "Australia", "Japan", "Iran", "Saudi Arabia",
                          "Iraq", "Jordan", "Uzbekistan"]},
    "New Zealand": "OFC",
}
TEAM_GROUP = {t: g for g, ts in GROUPS.items() for t in ts}
# alias map: dataset spelling -> our canonical spelling
ALIAS = {"Turkey": "Türkiye", "Bosnia and Herzegovina": "Bosnia & Herzegovina",
         "Korea Republic": "South Korea", "Czech Republic": "Czechia"}


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["date"])
    df["home_team"] = df.home_team.replace(ALIAS)
    df["away_team"] = df.away_team.replace(ALIAS)
    df["neutral"] = df.neutral.astype(str).str.upper().eq("TRUE").astype(int)
    played = df.home_score.notna()
    df["outcome"] = np.where(~played, None,
                      np.where(df.home_score > df.away_score, "home_win",
                      np.where(df.home_score < df.away_score, "away_win", "draw")))
    df["total_goals"] = (df.home_score + df.away_score)
    df["is_wc2026"] = ((df["tournament"].astype(str).str.strip() == "FIFA World Cup") &
                       (df["date"] >= pd.Timestamp("2026-06-11"))).astype(int)
    df = df.reset_index(drop=True)
    df.insert(0, "match_id", df.index + 1)
    return df


def build_db(df: pd.DataFrame):
    con = sqlite3.connect(DB)
    with open(os.path.join(SQLDIR, "schema.sql")) as f:
        con.executescript(f.read())
    dim = pd.DataFrame([{"team": t, "grp": g, "confederation": CONF.get(t, "?")}
                        for g, ts in GROUPS.items() for t in ts])
    dim.to_sql("dim_team", con, if_exists="append", index=False)
    cols = ["match_id", "date", "home_team", "away_team", "home_score", "away_score",
            "tournament", "city", "country", "neutral", "is_wc2026", "outcome", "total_goals"]
    out = df[cols].copy()
    out["date"] = out.date.dt.strftime("%Y-%m-%d")
    out.to_sql("fact_match", con, if_exists="append", index=False)
    con.commit()
    return con


def run_named_queries(con) -> dict:
    sql = open(os.path.join(SQLDIR, "queries.sql")).read()
    blocks = re.split(r"--\s*@name:\s*(\w+)", sql)[1:]  # [name, body, name, body, ...]
    results = {}
    for name, body in zip(blocks[::2], blocks[1::2]):
        stmt = body.strip().rstrip(";")
        results[name] = pd.read_sql_query(stmt, con)
    return results


def team_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Per-team KPIs: current Elo + last-10 international form."""
    played = df.dropna(subset=["home_score"]).copy()
    played["home_score"] = played.home_score.astype(int)
    played["away_score"] = played.away_score.astype(int)
    _, state = F.build(played)  # final Elo/form for every team

    teams = TEAM_GROUP.keys()
    rows = []
    for t in teams:
        last = played[(played.home_team == t) | (played.away_team == t)].tail(10)
        w = d = l = gf = ga = 0
        for r in last.itertuples(index=False):
            home = r.home_team == t
            tf, ta = (r.home_score, r.away_score) if home else (r.away_score, r.home_score)
            gf += tf; ga += ta
            if tf > ta: w += 1
            elif tf == ta: d += 1
            else: l += 1
        n = max(len(last), 1)
        rows.append(dict(
            team=t, grp=TEAM_GROUP[t], confederation=CONF.get(t, "?"),
            elo=round(state["elo"].get(t, 1500), 0),
            form_pts=round((3 * w + d) / n, 2),
            l10=f"{w}-{d}-{l}", gf10=gf, ga10=ga, gd10=gf - ga,
            avg_gf=round(gf / n, 2), avg_ga=round(ga / n, 2),
        ))
    k = pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)
    k.insert(0, "elo_rank", k.index + 1)
    return k


def main():
    df = load_clean()
    print(f"Cleaned {len(df):,} matches | WC2026 matches tagged: {int(df.is_wc2026.sum())} "
          f"({int((df.is_wc2026 == 1).mul(df.home_score.notna()).sum())} played so far)")
    con = build_db(df)
    q = run_named_queries(con)
    kpis = team_kpis(df)

    # export CSVs
    for name, frame in q.items():
        frame.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    kpis.to_csv(os.path.join(OUT, "team_kpis.csv"), index=False)

    summary = q["tournament_summary"].iloc[0].to_dict()
    print("Tournament so far:", {k: summary[k] for k in
          ["matches_played", "total_goals", "avg_goals_per_match"]})

    bi = dict(
        summary=summary,
        standings=q["standings"].to_dict(orient="records"),
        goals_by_day=q["goals_by_day"].to_dict(orient="records"),
        biggest_wins=q["biggest_wins"].to_dict(orient="records"),
        team_kpis=kpis.to_dict(orient="records"),
    )
    with open(os.path.join(OUT, "bi.json"), "w") as f:
        json.dump(bi, f, indent=2, default=str)
    con.close()
    print("Wrote wc2026.db, KPI CSVs, team_kpis.csv, bi.json")


if __name__ == "__main__":
    main()
