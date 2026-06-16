-- queries.sql — analytical queries that power the BI section.
-- Each named query is executed by etl.py and exported as a KPI table.

-- @name: standings
-- WC2026 group standings built from played group/knockout matches.
WITH m AS (
    SELECT * FROM fact_match WHERE is_wc2026 = 1 AND home_score IS NOT NULL
),
team_rows AS (
    SELECT home_team AS team, home_score AS gf, away_score AS ga,
           CASE WHEN home_score > away_score THEN 3
                WHEN home_score = away_score THEN 1 ELSE 0 END AS pts
    FROM m
    UNION ALL
    SELECT away_team AS team, away_score AS gf, home_score AS ga,
           CASE WHEN away_score > home_score THEN 3
                WHEN away_score = home_score THEN 1 ELSE 0 END AS pts
    FROM m
)
SELECT d.grp AS grp, t.team AS team,
       COUNT(*) AS played,
       SUM(CASE WHEN pts = 3 THEN 1 ELSE 0 END) AS won,
       SUM(CASE WHEN pts = 1 THEN 1 ELSE 0 END) AS drawn,
       SUM(CASE WHEN pts = 0 THEN 1 ELSE 0 END) AS lost,
       SUM(t.gf) AS gf, SUM(t.ga) AS ga, SUM(t.gf) - SUM(t.ga) AS gd,
       SUM(pts) AS points
FROM team_rows t
JOIN dim_team d ON d.team = t.team
GROUP BY d.grp, t.team
ORDER BY d.grp, points DESC, gd DESC, gf DESC;

-- @name: tournament_summary
-- Headline KPIs for the tournament so far.
SELECT
    COUNT(*) AS matches_played,
    SUM(total_goals) AS total_goals,
    ROUND(AVG(total_goals), 2) AS avg_goals_per_match,
    SUM(CASE WHEN outcome = 'home_win' THEN 1 ELSE 0 END) AS home_wins,
    SUM(CASE WHEN outcome = 'draw' THEN 1 ELSE 0 END) AS draws,
    SUM(CASE WHEN outcome = 'away_win' THEN 1 ELSE 0 END) AS away_wins,
    SUM(CASE WHEN home_score = 0 OR away_score = 0 THEN 1 ELSE 0 END) AS clean_sheets
FROM fact_match
WHERE is_wc2026 = 1 AND home_score IS NOT NULL;

-- @name: goals_by_day
-- Goal volume per matchday (for the trend line).
SELECT date, COUNT(*) AS matches, SUM(total_goals) AS goals
FROM fact_match
WHERE is_wc2026 = 1 AND home_score IS NOT NULL
GROUP BY date ORDER BY date;

-- @name: biggest_wins
-- Most dominant results so far.
SELECT date, home_team, away_team, home_score, away_score,
       ABS(home_score - away_score) AS margin
FROM fact_match
WHERE is_wc2026 = 1 AND home_score IS NOT NULL
ORDER BY margin DESC, total_goals DESC
LIMIT 10;
