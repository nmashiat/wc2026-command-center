-- schema.sql — FIFA World Cup 2026 analytics data model (SQLite)
-- A simple star layout: one match fact table + a team dimension.

DROP TABLE IF EXISTS fact_match;
DROP TABLE IF EXISTS dim_team;

CREATE TABLE dim_team (
    team          TEXT PRIMARY KEY,
    grp           TEXT,           -- World Cup 2026 group (A..L)
    confederation TEXT
);

CREATE TABLE fact_match (
    match_id    INTEGER PRIMARY KEY,
    date        TEXT,
    home_team   TEXT,
    away_team   TEXT,
    home_score  INTEGER,
    away_score  INTEGER,
    tournament  TEXT,
    city        TEXT,
    country     TEXT,
    neutral     INTEGER,
    is_wc2026   INTEGER,          -- 1 if this is a 2026 World Cup match
    outcome     TEXT,             -- home_win / draw / away_win (NULL if unplayed)
    total_goals INTEGER
);

CREATE INDEX idx_match_wc   ON fact_match(is_wc2026);
CREATE INDEX idx_match_home ON fact_match(home_team);
CREATE INDEX idx_match_away ON fact_match(away_team);
