-- ProjectSoccer Database Schema

-- Historical match results (from football-data.co.uk CSVs)
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league TEXT NOT NULL,
    season TEXT NOT NULL,
    match_date DATE NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    ft_home_goals INTEGER,
    ft_away_goals INTEGER,
    ft_result TEXT,              -- H, D, A
    ht_home_goals INTEGER,
    ht_away_goals INTEGER,
    ht_result TEXT,
    home_shots INTEGER,
    away_shots INTEGER,
    home_shots_target INTEGER,
    away_shots_target INTEGER,
    home_corners INTEGER,
    away_corners INTEGER,
    home_fouls INTEGER,
    away_fouls INTEGER,
    home_yellows INTEGER,
    away_yellows INTEGER,
    home_reds INTEGER,
    away_reds INTEGER,
    referee TEXT,
    -- Historical bookmaker odds (from CSV)
    b365_home REAL,
    b365_draw REAL,
    b365_away REAL,
    pinnacle_home REAL,
    pinnacle_draw REAL,
    pinnacle_away REAL,
    max_home REAL,
    max_draw REAL,
    max_away REAL,
    avg_home REAL,
    avg_draw REAL,
    avg_away REAL,
    -- Over/Under odds
    b365_over25 REAL,
    b365_under25 REAL,
    pinnacle_over25 REAL,
    pinnacle_under25 REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league, match_date, home_team, away_team)
);

-- Team ratings (recalculated after each matchday)
CREATE TABLE IF NOT EXISTS team_ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league TEXT NOT NULL,
    team TEXT NOT NULL,
    season TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    -- Elo
    elo_rating REAL DEFAULT 1500,
    elo_home REAL DEFAULT 1500,
    elo_away REAL DEFAULT 1500,
    -- Attack / Defence strength
    attack_strength REAL,
    defence_weakness REAL,
    home_attack_strength REAL,
    home_defence_weakness REAL,
    away_attack_strength REAL,
    away_defence_weakness REAL,
    -- Form
    form_last5 REAL,            -- points from last 5 (0-15)
    form_last10 REAL,
    goals_scored_last5 REAL,
    goals_conceded_last5 REAL,
    -- Streaks
    current_streak_type TEXT,   -- W, D, L, U (unbeaten)
    current_streak_length INTEGER DEFAULT 0,
    -- Season stats
    played INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    goals_for INTEGER DEFAULT 0,
    goals_against INTEGER DEFAULT 0,
    points INTEGER DEFAULT 0,
    -- Meta
    is_promoted INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league, team, as_of_date)
);

-- Referee profiles
CREATE TABLE IF NOT EXISTS referee_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    referee TEXT NOT NULL,
    league TEXT NOT NULL,
    matches_officiated INTEGER DEFAULT 0,
    avg_home_goals REAL,
    avg_away_goals REAL,
    avg_total_goals REAL,
    avg_fouls REAL,
    avg_yellows REAL,
    avg_reds REAL,
    home_win_pct REAL,
    draw_pct REAL,
    away_win_pct REAL,
    over25_pct REAL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(referee, league)
);

-- Head-to-head records
CREATE TABLE IF NOT EXISTS head_to_head (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league TEXT NOT NULL,
    team_a TEXT NOT NULL,
    team_b TEXT NOT NULL,
    total_matches INTEGER DEFAULT 0,
    team_a_wins INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    team_b_wins INTEGER DEFAULT 0,
    team_a_goals INTEGER DEFAULT 0,
    team_b_goals INTEGER DEFAULT 0,
    last_5_results TEXT,        -- JSON: recent H2H results
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league, team_a, team_b)
);

-- Upcoming fixtures
CREATE TABLE IF NOT EXISTS fixtures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league TEXT NOT NULL,
    external_id INTEGER,        -- football-data.org match ID
    match_date TIMESTAMP NOT NULL,
    matchday INTEGER,
    status TEXT DEFAULT 'SCHEDULED',
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    referee TEXT,
    venue TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league, match_date, home_team, away_team)
);

-- Model predictions
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER,
    league TEXT NOT NULL,
    match_date DATE NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    -- Individual model outputs
    poisson_home REAL,
    poisson_draw REAL,
    poisson_away REAL,
    poisson_over25 REAL,
    poisson_btts REAL,
    elo_home REAL,
    elo_draw REAL,
    elo_away REAL,
    xgboost_home REAL,
    xgboost_draw REAL,
    xgboost_away REAL,
    sentiment_home REAL,
    sentiment_draw REAL,
    sentiment_away REAL,
    -- Ensemble (final)
    ensemble_home REAL,
    ensemble_draw REAL,
    ensemble_away REAL,
    ensemble_over25 REAL,
    ensemble_btts REAL,
    confidence REAL,
    -- Scoreline matrix (JSON)
    scoreline_matrix TEXT,
    -- Features used (JSON)
    features_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league, match_date, home_team, away_team)
);

-- Bookmaker odds for upcoming matches
CREATE TABLE IF NOT EXISTS odds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER,
    league TEXT NOT NULL,
    match_date DATE NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    home_odds REAL,
    draw_odds REAL,
    away_odds REAL,
    over25_odds REAL,
    under25_odds REAL,
    -- Derived
    home_implied REAL,
    draw_implied REAL,
    away_implied REAL,
    margin REAL,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league, match_date, home_team, away_team, bookmaker, fetched_at)
);

-- Value bets identified
CREATE TABLE IF NOT EXISTS value_bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER,
    league TEXT NOT NULL,
    match_date DATE NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    bet_type TEXT NOT NULL,         -- home_win, draw, away_win, over25, btts
    model_probability REAL,
    best_bookmaker TEXT,
    best_odds REAL,
    implied_probability REAL,
    edge_percent REAL,
    kelly_stake REAL,
    confidence REAL,
    -- Result tracking
    result TEXT,                    -- win, lose, void, pending
    profit_loss REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sentiment scores
CREATE TABLE IF NOT EXISTS sentiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league TEXT NOT NULL,
    team TEXT NOT NULL,
    score_date DATE NOT NULL,
    -- Reddit
    reddit_score REAL,             -- -1 to +1
    reddit_volume INTEGER,         -- number of mentions
    reddit_positive INTEGER,
    reddit_negative INTEGER,
    reddit_neutral INTEGER,
    -- News
    news_score REAL,
    news_volume INTEGER,
    news_keywords TEXT,            -- JSON: flagged keywords
    -- Combined
    combined_score REAL,
    sentiment_trend REAL,          -- change vs previous day
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league, team, score_date)
);

-- API call log (for rate limiting)
CREATE TABLE IF NOT EXISTS api_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    api_name TEXT NOT NULL,
    endpoint TEXT,
    called_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    response_code INTEGER,
    cached INTEGER DEFAULT 0
);

-- Model performance tracking
CREATE TABLE IF NOT EXISTS model_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league TEXT NOT NULL,
    season TEXT NOT NULL,
    model_name TEXT NOT NULL,       -- poisson, elo, xgboost, sentiment, ensemble
    total_predictions INTEGER,
    correct_predictions INTEGER,
    accuracy REAL,
    brier_score REAL,
    log_loss REAL,
    roi_percent REAL,
    yield_percent REAL,
    calibration_json TEXT,         -- JSON: binned calibration data
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league, season, model_name)
);

-- User bet tracking (portfolio)
CREATE TABLE IF NOT EXISTS user_bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league TEXT NOT NULL,
    match_date DATE NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    bet_type TEXT NOT NULL,         -- home_win, draw, away_win, over25, under25, btts_yes, btts_no
    stake REAL NOT NULL,
    odds REAL NOT NULL,
    bookmaker TEXT,
    -- Model info at time of bet
    model_probability REAL,
    edge_percent REAL,
    -- Result
    status TEXT DEFAULT 'pending',  -- pending, won, lost, void
    payout REAL DEFAULT 0,
    profit_loss REAL DEFAULT 0,
    -- Timestamps
    placed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settled_at TIMESTAMP,
    notes TEXT
);

-- CLV (Closing Line Value) tracking
CREATE TABLE IF NOT EXISTS clv_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league TEXT,
    match_date DATE,
    home_team TEXT,
    away_team TEXT,
    bet_type TEXT,
    model_probability REAL,
    pinnacle_closing_implied REAL,
    clv_percent REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_clv_league_date ON clv_tracking(league, match_date);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_matches_league_date ON matches(league, match_date);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_ratings_team ON team_ratings(league, team, as_of_date);
CREATE INDEX IF NOT EXISTS idx_fixtures_date ON fixtures(league, match_date);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(league, match_date);
CREATE INDEX IF NOT EXISTS idx_sentiment_team ON sentiment(league, team, score_date);
CREATE INDEX IF NOT EXISTS idx_value_bets_date ON value_bets(league, match_date);
