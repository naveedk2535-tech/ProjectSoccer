"""
ProjectSoccer Configuration
Central place for all settings, rate limits, and constants.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "ProjectSoccer/1.0")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-fallback-key")

# --- Rate Limits (requests per window) ---
RATE_LIMITS = {
    "football_data_org": {"calls": 10, "period_seconds": 60},      # 10/min (API hard limit)
    "odds_api":          {"calls": 5,  "period_seconds": 86400},    # 5/day (~150/month, well under 500)
    "reddit":            {"calls": 2,  "period_seconds": 86400},    # 2/day (morning + pre-match)
    "newsapi":           {"calls": 2,  "period_seconds": 86400},    # 2/day (morning + pre-match)
    "football_data_uk":  {"calls": 1,  "period_seconds": 172800},   # 1 per 2 days
}

# --- Cache TTLs (seconds) ---
CACHE_TTL = {
    "fixtures": 43200,      # 12 hours
    "odds": 14400,          # 4 hours
    "sentiment": 86400,     # 24 hours
    "historical_csv": 172800,  # 2 days
    "standings": 43200,     # 12 hours
}

# --- League Configuration ---
LEAGUES = {
    "PL": {
        "name": "Premier League",
        "country": "England",
        "football_data_org_code": "PL",
        "football_data_uk_code": "E0",
        "odds_api_key": "soccer_epl",
        "subreddits": ["PremierLeague", "soccer"],
        "seasons": ["2526", "2425", "2324"],  # current + 2 historical
        "enabled": True,
    },
    "ELC": {
        "name": "Championship",
        "country": "England",
        "football_data_org_code": "ELC",
        "football_data_uk_code": "E1",
        "odds_api_key": "soccer_efl_champ",
        "subreddits": ["Championship", "soccer"],
        "seasons": ["2526", "2425", "2324"],
        "enabled": False,  # Phase 6
    },
    "SPL": {
        "name": "Scottish Premiership",
        "country": "Scotland",
        "football_data_org_code": "SPL",  # not available on free tier
        "football_data_uk_code": "SC0",
        "odds_api_key": "soccer_spl",
        "subreddits": ["ScottishFootball", "soccer"],
        "seasons": ["2526", "2425", "2324"],
        "enabled": False,
    },
    "PD": {
        "name": "La Liga",
        "country": "Spain",
        "football_data_org_code": "PD",
        "football_data_uk_code": "SP1",
        "odds_api_key": "soccer_spain_la_liga",
        "subreddits": ["LaLiga", "soccer"],
        "seasons": ["2526", "2425", "2324"],
        "enabled": False,
    },
    "BL1": {
        "name": "Bundesliga",
        "country": "Germany",
        "football_data_org_code": "BL1",
        "football_data_uk_code": "D1",
        "odds_api_key": "soccer_germany_bundesliga",
        "subreddits": ["Bundesliga", "soccer"],
        "seasons": ["2526", "2425", "2324"],
        "enabled": False,
    },
    "SA": {
        "name": "Serie A",
        "country": "Italy",
        "football_data_org_code": "SA",
        "football_data_uk_code": "I1",
        "odds_api_key": "soccer_italy_serie_a",
        "subreddits": ["SerieA", "soccer"],
        "seasons": ["2526", "2425", "2324"],
        "enabled": False,
    },
    "FL1": {
        "name": "Ligue 1",
        "country": "France",
        "football_data_org_code": "FL1",
        "football_data_uk_code": "F1",
        "odds_api_key": "soccer_france_ligue_one",
        "subreddits": ["Ligue1", "soccer"],
        "seasons": ["2526", "2425", "2324"],
        "enabled": False,
    },
}

# --- Model Settings ---
MODEL_WEIGHTS = {
    "poisson": 0.35,
    "elo": 0.25,
    "xgboost": 0.30,
    "sentiment": 0.10,
}

ELO_SETTINGS = {
    "initial_rating": 1500,
    "k_factor": 20,
    "home_advantage": 50,
    "decay_factor": 0.95,  # per season
}

POISSON_SETTINGS = {
    "min_matches": 5,  # minimum matches before predictions
    "time_decay_weight": 0.005,  # more recent matches weighted higher
}

VALUE_BET_SETTINGS = {
    "min_edge_percent": 5.0,      # minimum edge to flag as value
    "kelly_fraction": 0.25,       # quarter-Kelly
    "max_stake_percent": 5.0,     # never stake more than 5% of bankroll
    "min_odds": 1.20,             # ignore very short odds
    "max_odds": 15.0,             # ignore extreme longshots
}

# --- Database ---
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "projectsoccer.db")

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
