"""
Football-data.co.uk CSV downloader and parser.
Downloads historical match results + odds for backtesting and model training.
"""
import io
import logging
from datetime import datetime

import pandas as pd
import requests

import config
from data.rate_limiter import can_call, record_call, check_cache, save_cache
from data.team_names import standardise
from database import db

logger = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Column mapping from CSV to our schema
COLUMN_MAP = {
    "Date": "match_date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "ft_home_goals",
    "FTAG": "ft_away_goals",
    "FTR": "ft_result",
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HTR": "ht_result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_target",
    "AST": "away_shots_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
    "Referee": "referee",
    "B365H": "b365_home",
    "B365D": "b365_draw",
    "B365A": "b365_away",
    "PSH": "pinnacle_home",
    "PSD": "pinnacle_draw",
    "PSA": "pinnacle_away",
    "MaxH": "max_home",
    "MaxD": "max_draw",
    "MaxA": "max_away",
    "AvgH": "avg_home",
    "AvgD": "avg_draw",
    "AvgA": "avg_away",
    "BbOU": "b365_over25",        # may not exist in all CSVs
    "BbMx>2.5": "b365_over25",
    "BbMx<2.5": "b365_under25",
    "P>2.5": "pinnacle_over25",
    "P<2.5": "pinnacle_under25",
}

# Standardise team names (common variations)
TEAM_NAME_MAP = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham": "Nottingham Forest",
    "Sheffield Utd": "Sheffield United",
    "Sheffield United": "Sheffield United",
    "Wolves": "Wolverhampton",
    "Wolverhampton Wanderers": "Wolverhampton",
    "West Ham": "West Ham United",
    "Newcastle": "Newcastle United",
    "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Brighton": "Brighton & Hove Albion",
    "West Brom": "West Bromwich Albion",
    "Ipswich": "Ipswich Town",
    "Luton": "Luton Town",
    "Norwich": "Norwich City",
    "Bournemouth": "AFC Bournemouth",
}


def standardise_team_name(name):
    """Standardise team name to consistent format."""
    if pd.isna(name):
        return name
    return standardise(name)


def parse_date(date_str):
    """Parse date from CSV (handles multiple formats)."""
    if pd.isna(date_str):
        return None
    for fmt in ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"]:
        try:
            return datetime.strptime(str(date_str).strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    logger.warning("Could not parse date: %s", date_str)
    return None


def download_season(league_code, season, uk_code):
    """Download a single season CSV from football-data.co.uk."""
    cache_key = f"csv_{league_code}_{season}"
    cached = check_cache(cache_key, config.CACHE_TTL["historical_csv"])
    if cached:
        logger.info("Using cached CSV for %s %s", league_code, season)
        return pd.DataFrame(cached)

    if not can_call("football_data_uk"):
        logger.warning("Rate limit reached for football-data.co.uk")
        return None

    url = f"{BASE_URL}/{season}/{uk_code}.csv"
    logger.info("Downloading %s", url)

    try:
        response = requests.get(url, timeout=30)
        record_call("football_data_uk", url, response.status_code)

        if response.status_code != 200:
            logger.error("Failed to download %s: HTTP %d", url, response.status_code)
            return None

        df = pd.read_csv(io.StringIO(response.text), on_bad_lines="skip")
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"], how="any")

        if df.empty:
            logger.warning("Empty CSV for %s %s", league_code, season)
            return None

        # Cache raw data
        save_cache(cache_key, df.to_dict(orient="records"))
        logger.info("Downloaded %d matches for %s %s", len(df), league_code, season)
        return df

    except Exception as e:
        logger.error("Error downloading %s: %s", url, e)
        return None


def process_csv(df, league_code, season):
    """Clean and standardise a CSV dataframe, then insert into database."""
    # Rename columns
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Standardise team names
    if "home_team" in df.columns:
        df["home_team"] = df["home_team"].apply(standardise_team_name)
    if "away_team" in df.columns:
        df["away_team"] = df["away_team"].apply(standardise_team_name)

    # Parse dates
    if "match_date" in df.columns:
        df["match_date"] = df["match_date"].apply(parse_date)
        df = df.dropna(subset=["match_date"])

    # Add metadata
    df["league"] = league_code
    df["season"] = season

    # Select only columns that exist in our schema
    schema_cols = [
        "league", "season", "match_date", "home_team", "away_team",
        "ft_home_goals", "ft_away_goals", "ft_result",
        "ht_home_goals", "ht_away_goals", "ht_result",
        "home_shots", "away_shots", "home_shots_target", "away_shots_target",
        "home_corners", "away_corners", "home_fouls", "away_fouls",
        "home_yellows", "away_yellows", "home_reds", "away_reds",
        "referee",
        "b365_home", "b365_draw", "b365_away",
        "pinnacle_home", "pinnacle_draw", "pinnacle_away",
        "max_home", "max_draw", "max_away",
        "avg_home", "avg_draw", "avg_away",
        "b365_over25", "b365_under25",
        "pinnacle_over25", "pinnacle_under25",
    ]
    available_cols = [c for c in schema_cols if c in df.columns]
    df = df[available_cols]

    # Convert numeric columns
    numeric_cols = [c for c in df.columns if c not in
                    ["league", "season", "match_date", "home_team", "away_team",
                     "ft_result", "ht_result", "referee"]]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Insert into database
    rows = df.to_dict(orient="records")
    inserted = db.insert_many("matches", rows)
    logger.info("Inserted %d new matches for %s %s", inserted, league_code, season)
    return inserted


def download_all_leagues():
    """Download all enabled leagues and seasons."""
    total = 0
    for code, league in config.LEAGUES.items():
        if not league["enabled"]:
            continue
        for season in league["seasons"]:
            df = download_season(code, season, league["football_data_uk_code"])
            if df is not None:
                count = process_csv(df, code, season)
                total += count
    return total


def get_match_count(league=None):
    """Get total match count, optionally filtered by league."""
    if league:
        row = db.fetch_one("SELECT COUNT(*) as cnt FROM matches WHERE league = ?", [league])
    else:
        row = db.fetch_one("SELECT COUNT(*) as cnt FROM matches")
    return row["cnt"] if row else 0
