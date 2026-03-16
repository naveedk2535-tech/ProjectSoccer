"""
The Odds API client.
Fetches live bookmaker odds for upcoming matches.
"""
import logging
from datetime import datetime

import requests

import config
from data.rate_limiter import can_call, record_call, check_cache, save_cache
from data.team_names import standardise
from database import db

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"


def get_odds(league_code="PL", markets="h2h,totals"):
    """
    Fetch odds for upcoming matches in a league.
    markets: h2h (1X2), totals (over/under), spreads
    """
    league_config = config.LEAGUES.get(league_code)
    if not league_config:
        logger.error("Unknown league: %s", league_code)
        return []

    sport_key = league_config["odds_api_key"]
    cache_key = f"odds_{league_code}"
    cached = check_cache(cache_key, config.CACHE_TTL["odds"])
    if cached:
        logger.info("Using cached odds for %s", league_code)
        return cached

    if not can_call("odds_api"):
        logger.warning("Rate limit reached for The Odds API")
        return []

    try:
        response = requests.get(
            f"{BASE_URL}/sports/{sport_key}/odds",
            params={
                "apiKey": config.ODDS_API_KEY,
                "regions": "uk,eu",
                "markets": markets,
                "oddsFormat": "decimal",
            },
            timeout=30
        )
        record_call("odds_api", f"/sports/{sport_key}/odds", response.status_code)

        # Track remaining API calls
        remaining = response.headers.get("x-requests-remaining", "?")
        used = response.headers.get("x-requests-used", "?")
        logger.info("Odds API: %s used, %s remaining this month", used, remaining)

        if response.status_code != 200:
            logger.error("Odds API error %d: %s", response.status_code, response.text[:200])
            return []

        events = response.json()
        save_cache(cache_key, events)
        _save_odds_to_db(events, league_code)
        return events

    except Exception as e:
        logger.error("Odds API request failed: %s", e)
        return []


def _save_odds_to_db(events, league_code):
    """Parse odds response and save to database."""
    rows = []
    for event in events:
        home_team = standardise(event.get("home_team", ""))
        away_team = standardise(event.get("away_team", ""))
        match_date = event.get("commence_time", "")[:10]

        for bookmaker in event.get("bookmakers", []):
            bookie_name = bookmaker.get("title", "")
            odds_row = {
                "league": league_code,
                "match_date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "bookmaker": bookie_name,
                "home_odds": None,
                "draw_odds": None,
                "away_odds": None,
                "over25_odds": None,
                "under25_odds": None,
                "home_implied": None,
                "draw_implied": None,
                "away_implied": None,
                "margin": None,
                "fetched_at": datetime.utcnow().isoformat(),
            }

            for market in bookmaker.get("markets", []):
                if market["key"] == "h2h":
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == home_team:
                            odds_row["home_odds"] = outcome["price"]
                        elif outcome["name"] == away_team:
                            odds_row["away_odds"] = outcome["price"]
                        elif outcome["name"] == "Draw":
                            odds_row["draw_odds"] = outcome["price"]

                elif market["key"] == "totals":
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == "Over":
                            odds_row["over25_odds"] = outcome["price"]
                        elif outcome["name"] == "Under":
                            odds_row["under25_odds"] = outcome["price"]

            # Calculate implied probabilities
            if odds_row.get("home_odds") and odds_row.get("draw_odds") and odds_row.get("away_odds"):
                total = (1/odds_row["home_odds"] + 1/odds_row["draw_odds"] + 1/odds_row["away_odds"])
                odds_row["home_implied"] = round((1/odds_row["home_odds"]) / total, 4)
                odds_row["draw_implied"] = round((1/odds_row["draw_odds"]) / total, 4)
                odds_row["away_implied"] = round((1/odds_row["away_odds"]) / total, 4)
                odds_row["margin"] = round((total - 1) * 100, 2)

            rows.append(odds_row)

    if rows:
        db.insert_many("odds", rows)
        logger.info("Saved odds from %d bookmakers for %d events", len(rows), len(events))


def get_best_odds(league_code, home_team, away_team):
    """Get the best available odds across all bookmakers for a match."""
    rows = db.fetch_all(
        """SELECT bookmaker, home_odds, draw_odds, away_odds,
                  over25_odds, under25_odds, home_implied, draw_implied, away_implied, margin
           FROM odds
           WHERE league = ? AND home_team = ? AND away_team = ?
           ORDER BY fetched_at DESC""",
        [league_code, home_team, away_team]
    )
    if not rows:
        return None

    best = {
        "home": max(rows, key=lambda r: r.get("home_odds") or 0),
        "draw": max(rows, key=lambda r: r.get("draw_odds") or 0),
        "away": max(rows, key=lambda r: r.get("away_odds") or 0),
        "all_bookmakers": rows,
    }
    return best


def get_remaining_quota():
    """Check how many API calls remain this month."""
    try:
        response = requests.get(
            f"{BASE_URL}/sports",
            params={"apiKey": config.ODDS_API_KEY},
            timeout=10
        )
        return {
            "remaining": response.headers.get("x-requests-remaining", "unknown"),
            "used": response.headers.get("x-requests-used", "unknown"),
        }
    except Exception:
        return {"remaining": "unknown", "used": "unknown"}
