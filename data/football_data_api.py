"""
Football-data.org API client.
Fetches upcoming fixtures, standings, and match schedules.
"""
import logging
from datetime import datetime, timedelta

import requests

import config
from data.rate_limiter import can_call, record_call, check_cache, save_cache
from database import db

logger = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": config.FOOTBALL_DATA_API_KEY}


def _api_get(endpoint, cache_key=None, cache_ttl=None):
    """Make an authenticated GET request with caching and rate limiting."""
    if cache_key and cache_ttl:
        cached = check_cache(cache_key, cache_ttl)
        if cached:
            logger.info("Cache hit: %s", cache_key)
            return cached

    if not can_call("football_data_org"):
        logger.warning("Rate limit reached for football-data.org")
        return None

    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        record_call("football_data_org", endpoint, response.status_code)

        if response.status_code == 429:
            logger.warning("football-data.org rate limited us (429)")
            return None
        if response.status_code != 200:
            logger.error("API error %d: %s", response.status_code, url)
            return None

        data = response.json()
        if cache_key:
            save_cache(cache_key, data)
        return data

    except Exception as e:
        logger.error("API request failed: %s - %s", url, e)
        return None


def get_upcoming_fixtures(league_code="PL", days_ahead=14):
    """Fetch upcoming fixtures for a league."""
    date_from = datetime.utcnow().strftime("%Y-%m-%d")
    date_to = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    data = _api_get(
        f"/competitions/{league_code}/matches?dateFrom={date_from}&dateTo={date_to}&status=SCHEDULED,TIMED",
        cache_key=f"fixtures_{league_code}",
        cache_ttl=config.CACHE_TTL["fixtures"]
    )
    if not data or "matches" not in data:
        return []

    fixtures = []
    for m in data["matches"]:
        fixture = {
            "league": league_code,
            "external_id": m.get("id"),
            "match_date": m.get("utcDate"),
            "matchday": m.get("matchday"),
            "status": m.get("status"),
            "home_team": m.get("homeTeam", {}).get("name", ""),
            "away_team": m.get("awayTeam", {}).get("name", ""),
            "referee": (m.get("referees") or [{}])[0].get("name") if m.get("referees") else None,
            "venue": m.get("venue"),
        }
        fixtures.append(fixture)

    # Save to database
    if fixtures:
        db.insert_many("fixtures", fixtures)
        logger.info("Saved %d upcoming fixtures for %s", len(fixtures), league_code)

    return fixtures


def get_standings(league_code="PL"):
    """Fetch current league standings."""
    data = _api_get(
        f"/competitions/{league_code}/standings",
        cache_key=f"standings_{league_code}",
        cache_ttl=config.CACHE_TTL["standings"]
    )
    if not data or "standings" not in data:
        return []

    standings = []
    for table in data.get("standings", []):
        if table.get("type") == "TOTAL":
            for entry in table.get("table", []):
                team = {
                    "position": entry.get("position"),
                    "team": entry.get("team", {}).get("name", ""),
                    "team_crest": entry.get("team", {}).get("crest", ""),
                    "played": entry.get("playedGames"),
                    "won": entry.get("won"),
                    "draw": entry.get("draw"),
                    "lost": entry.get("lost"),
                    "goals_for": entry.get("goalsFor"),
                    "goals_against": entry.get("goalsAgainst"),
                    "goal_difference": entry.get("goalDifference"),
                    "points": entry.get("points"),
                    "form": entry.get("form"),
                }
                standings.append(team)
    return standings


def get_team_matches(league_code="PL", team_id=None, limit=10):
    """Fetch recent matches for a specific team."""
    data = _api_get(
        f"/competitions/{league_code}/matches?status=FINISHED&limit={limit}",
        cache_key=f"recent_{league_code}_{limit}",
        cache_ttl=config.CACHE_TTL["fixtures"]
    )
    if not data or "matches" not in data:
        return []
    return data["matches"]


def get_competition_info(league_code="PL"):
    """Fetch competition metadata (current season, matchday, etc.)."""
    data = _api_get(
        f"/competitions/{league_code}",
        cache_key=f"competition_{league_code}",
        cache_ttl=config.CACHE_TTL["standings"]
    )
    if not data:
        return {}
    return {
        "name": data.get("name"),
        "current_season": data.get("currentSeason", {}),
        "current_matchday": data.get("currentSeason", {}).get("currentMatchday"),
    }
