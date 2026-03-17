"""
Football-data.org API client.
Fetches upcoming fixtures, standings, and match schedules.
"""
import logging
from datetime import datetime, timedelta

import requests

import config
from data.rate_limiter import can_call, record_call, check_cache, save_cache
from data.team_names import standardise
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
    """Fetch upcoming AND recently finished fixtures for a league."""
    date_from = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
    date_to = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    data = _api_get(
        f"/competitions/{league_code}/matches?dateFrom={date_from}&dateTo={date_to}",
        cache_key=f"fixtures_{league_code}",
        cache_ttl=config.CACHE_TTL["fixtures"]
    )
    if not data or "matches" not in data:
        return []

    fixtures = []
    finished_count = 0
    for m in data["matches"]:
        home_team = standardise(m.get("homeTeam", {}).get("name", ""))
        away_team = standardise(m.get("awayTeam", {}).get("name", ""))
        status = m.get("status")
        match_date = m.get("utcDate", "")
        score = m.get("score", {})
        ft = score.get("fullTime", {})
        referee_name = (m.get("referees") or [{}])[0].get("name") if m.get("referees") else None

        fixture = {
            "league": league_code,
            "external_id": m.get("id"),
            "match_date": match_date,
            "matchday": m.get("matchday"),
            "status": status,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": ft.get("home"),
            "away_score": ft.get("away"),
            "referee": referee_name,
            "venue": m.get("venue"),
        }

        # Update fixture status (TIMED → FINISHED)
        db.execute(
            """INSERT INTO fixtures (league, external_id, match_date, matchday, status,
                   home_team, away_team, home_score, away_score, referee, venue)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(league, match_date, home_team, away_team) DO UPDATE SET
                   status=excluded.status, home_score=excluded.home_score,
                   away_score=excluded.away_score, referee=excluded.referee""",
            [league_code, fixture["external_id"], match_date, fixture["matchday"],
             status, home_team, away_team, ft.get("home"), ft.get("away"),
             referee_name, fixture["venue"]]
        )

        # If match is finished, also insert into matches table
        if status == "FINISHED" and ft.get("home") is not None:
            home_goals = ft.get("home")
            away_goals = ft.get("away")
            ht = score.get("halfTime", {})
            result = "H" if home_goals > away_goals else ("A" if away_goals > home_goals else "D")

            db.execute(
                """INSERT OR IGNORE INTO matches
                   (league, season, match_date, home_team, away_team,
                    ft_home_goals, ft_away_goals, ft_result,
                    ht_home_goals, ht_away_goals, referee)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [league_code, "2526", match_date[:10], home_team, away_team,
                 home_goals, away_goals, result,
                 ht.get("home"), ht.get("away"), referee_name]
            )
            finished_count += 1

        if status in ("SCHEDULED", "TIMED"):
            fixtures.append(fixture)

    logger.info("Saved %d fixtures for %s (%d finished results imported)",
                len(fixtures) + finished_count, league_code, finished_count)

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
                    "team": standardise(entry.get("team", {}).get("name", "")),
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
