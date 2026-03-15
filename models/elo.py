"""
Elo Rating System for football teams.

Maintains a rolling Elo rating per team, updated after every match.
Generates win/draw/loss probabilities from Elo difference.
Supports home advantage, time decay, and separate home/away Elo.
"""
import math
import logging
from datetime import datetime

from database import db
import config

logger = logging.getLogger(__name__)

SETTINGS = config.ELO_SETTINGS


def expected_score(rating_a, rating_b):
    """Calculate expected score for team A against team B."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating, expected, actual, k=None):
    """Update Elo rating after a match."""
    k = k or SETTINGS["k_factor"]
    return rating + k * (actual - expected)


def result_to_score(result, is_home):
    """Convert match result (H/D/A) to Elo score (1/0.5/0)."""
    if result == "H":
        return 1.0 if is_home else 0.0
    elif result == "A":
        return 0.0 if is_home else 1.0
    else:
        return 0.5


def build_ratings(league="PL"):
    """
    Build Elo ratings from scratch using all historical matches.
    Returns dict of team -> {elo, elo_home, elo_away, form, streak}.
    """
    matches = db.fetch_all(
        """SELECT * FROM matches WHERE league = ? AND ft_result IS NOT NULL
           ORDER BY match_date ASC""",
        [league]
    )
    if not matches:
        return {}

    ratings = {}
    current_season = None

    for m in matches:
        home = m["home_team"]
        away = m["away_team"]
        result = m["ft_result"]
        season = m["season"]

        # Apply decay at season boundaries
        if current_season and season != current_season:
            decay = SETTINGS["decay_factor"]
            base = SETTINGS["initial_rating"]
            for team in ratings:
                ratings[team]["elo"] = base + (ratings[team]["elo"] - base) * decay
                ratings[team]["elo_home"] = base + (ratings[team]["elo_home"] - base) * decay
                ratings[team]["elo_away"] = base + (ratings[team]["elo_away"] - base) * decay
        current_season = season

        # Initialize new teams
        for team in [home, away]:
            if team not in ratings:
                ratings[team] = {
                    "elo": SETTINGS["initial_rating"],
                    "elo_home": SETTINGS["initial_rating"],
                    "elo_away": SETTINGS["initial_rating"],
                    "results": [],
                    "last_match_date": None,
                }

        # Calculate expected scores with home advantage
        home_elo = ratings[home]["elo"] + SETTINGS["home_advantage"]
        away_elo = ratings[away]["elo"]
        exp_home = expected_score(home_elo, away_elo)
        exp_away = 1 - exp_home

        # Actual scores
        actual_home = result_to_score(result, True)
        actual_away = result_to_score(result, False)

        # Goal difference bonus (larger wins = bigger Elo change)
        gd = abs(m["ft_home_goals"] - m["ft_away_goals"])
        k_mult = 1.0
        if gd == 2:
            k_mult = 1.5
        elif gd == 3:
            k_mult = 1.75
        elif gd >= 4:
            k_mult = 1.75 + (gd - 3) * 0.125
        k = SETTINGS["k_factor"] * k_mult

        # Update ratings
        ratings[home]["elo"] = update_elo(ratings[home]["elo"], exp_home, actual_home, k)
        ratings[away]["elo"] = update_elo(ratings[away]["elo"], exp_away, actual_away, k)
        ratings[home]["elo_home"] = update_elo(ratings[home]["elo_home"], exp_home, actual_home, k)
        ratings[away]["elo_away"] = update_elo(ratings[away]["elo_away"], exp_away, actual_away, k)

        # Track results for form calculation
        ratings[home]["results"].append(result)
        ratings[away]["results"].append({"H": "A", "A": "H", "D": "D"}[result])
        ratings[home]["last_match_date"] = m["match_date"]
        ratings[away]["last_match_date"] = m["match_date"]

    # Calculate form and streaks
    for team, data in ratings.items():
        recent = data["results"][-5:] if len(data["results"]) >= 5 else data["results"]
        data["form_last5"] = sum(3 if r == "H" else (1 if r == "D" else 0) for r in recent)
        data["form_last10"] = sum(
            3 if r == "H" else (1 if r == "D" else 0)
            for r in (data["results"][-10:] if len(data["results"]) >= 10 else data["results"])
        )

        # Streak detection
        if data["results"]:
            streak_type = data["results"][-1]
            streak_len = 0
            for r in reversed(data["results"]):
                if r == streak_type:
                    streak_len += 1
                elif streak_type in ("H", "D") and r in ("H", "D"):
                    # Unbeaten streak
                    streak_type = "U"
                    streak_len += 1
                else:
                    break
            data["streak_type"] = streak_type
            data["streak_length"] = streak_len
        else:
            data["streak_type"] = None
            data["streak_length"] = 0

        data["elo"] = round(data["elo"], 1)
        data["elo_home"] = round(data["elo_home"], 1)
        data["elo_away"] = round(data["elo_away"], 1)

    return ratings


def elo_to_probabilities(home_elo, away_elo, home_advantage=None):
    """
    Convert Elo ratings to win/draw/loss probabilities.
    Uses the Elo expected score + empirical draw adjustment.
    """
    ha = home_advantage if home_advantage is not None else SETTINGS["home_advantage"]
    exp_home = expected_score(home_elo + ha, away_elo)
    exp_away = 1 - exp_home

    # Empirical draw probability based on Elo closeness
    # Draws are more likely when teams are close in rating
    elo_diff = abs((home_elo + ha) - away_elo)
    draw_prob = 0.28 * math.exp(-0.002 * elo_diff)  # peaks at ~28% for equal teams

    # Adjust home/away probs
    home_win = exp_home * (1 - draw_prob)
    away_win = exp_away * (1 - draw_prob)

    return {
        "home_win": round(home_win, 4),
        "draw": round(draw_prob, 4),
        "away_win": round(away_win, 4),
    }


def predict(home_team, away_team, league="PL"):
    """Generate match prediction using Elo ratings."""
    ratings = build_ratings(league)

    if home_team not in ratings or away_team not in ratings:
        logger.warning("Missing Elo data for %s vs %s", home_team, away_team)
        return None

    home_data = ratings[home_team]
    away_data = ratings[away_team]

    probs = elo_to_probabilities(home_data["elo"], away_data["elo"])

    # Confidence: higher when Elo gap is larger
    elo_diff = abs(home_data["elo"] - away_data["elo"])
    confidence = min(elo_diff / 300, 1.0)  # normalise to 0-1

    return {
        "home_win": probs["home_win"],
        "draw": probs["draw"],
        "away_win": probs["away_win"],
        "confidence": round(confidence, 4),
        "details": {
            "home_elo": home_data["elo"],
            "away_elo": away_data["elo"],
            "home_elo_home": home_data["elo_home"],
            "away_elo_away": away_data["elo_away"],
            "home_form_last5": home_data["form_last5"],
            "away_form_last5": away_data["form_last5"],
            "home_streak": f"{home_data['streak_type']}{home_data['streak_length']}",
            "away_streak": f"{away_data['streak_type']}{away_data['streak_length']}",
            "elo_difference": round(home_data["elo"] - away_data["elo"], 1),
        }
    }
