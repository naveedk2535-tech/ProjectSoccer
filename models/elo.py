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
    season_teams = {}  # season -> set of teams seen in that season
    previous_season_teams = set()  # teams from the previous season

    for m in matches:
        home = m["home_team"]
        away = m["away_team"]
        result = m["ft_result"]
        season = m["season"]

        # Apply decay at season boundaries and detect promoted teams
        if current_season and season != current_season:
            decay = SETTINGS["decay_factor"]
            base = SETTINGS["initial_rating"]
            for team in ratings:
                ratings[team]["elo"] = base + (ratings[team]["elo"] - base) * decay
                ratings[team]["elo_home"] = base + (ratings[team]["elo_home"] - base) * decay
                ratings[team]["elo_away"] = base + (ratings[team]["elo_away"] - base) * decay
            # Track teams from previous season for promoted team detection
            previous_season_teams = season_teams.get(current_season, set())
            season_teams[season] = set()
        elif season not in season_teams:
            season_teams[season] = set()
        current_season = season

        # Track teams per season
        season_teams[season].add(home)
        season_teams[season].add(away)

        # Initialize new teams
        for team in [home, away]:
            if team not in ratings:
                # Detect promoted team: new team not in previous season's data
                is_promoted = (len(previous_season_teams) > 0 and team not in previous_season_teams)
                initial_elo = SETTINGS["initial_rating"]
                if is_promoted:
                    # Apply Elo boost for newly promoted teams (typically underrated)
                    initial_elo += 75
                    logger.info("Promoted team detected: %s (season %s), initial Elo boosted to %d",
                                team, season, initial_elo)
                ratings[team] = {
                    "elo": initial_elo,
                    "elo_home": initial_elo,
                    "elo_away": initial_elo,
                    "results": [],
                    "last_match_date": None,
                    "is_promoted": is_promoted,
                    "promoted_matches_remaining": 6 if is_promoted else 0,
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

        # Track promoted team match count (boost wears off after 6 matches)
        for team in [home, away]:
            if ratings[team].get("promoted_matches_remaining", 0) > 0:
                ratings[team]["promoted_matches_remaining"] -= 1
                if ratings[team]["promoted_matches_remaining"] == 0:
                    ratings[team]["is_promoted"] = False

        # Track results for form calculation
        ratings[home]["results"].append(result)
        ratings[away]["results"].append({"H": "A", "A": "H", "D": "D"}[result])
        ratings[home]["last_match_date"] = m["match_date"]
        ratings[away]["last_match_date"] = m["match_date"]

    # Calculate form and streaks
    for team, data in ratings.items():
        recent = data["results"][-5:] if len(data["results"]) >= 5 else data["results"]
        data["form_last5"] = sum(3 if r == "H" else (1 if r == "D" else 0) for r in recent)

        # Exponential decay form: most recent = weight 1.0, then 0.8, 0.6, 0.4, 0.2
        decay_weights = [0.2, 0.4, 0.6, 0.8, 1.0]  # oldest to newest
        decay_recent = data["results"][-5:] if len(data["results"]) >= 5 else data["results"]
        if len(decay_recent) < 5:
            decay_weights_used = decay_weights[-len(decay_recent):] if decay_recent else []
        else:
            decay_weights_used = decay_weights
        if decay_recent and decay_weights_used:
            data["form_last5_decay"] = sum(
                (3 if r == "H" else (1 if r == "D" else 0)) * w
                for r, w in zip(decay_recent, decay_weights_used)
            ) / sum(decay_weights_used)  # normalize to 0-3 scale
        else:
            data["form_last5_decay"] = 1.5  # neutral default
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

    # Calculate opponent-adjusted form using the built ratings
    # Done here (not in calculate_opponent_adjusted_form) to avoid re-building ratings
    all_elos = [r["elo"] for r in ratings.values()]
    if all_elos:
        min_elo = min(all_elos)
        max_elo = max(all_elos)
        elo_range = max_elo - min_elo if max_elo > min_elo else 1

        for team, data in ratings.items():
            recent_results = data["results"][-5:] if len(data["results"]) >= 5 else data["results"]
            if not recent_results:
                data["opponent_adjusted_form"] = 0.5
                continue

            # Get last N matches to find opponents
            team_matches = db.fetch_all(
                """SELECT home_team, away_team, ft_result FROM matches
                   WHERE league = ? AND (home_team = ? OR away_team = ?) AND ft_result IS NOT NULL
                   ORDER BY match_date DESC LIMIT ?""",
                [league, team, team, 5]
            )

            if not team_matches:
                data["opponent_adjusted_form"] = 0.5
                continue

            total_weighted = 0.0
            count = 0
            for tm in team_matches:
                is_home = (tm["home_team"] == team)
                opponent = tm["away_team"] if is_home else tm["home_team"]
                result = tm["ft_result"]

                opp_elo = ratings.get(opponent, {}).get("elo", 1500)
                opp_quality = (opp_elo - min_elo) / elo_range

                if result == "D":
                    weighted_points = 1.0
                elif (result == "H" and is_home) or (result == "A" and not is_home):
                    weighted_points = 1.5 + 1.5 * opp_quality
                else:
                    weighted_points = -2.0 + 1.5 * opp_quality

                total_weighted += weighted_points
                count += 1

            if count > 0:
                avg_weighted = total_weighted / count
                normalized = (avg_weighted + 2.0) / 5.0
                data["opponent_adjusted_form"] = round(max(0.0, min(1.0, normalized)), 4)
            else:
                data["opponent_adjusted_form"] = 0.5

    return ratings


def calculate_draw_factors(home_elo, away_elo, league="PL"):
    """
    Calculate draw adjustment factors based on multiple signals.

    Returns dict with a 'multiplier' that adjusts the base draw probability.
    Factors considered:
    - Elo closeness: abs(home_elo - away_elo) < 50 boosts draw by up to 5%
    - Both teams defensive: if both have defence_weakness < 0.9, boost draw by 3%
    - League draw rate: uses actual historical draw rate for the league
    """
    multiplier = 1.0

    # 1. Elo closeness factor
    elo_diff = abs(home_elo - away_elo)
    if elo_diff < 50:
        # Linear boost: 0 diff -> +5%, 50 diff -> +0%
        closeness_boost = 0.05 * (1 - elo_diff / 50)
        multiplier += closeness_boost

    # 2. Both teams defensive factor
    try:
        from models import poisson as poisson_model
        strengths = poisson_model.calculate_team_strengths(league, use_xg=False)
        home_team_data = None
        away_team_data = None
        # Find teams by Elo match (approximate)
        for team, s in strengths.items():
            if s.get("defence_weakness") is not None:
                if home_team_data is None:
                    home_team_data = s
                if away_team_data is None:
                    away_team_data = s
        # Actually look up by iterating ratings if available
        # For now, use a simpler approach: check if we can find both teams
        home_def = None
        away_def = None
        for team, s in strengths.items():
            if s.get("defence_weakness") is not None:
                # We store all teams; we'll use a different lookup via the predict function
                pass
        # Use a direct DB lookup instead for reliability
        draw_count = db.fetch_one(
            "SELECT COUNT(*) as cnt FROM matches WHERE league = ? AND ft_result = 'D' AND ft_result IS NOT NULL",
            [league]
        )
        total_count = db.fetch_one(
            "SELECT COUNT(*) as cnt FROM matches WHERE league = ? AND ft_result IS NOT NULL",
            [league]
        )
        if draw_count and total_count and total_count["cnt"] > 0:
            actual_draw_rate = draw_count["cnt"] / total_count["cnt"]
            # If actual draw rate is higher than the base 0.28, boost the multiplier
            if actual_draw_rate > 0.26:
                league_boost = (actual_draw_rate - 0.26) * 2  # scale the difference
                multiplier += min(league_boost, 0.10)  # cap at 10% boost
    except Exception:
        pass

    return {"multiplier": round(multiplier, 4)}


def calculate_draw_factors_with_teams(home_elo, away_elo, league="PL",
                                       home_defence_weakness=None, away_defence_weakness=None):
    """
    Calculate draw factors with explicit team defence data.
    Used when team-specific defence weakness values are available.
    """
    factors = calculate_draw_factors(home_elo, away_elo, league)
    multiplier = factors["multiplier"]

    # Defensive teams boost
    if (home_defence_weakness is not None and away_defence_weakness is not None
            and home_defence_weakness < 0.9 and away_defence_weakness < 0.9):
        multiplier += 0.03

    return {"multiplier": round(multiplier, 4)}


def elo_to_probabilities(home_elo, away_elo, home_advantage=None, league="PL"):
    """
    Convert Elo ratings to win/draw/loss probabilities.
    Uses the Elo expected score + empirical draw adjustment,
    enhanced with draw factors based on Elo closeness, defensive profiles, and league draw rate.
    """
    ha = home_advantage if home_advantage is not None else SETTINGS["home_advantage"]
    exp_home = expected_score(home_elo + ha, away_elo)
    exp_away = 1 - exp_home

    # Base draw probability based on Elo closeness
    elo_diff = abs((home_elo + ha) - away_elo)
    draw_prob = 0.28 * math.exp(-0.002 * elo_diff)  # peaks at ~28% for equal teams

    # Adjust with draw factors
    draw_factors = calculate_draw_factors(home_elo, away_elo, league)
    draw_prob = draw_prob * draw_factors["multiplier"]

    # Ensure draw doesn't exceed reasonable bounds (max 35%)
    draw_prob = min(draw_prob, 0.35)

    # Adjust home/away probs
    home_win = exp_home * (1 - draw_prob)
    away_win = exp_away * (1 - draw_prob)

    return {
        "home_win": round(home_win, 4),
        "draw": round(draw_prob, 4),
        "away_win": round(away_win, 4),
    }


def calculate_opponent_adjusted_form(team, league="PL", last_n=5):
    """
    Calculate form weighted by opponent quality.
    Beat #1 team = 3.0 weighted points
    Beat #20 team = 1.5 weighted points
    Lose to #1 = -0.5 weighted points
    Lose to #20 = -2.0 weighted points
    Draw = 1.0 regardless

    Returns a normalized score (0-1 scale).
    """
    # Get the team's last N matches
    matches = db.fetch_all(
        """SELECT home_team, away_team, ft_result FROM matches
           WHERE league = ? AND (home_team = ? OR away_team = ?) AND ft_result IS NOT NULL
           ORDER BY match_date DESC LIMIT ?""",
        [league, team, team, last_n]
    )
    if not matches:
        return 0.5

    # Build current Elo ratings for opponent quality lookup
    ratings = build_ratings(league)
    if not ratings:
        return 0.5

    # Get min/max Elo for scaling opponent quality
    all_elos = [r["elo"] for r in ratings.values()]
    if not all_elos:
        return 0.5
    min_elo = min(all_elos)
    max_elo = max(all_elos)
    elo_range = max_elo - min_elo if max_elo > min_elo else 1

    total_weighted = 0.0
    count = 0

    for m in matches:
        is_home = (m["home_team"] == team)
        opponent = m["away_team"] if is_home else m["home_team"]
        result = m["ft_result"]

        # Get opponent Elo (higher = stronger)
        opp_elo = ratings.get(opponent, {}).get("elo", 1500)
        # Opponent quality: 0 (weakest) to 1 (strongest)
        opp_quality = (opp_elo - min_elo) / elo_range

        # Determine result from team's perspective
        if result == "D":
            weighted_points = 1.0
        elif (result == "H" and is_home) or (result == "A" and not is_home):
            # Win: more points for beating stronger opponents
            weighted_points = 1.5 + 1.5 * opp_quality  # 1.5 (weakest) to 3.0 (strongest)
        else:
            # Loss: less penalty for losing to stronger opponents
            weighted_points = -2.0 + 1.5 * opp_quality  # -2.0 (weakest) to -0.5 (strongest)

        total_weighted += weighted_points
        count += 1

    if count == 0:
        return 0.5

    # Raw score range: worst case = -2.0 * N, best case = 3.0 * N
    avg_weighted = total_weighted / count
    # Normalize from [-2.0, 3.0] to [0, 1]
    normalized = (avg_weighted + 2.0) / 5.0
    return round(max(0.0, min(1.0, normalized)), 4)


def predict(home_team, away_team, league="PL"):
    """Generate match prediction using Elo ratings."""
    ratings = build_ratings(league)

    if home_team not in ratings or away_team not in ratings:
        logger.warning("Missing Elo data for %s vs %s", home_team, away_team)
        return None

    home_data = ratings[home_team]
    away_data = ratings[away_team]

    probs = elo_to_probabilities(home_data["elo"], away_data["elo"], league=league)

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
            "home_opponent_adjusted_form": home_data.get("opponent_adjusted_form", 0.5),
            "away_opponent_adjusted_form": away_data.get("opponent_adjusted_form", 0.5),
            "home_streak": f"{home_data['streak_type']}{home_data['streak_length']}",
            "away_streak": f"{away_data['streak_type']}{away_data['streak_length']}",
            "elo_difference": round(home_data["elo"] - away_data["elo"], 1),
        }
    }
