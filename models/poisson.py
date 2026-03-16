"""
Poisson + Dixon-Coles Prediction Model.

Calculates attack strength and defence weakness per team,
then uses Poisson distribution to generate a full scoreline probability matrix.
Dixon-Coles correction adjusts for low-scoring bias.
"""
import logging
import math
from datetime import datetime, timedelta

import numpy as np
from scipy.stats import poisson

from database import db
import config

logger = logging.getLogger(__name__)

MAX_GOALS = 7  # 0-6 goals per team in probability matrix


def calculate_half_profiles(league="PL"):
    """
    Build per-team half-time performance profiles.

    For each team, calculate:
    - Avg goals scored in 1st half vs 2nd half (home and away)
    - Comeback rate: % of games losing at HT but won/drew FT
    - Collapse rate: % of games winning at HT but drew/lost FT

    Returns dict of team -> profile dict.
    """
    matches = db.fetch_all(
        """SELECT * FROM matches WHERE league = ? AND ft_home_goals IS NOT NULL
           AND ht_home_goals IS NOT NULL
           ORDER BY match_date ASC""",
        [league]
    )
    if not matches:
        return {}

    team_profiles = {}

    for m in matches:
        home = m["home_team"]
        away = m["away_team"]

        ht_home = m.get("ht_home_goals")
        ht_away = m.get("ht_away_goals")
        ft_home = m["ft_home_goals"]
        ft_away = m["ft_away_goals"]

        # Skip if HT data is missing
        if ht_home is None or ht_away is None:
            continue

        second_half_home = ft_home - ht_home
        second_half_away = ft_away - ht_away

        # Determine HT and FT results for each team's perspective
        if ht_home > ht_away:
            ht_home_status, ht_away_status = "winning", "losing"
        elif ht_home < ht_away:
            ht_home_status, ht_away_status = "losing", "winning"
        else:
            ht_home_status, ht_away_status = "drawing", "drawing"

        if ft_home > ft_away:
            ft_home_status, ft_away_status = "won", "lost"
        elif ft_home < ft_away:
            ft_home_status, ft_away_status = "lost", "won"
        else:
            ft_home_status, ft_away_status = "drew", "drew"

        for team, is_home in [(home, True), (away, False)]:
            if team not in team_profiles:
                team_profiles[team] = {
                    "home_1h_goals": 0, "home_2h_goals": 0, "home_games": 0,
                    "away_1h_goals": 0, "away_2h_goals": 0, "away_games": 0,
                    "losing_ht_count": 0, "comeback_count": 0,
                    "winning_ht_count": 0, "collapse_count": 0,
                }

            tp = team_profiles[team]
            if is_home:
                tp["home_1h_goals"] += ht_home
                tp["home_2h_goals"] += second_half_home
                tp["home_games"] += 1
                ht_status = ht_home_status
                ft_status = ft_home_status
            else:
                tp["away_1h_goals"] += ht_away
                tp["away_2h_goals"] += second_half_away
                tp["away_games"] += 1
                ht_status = ht_away_status
                ft_status = ft_away_status

            # Comeback: losing at HT, won or drew FT
            if ht_status == "losing":
                tp["losing_ht_count"] += 1
                if ft_status in ("won", "drew"):
                    tp["comeback_count"] += 1

            # Collapse: winning at HT, drew or lost FT
            if ht_status == "winning":
                tp["winning_ht_count"] += 1
                if ft_status in ("drew", "lost"):
                    tp["collapse_count"] += 1

    # Build final profiles
    profiles = {}
    for team, tp in team_profiles.items():
        hg = tp["home_games"] or 1
        ag = tp["away_games"] or 1

        profiles[team] = {
            "home_avg_1h_goals": round(tp["home_1h_goals"] / hg, 3),
            "home_avg_2h_goals": round(tp["home_2h_goals"] / hg, 3),
            "away_avg_1h_goals": round(tp["away_1h_goals"] / ag, 3),
            "away_avg_2h_goals": round(tp["away_2h_goals"] / ag, 3),
            "comeback_rate": round(tp["comeback_count"] / tp["losing_ht_count"], 3) if tp["losing_ht_count"] > 0 else 0.0,
            "collapse_rate": round(tp["collapse_count"] / tp["winning_ht_count"], 3) if tp["winning_ht_count"] > 0 else 0.0,
            "home_games": tp["home_games"],
            "away_games": tp["away_games"],
        }

    return profiles


def calculate_second_half_strength(league="PL"):
    """
    Calculate a second-half strength multiplier per team.

    Teams that score proportionally more in the 2nd half may be stronger
    than their 1st-half performance suggests (fitness, tactical adjustments).

    Returns dict of team -> second_half_strength multiplier (1.0 = neutral).
    """
    profiles = calculate_half_profiles(league)
    if not profiles:
        return {}

    multipliers = {}
    for team, p in profiles.items():
        total_1h = p["home_avg_1h_goals"] + p["away_avg_1h_goals"]
        total_2h = p["home_avg_2h_goals"] + p["away_avg_2h_goals"]

        if total_1h > 0:
            # Ratio of 2nd half to 1st half scoring
            ratio = total_2h / total_1h
            # Convert to a bounded multiplier: 0.9 to 1.1
            # ratio > 1 means team scores more in 2nd half (stronger)
            multiplier = 1.0 + (ratio - 1.0) * 0.1
            multiplier = max(0.9, min(1.1, multiplier))
        else:
            multiplier = 1.0

        multipliers[team] = round(multiplier, 4)

    return multipliers


def calculate_league_averages(league="PL", season=None):
    """Calculate league-wide average goals per game (home and away)."""
    query = "SELECT * FROM matches WHERE league = ? AND ft_home_goals IS NOT NULL"
    params = [league]
    if season:
        query += " AND season = ?"
        params.append(season)

    matches = db.fetch_all(query, params)
    if not matches:
        return {"home_avg": 1.5, "away_avg": 1.2, "total_matches": 0}

    home_goals = sum(m["ft_home_goals"] for m in matches)
    away_goals = sum(m["ft_away_goals"] for m in matches)
    n = len(matches)

    return {
        "home_avg": home_goals / n,
        "away_avg": away_goals / n,
        "total_matches": n,
    }


def calculate_team_strengths(league="PL", time_decay=True):
    """
    Calculate attack strength and defence weakness for each team.

    Attack Strength = Team's avg goals scored / League avg goals scored
    Defence Weakness = Team's avg goals conceded / League avg goals conceded

    With time decay: recent matches weighted more heavily.
    """
    matches = db.fetch_all(
        """SELECT * FROM matches WHERE league = ? AND ft_home_goals IS NOT NULL
           ORDER BY match_date ASC""",
        [league]
    )
    if not matches:
        return {}

    avgs = calculate_league_averages(league)
    if avgs["total_matches"] < config.POISSON_SETTINGS["min_matches"]:
        return {}

    # Aggregate per team with optional time decay
    team_stats = {}
    now = datetime.utcnow()

    for m in matches:
        home = m["home_team"]
        away = m["away_team"]
        match_date = datetime.strptime(m["match_date"], "%Y-%m-%d")

        # Time decay weight: more recent = higher weight
        if time_decay:
            days_ago = (now - match_date).days
            weight = math.exp(-config.POISSON_SETTINGS["time_decay_weight"] * days_ago)
        else:
            weight = 1.0

        for team in [home, away]:
            if team not in team_stats:
                team_stats[team] = {
                    "home_scored": 0, "home_conceded": 0, "home_games": 0,
                    "away_scored": 0, "away_conceded": 0, "away_games": 0,
                    "total_weight_home": 0, "total_weight_away": 0,
                }

        # Home team stats
        team_stats[home]["home_scored"] += m["ft_home_goals"] * weight
        team_stats[home]["home_conceded"] += m["ft_away_goals"] * weight
        team_stats[home]["home_games"] += 1
        team_stats[home]["total_weight_home"] += weight

        # Away team stats
        team_stats[away]["away_scored"] += m["ft_away_goals"] * weight
        team_stats[away]["away_conceded"] += m["ft_home_goals"] * weight
        team_stats[away]["away_games"] += 1
        team_stats[away]["total_weight_away"] += weight

    # Calculate strengths
    strengths = {}
    for team, stats in team_stats.items():
        wh = stats["total_weight_home"] or 1
        wa = stats["total_weight_away"] or 1

        home_attack = (stats["home_scored"] / wh) / avgs["home_avg"] if avgs["home_avg"] > 0 else 1
        home_defence = (stats["home_conceded"] / wh) / avgs["away_avg"] if avgs["away_avg"] > 0 else 1
        away_attack = (stats["away_scored"] / wa) / avgs["away_avg"] if avgs["away_avg"] > 0 else 1
        away_defence = (stats["away_conceded"] / wa) / avgs["home_avg"] if avgs["home_avg"] > 0 else 1

        strengths[team] = {
            "home_attack": round(home_attack, 4),
            "home_defence": round(home_defence, 4),
            "away_attack": round(away_attack, 4),
            "away_defence": round(away_defence, 4),
            "attack_strength": round((home_attack + away_attack) / 2, 4),
            "defence_weakness": round((home_defence + away_defence) / 2, 4),
            "home_games": stats["home_games"],
            "away_games": stats["away_games"],
        }

    return strengths


def dixon_coles_correction(home_goals, away_goals, home_lambda, away_lambda, rho=-0.13):
    """
    Dixon-Coles correction for low-scoring outcomes.
    Adjusts probabilities for 0-0, 1-0, 0-1, 1-1 where Poisson underestimates correlation.
    rho is typically negative (-0.13 is a common empirical value).
    """
    if home_goals == 0 and away_goals == 0:
        return 1 - home_lambda * away_lambda * rho
    elif home_goals == 1 and away_goals == 0:
        return 1 + away_lambda * rho
    elif home_goals == 0 and away_goals == 1:
        return 1 + home_lambda * rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    else:
        return 1.0


def generate_scoreline_matrix(home_lambda, away_lambda, apply_dc=True):
    """
    Generate full scoreline probability matrix using Poisson distribution.
    Returns a MAX_GOALS x MAX_GOALS matrix of probabilities.
    """
    matrix = np.zeros((MAX_GOALS, MAX_GOALS))

    for i in range(MAX_GOALS):
        for j in range(MAX_GOALS):
            p = poisson.pmf(i, home_lambda) * poisson.pmf(j, away_lambda)
            if apply_dc:
                p *= dixon_coles_correction(i, j, home_lambda, away_lambda)
            matrix[i][j] = p

    # Normalise to ensure probabilities sum to 1
    total = matrix.sum()
    if total > 0:
        matrix = matrix / total

    return matrix


def predict(home_team, away_team, league="PL"):
    """
    Generate match prediction using Poisson + Dixon-Coles model.

    Returns dict with probabilities and full scoreline matrix.
    """
    strengths = calculate_team_strengths(league)
    avgs = calculate_league_averages(league)

    if home_team not in strengths or away_team not in strengths:
        logger.warning("Missing team data for %s vs %s", home_team, away_team)
        return None

    home = strengths[home_team]
    away = strengths[away_team]

    # Get second-half strength multipliers (goal timing weights)
    try:
        sh_multipliers = calculate_second_half_strength(league)
    except Exception:
        sh_multipliers = {}
    home_sh = sh_multipliers.get(home_team, 1.0)
    away_sh = sh_multipliers.get(away_team, 1.0)

    # Expected goals (lambda), adjusted by second-half strength
    home_lambda = avgs["home_avg"] * home["home_attack"] * away["away_defence"] * home_sh
    away_lambda = avgs["away_avg"] * away["away_attack"] * home["home_defence"] * away_sh

    # Clamp to reasonable range
    home_lambda = max(0.2, min(home_lambda, 5.0))
    away_lambda = max(0.2, min(away_lambda, 5.0))

    # Generate scoreline matrix
    matrix = generate_scoreline_matrix(home_lambda, away_lambda)

    # Extract outcome probabilities
    home_win = sum(matrix[i][j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if i > j)
    draw = sum(matrix[i][j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if i == j)
    away_win = sum(matrix[i][j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if i < j)

    # Over/Under 2.5
    over25 = sum(matrix[i][j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if i + j > 2)
    under25 = 1 - over25

    # BTTS (Both Teams To Score)
    btts_yes = sum(matrix[i][j] for i in range(1, MAX_GOALS) for j in range(1, MAX_GOALS))
    btts_no = 1 - btts_yes

    # Most likely scoreline
    max_idx = np.unravel_index(matrix.argmax(), matrix.shape)
    most_likely_score = f"{max_idx[0]}-{max_idx[1]}"

    # Confidence based on how decisive the prediction is
    probs = sorted([home_win, draw, away_win], reverse=True)
    confidence = probs[0] - probs[1]  # gap between top two

    return {
        "home_win": round(float(home_win), 4),
        "draw": round(float(draw), 4),
        "away_win": round(float(away_win), 4),
        "over25": round(float(over25), 4),
        "under25": round(float(under25), 4),
        "btts_yes": round(float(btts_yes), 4),
        "btts_no": round(float(btts_no), 4),
        "home_lambda": round(home_lambda, 3),
        "away_lambda": round(away_lambda, 3),
        "most_likely_score": most_likely_score,
        "confidence": round(float(confidence), 4),
        "scoreline_matrix": matrix.tolist(),
        "details": {
            "home_attack": home["home_attack"],
            "home_defence": home["home_defence"],
            "away_attack": away["away_attack"],
            "away_defence": away["away_defence"],
            "league_home_avg": avgs["home_avg"],
            "league_away_avg": avgs["away_avg"],
            "home_second_half_strength": home_sh,
            "away_second_half_strength": away_sh,
        }
    }
