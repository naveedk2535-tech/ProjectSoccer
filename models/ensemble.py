"""
Ensemble model — combines all individual models with optimised weights.

Blends: Poisson/Dixon-Coles, Elo, XGBoost, Sentiment
into a single probability output.
"""
import logging
import json
from datetime import datetime

from models import poisson as poisson_model
from models import elo as elo_model
from models import xgboost_model
from models import sentiment as sentiment_model
from database import db
import config

logger = logging.getLogger(__name__)

WEIGHTS = config.MODEL_WEIGHTS


def predict(home_team, away_team, league="PL", match_date=None):
    """
    Generate ensemble prediction by combining all models.
    Falls back gracefully if individual models fail.
    """
    predictions = {}
    model_details = {}
    available_weight = 0

    # Poisson + Dixon-Coles
    try:
        p = poisson_model.predict(home_team, away_team, league)
        if p:
            predictions["poisson"] = p
            available_weight += WEIGHTS["poisson"]
            model_details["poisson"] = {
                "weight": WEIGHTS["poisson"],
                "home_win": p["home_win"],
                "draw": p["draw"],
                "away_win": p["away_win"],
                "home_lambda": p.get("home_lambda"),
                "away_lambda": p.get("away_lambda"),
                "most_likely_score": p.get("most_likely_score"),
            }
    except Exception as e:
        logger.error("Poisson model failed: %s", e)

    # Elo
    try:
        e = elo_model.predict(home_team, away_team, league)
        if e:
            predictions["elo"] = e
            available_weight += WEIGHTS["elo"]
            model_details["elo"] = {
                "weight": WEIGHTS["elo"],
                "home_win": e["home_win"],
                "draw": e["draw"],
                "away_win": e["away_win"],
                "elo_diff": e["details"].get("elo_difference"),
                "home_form": e["details"].get("home_form_last5"),
                "away_form": e["details"].get("away_form_last5"),
            }
    except Exception as e:
        logger.error("Elo model failed: %s", e)

    # XGBoost
    try:
        x = xgboost_model.predict(home_team, away_team, league, match_date)
        if x:
            predictions["xgboost"] = x
            available_weight += WEIGHTS["xgboost"]
            model_details["xgboost"] = {
                "weight": WEIGHTS["xgboost"],
                "home_win": x["home_win"],
                "draw": x["draw"],
                "away_win": x["away_win"],
                "top_features": x["details"].get("top_features", []),
            }
    except Exception as e:
        logger.error("XGBoost model failed: %s", e)

    # Sentiment
    try:
        s = sentiment_model.predict(home_team, away_team, league)
        if s:
            predictions["sentiment"] = s
            available_weight += WEIGHTS["sentiment"]
            model_details["sentiment"] = {
                "weight": WEIGHTS["sentiment"],
                "home_win": s["home_win"],
                "draw": s["draw"],
                "away_win": s["away_win"],
                "home_signal": s["details"]["home_sentiment"]["signal"],
                "away_signal": s["details"]["away_sentiment"]["signal"],
            }
    except Exception as e:
        logger.error("Sentiment model failed: %s", e)

    if not predictions:
        logger.error("All models failed for %s vs %s", home_team, away_team)
        return None

    # Weighted blend (re-normalise weights based on available models)
    home_win = 0
    draw = 0
    away_win = 0

    for model_name, pred in predictions.items():
        w = WEIGHTS[model_name] / available_weight  # re-normalised weight
        home_win += pred["home_win"] * w
        draw += pred["draw"] * w
        away_win += pred["away_win"] * w

    # Ensure probabilities sum to 1
    total = home_win + draw + away_win
    home_win /= total
    draw /= total
    away_win /= total

    # Over/Under and BTTS from Poisson (only Poisson can do scoreline matrix)
    over25 = predictions.get("poisson", {}).get("over25")
    btts = predictions.get("poisson", {}).get("btts_yes")
    scoreline_matrix = predictions.get("poisson", {}).get("scoreline_matrix")
    most_likely_score = predictions.get("poisson", {}).get("most_likely_score")

    # Confidence: weighted average of model confidences
    confidences = []
    for model_name, pred in predictions.items():
        if "confidence" in pred:
            w = WEIGHTS[model_name] / available_weight
            confidences.append(pred["confidence"] * w)
    confidence = sum(confidences) if confidences else 0.5

    # Determine predicted outcome
    if home_win >= draw and home_win >= away_win:
        predicted = "H"
    elif away_win >= draw:
        predicted = "A"
    else:
        predicted = "D"

    result = {
        "home_win": round(home_win, 4),
        "draw": round(draw, 4),
        "away_win": round(away_win, 4),
        "predicted_outcome": predicted,
        "confidence": round(confidence, 4),
        "over25": round(over25, 4) if over25 else None,
        "btts": round(btts, 4) if btts else None,
        "most_likely_score": most_likely_score,
        "scoreline_matrix": scoreline_matrix,
        "models_used": list(predictions.keys()),
        "models_available": len(predictions),
        "model_details": model_details,
    }

    # Convert to implied odds
    result["implied_odds"] = {
        "home": round(1 / home_win, 2) if home_win > 0 else None,
        "draw": round(1 / draw, 2) if draw > 0 else None,
        "away": round(1 / away_win, 2) if away_win > 0 else None,
    }

    return result


def calculate_value(prediction, odds_data):
    """
    Compare model prediction against bookmaker odds to find value.
    Returns value bets with edge calculation.
    """
    if not prediction or not odds_data:
        return []

    value_bets = []
    settings = config.VALUE_BET_SETTINGS

    bet_types = [
        ("home_win", "home_odds", "home_implied"),
        ("draw", "draw_odds", "draw_implied"),
        ("away_win", "away_odds", "away_implied"),
    ]

    for prob_key, odds_key, implied_key in bet_types:
        model_prob = prediction[prob_key]

        # Find best odds across bookmakers
        best_odds = 0
        best_bookie = ""
        for bookie in odds_data.get("all_bookmakers", []):
            o = bookie.get(odds_key, 0) or 0
            if o > best_odds:
                best_odds = o
                best_bookie = bookie.get("bookmaker", "")

        if best_odds < settings["min_odds"] or best_odds > settings["max_odds"]:
            continue

        implied_prob = 1 / best_odds if best_odds > 0 else 1
        edge = (model_prob - implied_prob) * 100

        if edge >= settings["min_edge_percent"]:
            # Kelly Criterion: f = (bp - q) / b
            # where b = odds - 1, p = model prob, q = 1 - p
            b = best_odds - 1
            q = 1 - model_prob
            kelly_full = (b * model_prob - q) / b if b > 0 else 0
            kelly_stake = max(0, kelly_full * settings["kelly_fraction"])
            kelly_stake = min(kelly_stake, settings["max_stake_percent"] / 100)

            value_bets.append({
                "bet_type": prob_key,
                "model_probability": round(model_prob, 4),
                "best_bookmaker": best_bookie,
                "best_odds": best_odds,
                "implied_probability": round(implied_prob, 4),
                "edge_percent": round(edge, 2),
                "kelly_stake": round(kelly_stake * 100, 2),  # as percentage
                "confidence": prediction["confidence"],
            })

    return sorted(value_bets, key=lambda x: x["edge_percent"], reverse=True)
