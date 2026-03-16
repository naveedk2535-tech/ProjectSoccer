"""
Self-Diagnosis: Automatic Model Monitoring.

Tracks rolling performance metrics and flags degradation
so the system can recommend retraining.
"""
import logging
from datetime import datetime

from database import db

logger = logging.getLogger(__name__)


def calculate_rolling_performance(league="PL", window=50):
    """
    Calculate rolling model performance metrics.

    Gets last N predictions that have been settled (match played),
    compares prediction vs actual result, and returns metrics.
    """
    try:
        # Get settled predictions: predictions where the match has a result
        rows = db.fetch_all(
            """SELECT p.ensemble_home, p.ensemble_draw, p.ensemble_away,
                      p.confidence, m.ft_result
               FROM predictions p
               JOIN matches m ON p.league = m.league
                    AND p.home_team = m.home_team
                    AND p.away_team = m.away_team
                    AND p.match_date = m.match_date
               WHERE p.league = ? AND m.ft_result IS NOT NULL
               ORDER BY m.match_date DESC
               LIMIT ?""",
            [league, window]
        )

        if not rows:
            return {
                "rolling_accuracy": None,
                "rolling_brier": None,
                "rolling_roi": None,
                "sample_size": 0,
                "degraded": False,
                "message": "No settled predictions available",
            }

        correct = 0
        total_brier = 0.0
        total_roi = 0.0

        for row in rows:
            h = row.get("ensemble_home") or 0
            d = row.get("ensemble_draw") or 0
            a = row.get("ensemble_away") or 0
            actual = row.get("ft_result")

            # Determine predicted outcome
            probs = {"H": h, "D": d, "A": a}
            predicted = max(probs, key=probs.get)

            if predicted == actual:
                correct += 1

            # Brier score components
            actual_h = 1.0 if actual == "H" else 0.0
            actual_d = 1.0 if actual == "D" else 0.0
            actual_a = 1.0 if actual == "A" else 0.0
            total_brier += (h - actual_h) ** 2 + (d - actual_d) ** 2 + (a - actual_a) ** 2

            # Simple ROI: if we bet on the predicted outcome at fair odds
            predicted_prob = probs.get(predicted, 0.33)
            if predicted_prob > 0:
                fair_odds = 1.0 / predicted_prob
                if predicted == actual:
                    total_roi += (fair_odds - 1.0)  # profit
                else:
                    total_roi -= 1.0  # loss

        n = len(rows)
        rolling_accuracy = round(correct / n, 4) if n > 0 else None
        rolling_brier = round(total_brier / n, 4) if n > 0 else None
        rolling_roi = round(total_roi / n * 100, 2) if n > 0 else None

        # Get historical average for comparison
        hist = db.fetch_one(
            """SELECT AVG(accuracy) as avg_acc, AVG(brier_score) as avg_brier
               FROM model_performance
               WHERE league = ? AND model_name = 'ensemble'""",
            [league]
        )

        degraded = False
        message = "Performing normally"

        if hist and hist.get("avg_brier") and rolling_brier is not None:
            hist_brier = hist["avg_brier"]
            if hist_brier > 0 and rolling_brier > hist_brier * 1.10:
                degraded = True
                message = f"Brier score degraded: {rolling_brier:.4f} vs historical {hist_brier:.4f}"

        return {
            "rolling_accuracy": rolling_accuracy,
            "rolling_brier": rolling_brier,
            "rolling_roi": rolling_roi,
            "sample_size": n,
            "degraded": degraded,
            "message": message,
        }

    except Exception as e:
        logger.error("Error calculating rolling performance: %s", e)
        return {
            "rolling_accuracy": None,
            "rolling_brier": None,
            "rolling_roi": None,
            "sample_size": 0,
            "degraded": False,
            "message": f"Error: {str(e)}",
        }


def get_model_health():
    """
    Return health status for each model component.

    Checks each model's recent accuracy vs its historical average.
    Returns: {model_name: {status, accuracy, historical_accuracy, ...}}
    """
    models = ["poisson", "elo", "xgboost", "sentiment", "ensemble"]
    health = {}

    for model_name in models:
        try:
            # Get historical performance
            hist = db.fetch_one(
                """SELECT AVG(accuracy) as avg_acc, AVG(brier_score) as avg_brier,
                          COUNT(*) as seasons
                   FROM model_performance
                   WHERE model_name = ?""",
                [model_name]
            )

            hist_acc = hist["avg_acc"] if hist and hist.get("avg_acc") else None
            hist_brier = hist["avg_brier"] if hist and hist.get("avg_brier") else None
            seasons = hist["seasons"] if hist else 0

            # Get recent predictions for this model
            col_map = {
                "poisson": ("poisson_home", "poisson_draw", "poisson_away"),
                "elo": ("elo_home", "elo_draw", "elo_away"),
                "xgboost": ("xgboost_home", "xgboost_draw", "xgboost_away"),
                "sentiment": ("sentiment_home", "sentiment_draw", "sentiment_away"),
                "ensemble": ("ensemble_home", "ensemble_draw", "ensemble_away"),
            }

            cols = col_map.get(model_name, col_map["ensemble"])

            rows = db.fetch_all(
                f"""SELECT p.{cols[0]} as ph, p.{cols[1]} as pd, p.{cols[2]} as pa,
                           m.ft_result
                    FROM predictions p
                    JOIN matches m ON p.league = m.league
                         AND p.home_team = m.home_team
                         AND p.away_team = m.away_team
                         AND p.match_date = m.match_date
                    WHERE m.ft_result IS NOT NULL
                          AND p.{cols[0]} IS NOT NULL
                    ORDER BY m.match_date DESC
                    LIMIT 50"""
            )

            if not rows:
                health[model_name] = {
                    "status": "unknown",
                    "recent_accuracy": None,
                    "historical_accuracy": hist_acc,
                    "recent_brier": None,
                    "historical_brier": hist_brier,
                    "sample_size": 0,
                    "message": "No settled predictions",
                }
                continue

            correct = 0
            total_brier = 0.0
            for row in rows:
                h = row.get("ph") or 0
                d = row.get("pd") or 0
                a = row.get("pa") or 0
                actual = row.get("ft_result")

                probs = {"H": h, "D": d, "A": a}
                predicted = max(probs, key=probs.get)
                if predicted == actual:
                    correct += 1

                actual_h = 1.0 if actual == "H" else 0.0
                actual_d = 1.0 if actual == "D" else 0.0
                actual_a = 1.0 if actual == "A" else 0.0
                total_brier += (h - actual_h) ** 2 + (d - actual_d) ** 2 + (a - actual_a) ** 2

            n = len(rows)
            recent_acc = round(correct / n, 4) if n > 0 else None
            recent_brier = round(total_brier / n, 4) if n > 0 else None

            # Determine status
            status = "healthy"
            message = "Performing within expected range"

            if hist_brier and recent_brier is not None:
                if recent_brier > hist_brier * 1.20:
                    status = "critical"
                    message = f"Brier score {recent_brier:.3f} is >20% worse than historical {hist_brier:.3f}"
                elif recent_brier > hist_brier * 1.10:
                    status = "degrading"
                    message = f"Brier score {recent_brier:.3f} is >10% worse than historical {hist_brier:.3f}"
            elif recent_brier is not None and recent_brier > 0.7:
                status = "degrading"
                message = f"Brier score {recent_brier:.3f} is above 0.7 threshold"

            health[model_name] = {
                "status": status,
                "recent_accuracy": recent_acc,
                "historical_accuracy": round(hist_acc, 4) if hist_acc else None,
                "recent_brier": recent_brier,
                "historical_brier": round(hist_brier, 4) if hist_brier else None,
                "sample_size": n,
                "message": message,
            }

        except Exception as e:
            logger.error("Error checking health for %s: %s", model_name, e)
            health[model_name] = {
                "status": "unknown",
                "recent_accuracy": None,
                "historical_accuracy": None,
                "recent_brier": None,
                "historical_brier": None,
                "sample_size": 0,
                "message": f"Error: {str(e)}",
            }

    return health


def should_retrain():
    """
    Determine if XGBoost needs retraining.

    If recent Brier score is > 10% worse than historical, recommend retrain.
    Returns: {retrain: bool, reason: str}
    """
    try:
        health = get_model_health()
        xgb_health = health.get("xgboost", {})

        status = xgb_health.get("status", "unknown")
        recent_brier = xgb_health.get("recent_brier")
        hist_brier = xgb_health.get("historical_brier")

        if status == "unknown" or recent_brier is None:
            return {
                "retrain": False,
                "reason": "Not enough data to determine if retraining is needed",
            }

        if status in ("degrading", "critical"):
            return {
                "retrain": True,
                "reason": xgb_health.get("message", "Performance degraded"),
            }

        # Also check if it's been a long time since last training
        # (model file modification time could be checked but we keep it simple)
        return {
            "retrain": False,
            "reason": f"Model healthy. Brier: {recent_brier}",
        }

    except Exception as e:
        logger.error("Error in should_retrain: %s", e)
        return {
            "retrain": False,
            "reason": f"Error checking retrain status: {str(e)}",
        }
