"""
Ensemble model — combines all individual models with optimised weights.

Blends: Poisson/Dixon-Coles, Elo, XGBoost, Sentiment
into a single probability output.

Supports two blending methods:
1. Weighted average (default fallback)
2. Stacking ensemble via logistic regression meta-model
"""
import logging
import json
import os
import pickle
from datetime import datetime

import numpy as np

from models import poisson as poisson_model
from models import elo as elo_model
from models import xgboost_model
from models import sentiment as sentiment_model
from models import over_under as over_under_model
from database import db
import config

logger = logging.getLogger(__name__)

STACKER_PATH = os.path.join(config.BASE_DIR, "data", "cache", "stacker_model.pkl")

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "app_settings.json")


def _load_weights():
    """Load model weights, preferring optimized weights from settings file."""
    try:
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)
        saved_weights = settings.get("model_weights")
        if saved_weights and isinstance(saved_weights, dict):
            # Validate that all required keys are present
            required = {"poisson", "elo", "xgboost", "sentiment"}
            if required.issubset(saved_weights.keys()):
                logger.debug("Using optimized model weights from settings")
                return saved_weights
    except (FileNotFoundError, json.JSONDecodeError, Exception):
        pass
    return config.MODEL_WEIGHTS


WEIGHTS = _load_weights()


def train_stacker(league="PL"):
    """
    Train a logistic regression meta-model that learns the optimal
    way to combine individual model predictions based on context.

    Features for the stacker:
    - All 4 model probabilities (12 values: H/D/A for each model)
    - Elo difference (context: how close are the teams)
    - Model agreement score (do models agree or disagree)
    - Max probability (how confident is the best model)

    Target: actual match result (0=H, 1=D, 2=A)
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except ImportError:
        logger.error("sklearn not installed, cannot train stacker")
        return None

    # Get historical predictions with actual results
    rows = db.fetch_all(
        """SELECT p.poisson_home, p.poisson_draw, p.poisson_away,
                  p.elo_home, p.elo_draw, p.elo_away,
                  p.xgboost_home, p.xgboost_draw, p.xgboost_away,
                  p.sentiment_home, p.sentiment_draw, p.sentiment_away,
                  m.ft_result,
                  p.home_team, p.away_team
           FROM predictions p
           JOIN matches m ON p.league = m.league
                AND p.home_team = m.home_team
                AND p.away_team = m.away_team
                AND p.match_date = m.match_date
           WHERE p.league = ? AND m.ft_result IS NOT NULL
                 AND p.poisson_home IS NOT NULL
                 AND p.elo_home IS NOT NULL""",
        [league]
    )

    if not rows or len(rows) < 30:
        logger.warning("Not enough settled predictions to train stacker: %d", len(rows) if rows else 0)
        return None

    # Build Elo ratings for elo_diff feature
    try:
        ratings = elo_model.build_ratings(league)
    except Exception:
        ratings = {}

    X_rows = []
    y = []
    result_map = {"H": 0, "D": 1, "A": 2}

    for row in rows:
        features = []

        # 12 model probability features (fill missing with neutral 0.33)
        for model_prefix in ["poisson", "elo", "xgboost", "sentiment"]:
            h = row.get(f"{model_prefix}_home") or 0.33
            d = row.get(f"{model_prefix}_draw") or 0.33
            a = row.get(f"{model_prefix}_away") or 0.33
            features.extend([h, d, a])

        # Elo difference
        home_elo = ratings.get(row["home_team"], {}).get("elo", 1500)
        away_elo = ratings.get(row["away_team"], {}).get("elo", 1500)
        features.append(home_elo - away_elo)

        # Model agreement score: std deviation of home_win predictions across models
        home_probs = []
        for model_prefix in ["poisson", "elo", "xgboost", "sentiment"]:
            h = row.get(f"{model_prefix}_home")
            if h is not None:
                home_probs.append(h)
        agreement = 1.0 - np.std(home_probs) if len(home_probs) > 1 else 0.5
        features.append(agreement)

        # Max probability across all models
        all_probs = []
        for model_prefix in ["poisson", "elo", "xgboost", "sentiment"]:
            for suffix in ["home", "draw", "away"]:
                val = row.get(f"{model_prefix}_{suffix}")
                if val is not None:
                    all_probs.append(val)
        max_prob = max(all_probs) if all_probs else 0.33
        features.append(max_prob)

        X_rows.append(features)
        y.append(result_map.get(row["ft_result"], 1))

    X = np.array(X_rows)
    y = np.array(y)

    # Train with cross-validation to check quality
    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        random_state=42,
    )

    try:
        cv_scores = cross_val_score(model, X, y, cv=min(5, len(X) // 10), scoring="neg_log_loss")
        avg_cv = -np.mean(cv_scores)
        logger.info("Stacker CV log-loss: %.4f (samples: %d)", avg_cv, len(X))
    except Exception as e:
        logger.warning("Stacker CV failed (training anyway): %s", e)

    # Train on all data
    model.fit(X, y)

    # Save the trained stacker
    feature_names = (
        ["poisson_h", "poisson_d", "poisson_a",
         "elo_h", "elo_d", "elo_a",
         "xgboost_h", "xgboost_d", "xgboost_a",
         "sentiment_h", "sentiment_d", "sentiment_a",
         "elo_diff", "model_agreement", "max_prob"]
    )

    os.makedirs(os.path.dirname(STACKER_PATH), exist_ok=True)
    with open(STACKER_PATH, "wb") as f:
        pickle.dump({"model": model, "features": feature_names, "league": league,
                      "trained_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                      "sample_size": len(X)}, f)

    logger.info("Stacker model trained on %d samples, saved to %s", len(X), STACKER_PATH)
    return model


def _stacker_predict(predictions, home_elo, away_elo):
    """
    Use the trained stacker to blend model predictions.
    Returns (home_win, draw, away_win) probabilities or None if stacker unavailable.
    """
    if not os.path.exists(STACKER_PATH):
        return None

    try:
        with open(STACKER_PATH, "rb") as f:
            saved = pickle.load(f)
        model = saved["model"]
    except Exception as e:
        logger.warning("Failed to load stacker model: %s", e)
        return None

    # Build feature vector
    features = []

    for model_name in ["poisson", "elo", "xgboost", "sentiment"]:
        pred = predictions.get(model_name)
        if pred:
            features.extend([pred["home_win"], pred["draw"], pred["away_win"]])
        else:
            features.extend([0.33, 0.33, 0.33])

    # Elo difference
    features.append(home_elo - away_elo)

    # Model agreement
    home_probs = [p["home_win"] for p in predictions.values() if "home_win" in p]
    agreement = 1.0 - np.std(home_probs) if len(home_probs) > 1 else 0.5
    features.append(agreement)

    # Max probability
    all_probs = []
    for p in predictions.values():
        for key in ["home_win", "draw", "away_win"]:
            if key in p:
                all_probs.append(p[key])
    max_prob = max(all_probs) if all_probs else 0.33
    features.append(max_prob)

    try:
        X = np.array([features])
        probs = model.predict_proba(X)[0]
        return {
            "home_win": float(probs[0]),
            "draw": float(probs[1]),
            "away_win": float(probs[2]),
        }
    except Exception as e:
        logger.warning("Stacker prediction failed: %s", e)
        return None


def predict(home_team, away_team, league="PL", match_date=None):
    """
    Generate ensemble prediction by combining all models.
    Falls back gracefully if individual models fail.
    """
    # Reload weights each time to pick up optimized values
    weights = _load_weights()

    predictions = {}
    model_details = {}
    available_weight = 0

    # Poisson + Dixon-Coles
    try:
        p = poisson_model.predict(home_team, away_team, league)
        if p:
            predictions["poisson"] = p
            available_weight += weights["poisson"]
            model_details["poisson"] = {
                "weight": weights["poisson"],
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
            available_weight += weights["elo"]
            model_details["elo"] = {
                "weight": weights["elo"],
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
            available_weight += weights["xgboost"]
            model_details["xgboost"] = {
                "weight": weights["xgboost"],
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
            available_weight += weights["sentiment"]
            model_details["sentiment"] = {
                "weight": weights["sentiment"],
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

    # Get Elo values for stacker context
    elo_pred = predictions.get("elo")
    home_elo = elo_pred["details"].get("home_elo", 1500) if elo_pred and "details" in elo_pred else 1500
    away_elo = elo_pred["details"].get("away_elo", 1500) if elo_pred and "details" in elo_pred else 1500

    # Try stacker first, fall back to weighted average
    blend_method = "weighted"
    stacker_result = None
    try:
        stacker_result = _stacker_predict(predictions, home_elo, away_elo)
    except Exception as e:
        logger.debug("Stacker unavailable, using weighted average: %s", e)

    if stacker_result:
        home_win = stacker_result["home_win"]
        draw = stacker_result["draw"]
        away_win = stacker_result["away_win"]
        blend_method = "stacker"
    else:
        # Weighted blend (re-normalise weights based on available models)
        home_win = 0
        draw = 0
        away_win = 0

        for model_name, pred in predictions.items():
            w = weights[model_name] / available_weight  # re-normalised weight
            home_win += pred["home_win"] * w
            draw += pred["draw"] * w
            away_win += pred["away_win"] * w

    # Ensure probabilities sum to 1
    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total

    # Over/Under and BTTS from Poisson (only Poisson can do scoreline matrix)
    poisson_over25 = predictions.get("poisson", {}).get("over25")
    btts = predictions.get("poisson", {}).get("btts_yes")
    scoreline_matrix = predictions.get("poisson", {}).get("scoreline_matrix")
    most_likely_score = predictions.get("poisson", {}).get("most_likely_score")

    # Blend Over/Under 2.5 with dedicated O/U model (50/50 blend)
    over25 = poisson_over25
    try:
        ou_pred = over_under_model.predict_over_under(home_team, away_team, league, match_date)
        if ou_pred and ou_pred.get("over25_prob") is not None:
            if poisson_over25 is not None:
                over25 = poisson_over25 * 0.5 + ou_pred["over25_prob"] * 0.5
            else:
                over25 = ou_pred["over25_prob"]
    except Exception as e:
        logger.debug("Over/Under model unavailable, using Poisson only: %s", e)

    # Confidence: weighted average of model confidences
    confidences = []
    for model_name, pred in predictions.items():
        if "confidence" in pred:
            w = weights[model_name] / available_weight if available_weight > 0 else 0.25
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
        "method": blend_method,
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

    # Combined Edge Signal: when underdog has edge, draw+underdog combined is strong
    # "If our model says the underdog is tougher than market thinks,
    #  they're likely to compete — draw OR win covers the model's insight"
    combined_signals = []

    # Check if away team has edge (underdog scenario when home is favoured)
    away_vb = [v for v in value_bets if v["bet_type"] == "away_win"]
    if away_vb and prediction["home_win"] > prediction["away_win"]:
        # Away is underdog but has value — combine draw + away
        combined_prob = prediction["draw"] + prediction["away_win"]
        # Find best bookmaker implied for home (to compare)
        home_implied = 0
        for bookie in odds_data.get("all_bookmakers", []):
            h = bookie.get("home_odds", 0) or 0
            if h > 0:
                home_implied = max(home_implied, 1 / h)
        lay_home_prob = 1 - home_implied if home_implied > 0 else 0
        combined_edge = (combined_prob - lay_home_prob) * 100 if lay_home_prob > 0 else 0

        combined_signals.append({
            "type": "lay_favourite",
            "label": "Underdog Edge: Draw + Away Win",
            "combined_probability": round(combined_prob, 4),
            "combined_percent": round(combined_prob * 100, 1),
            "market_probability": round(lay_home_prob, 4),
            "edge": round(combined_edge, 1),
            "insight": f"Model says {prediction['away_win']*100:.0f}% away win + {prediction['draw']*100:.0f}% draw = {combined_prob*100:.0f}% chance favourite doesn't win",
            "away_edge": away_vb[0]["edge_percent"],
        })

    # Check if home team has edge but is actually underdog (away favoured)
    home_vb = [v for v in value_bets if v["bet_type"] == "home_win"]
    if home_vb and prediction["away_win"] > prediction["home_win"]:
        combined_prob = prediction["draw"] + prediction["home_win"]
        away_implied = 0
        for bookie in odds_data.get("all_bookmakers", []):
            a = bookie.get("away_odds", 0) or 0
            if a > 0:
                away_implied = max(away_implied, 1 / a)
        lay_away_prob = 1 - away_implied if away_implied > 0 else 0
        combined_edge = (combined_prob - lay_away_prob) * 100 if lay_away_prob > 0 else 0

        combined_signals.append({
            "type": "lay_favourite",
            "label": "Underdog Edge: Draw + Home Win",
            "combined_probability": round(combined_prob, 4),
            "combined_percent": round(combined_prob * 100, 1),
            "market_probability": round(lay_away_prob, 4),
            "edge": round(combined_edge, 1),
            "insight": f"Model says {prediction['home_win']*100:.0f}% home win + {prediction['draw']*100:.0f}% draw = {combined_prob*100:.0f}% chance favourite doesn't win",
            "home_edge": home_vb[0]["edge_percent"],
        })

    # Attach combined signals to value bets
    for vb in value_bets:
        vb["combined_signals"] = combined_signals

    return sorted(value_bets, key=lambda x: x["edge_percent"], reverse=True)


def optimize_weights(league="PL"):
    """
    Optimize ensemble model weights by minimizing Brier score
    on historical predictions that have actual results.

    Uses scipy.optimize to find the best weight combination.
    Saves optimized weights to data/app_settings.json under key "model_weights".
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        logger.error("scipy not installed, falling back to grid search")
        return _optimize_weights_grid(league)

    # Get all historical predictions with actual results
    rows = db.fetch_all(
        """SELECT p.poisson_home, p.poisson_draw, p.poisson_away,
                  p.elo_home, p.elo_draw, p.elo_away,
                  p.xgboost_home, p.xgboost_draw, p.xgboost_away,
                  p.sentiment_home, p.sentiment_draw, p.sentiment_away,
                  m.ft_result
           FROM predictions p
           JOIN matches m ON p.league = m.league
                AND p.home_team = m.home_team
                AND p.away_team = m.away_team
                AND p.match_date = m.match_date
           WHERE p.league = ? AND m.ft_result IS NOT NULL
                 AND p.poisson_home IS NOT NULL
                 AND p.elo_home IS NOT NULL""",
        [league]
    )

    if not rows or len(rows) < 20:
        logger.warning("Not enough settled predictions to optimize weights: %d", len(rows) if rows else 0)
        return None

    def brier_score(weights_raw):
        """Calculate ensemble Brier score for a given set of weights."""
        # Softmax to ensure weights sum to 1 and are positive
        import numpy as np
        w = np.exp(weights_raw) / np.sum(np.exp(weights_raw))
        w_poisson, w_elo, w_xgboost, w_sentiment = w

        total_brier = 0.0
        count = 0

        for row in rows:
            # Collect available model predictions
            model_preds = {}
            if row.get("poisson_home") is not None:
                model_preds["poisson"] = (row["poisson_home"], row["poisson_draw"], row["poisson_away"])
            if row.get("elo_home") is not None:
                model_preds["elo"] = (row["elo_home"], row["elo_draw"], row["elo_away"])
            if row.get("xgboost_home") is not None:
                model_preds["xgboost"] = (row["xgboost_home"], row["xgboost_draw"], row["xgboost_away"])
            if row.get("sentiment_home") is not None:
                model_preds["sentiment"] = (row["sentiment_home"], row["sentiment_draw"], row["sentiment_away"])

            if not model_preds:
                continue

            weight_map = {"poisson": w_poisson, "elo": w_elo, "xgboost": w_xgboost, "sentiment": w_sentiment}

            avail_weight = sum(weight_map[m] for m in model_preds)
            if avail_weight <= 0:
                continue

            # Blend
            h = sum(model_preds[m][0] * weight_map[m] / avail_weight for m in model_preds)
            d = sum(model_preds[m][1] * weight_map[m] / avail_weight for m in model_preds)
            a = sum(model_preds[m][2] * weight_map[m] / avail_weight for m in model_preds)

            # Normalize
            total = h + d + a
            if total > 0:
                h /= total
                d /= total
                a /= total

            # Actual result
            actual = row["ft_result"]
            actual_h = 1.0 if actual == "H" else 0.0
            actual_d = 1.0 if actual == "D" else 0.0
            actual_a = 1.0 if actual == "A" else 0.0

            total_brier += (h - actual_h) ** 2 + (d - actual_d) ** 2 + (a - actual_a) ** 2
            count += 1

        return total_brier / count if count > 0 else 999.0

    import numpy as np

    # Start from current weights (in log space for softmax parameterization)
    current = _load_weights()
    x0 = np.log([current["poisson"], current["elo"], current["xgboost"], current["sentiment"]])

    result = minimize(brier_score, x0, method="Nelder-Mead",
                      options={"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-6})

    # Convert back from log space
    optimal_raw = result.x
    optimal_w = np.exp(optimal_raw) / np.sum(np.exp(optimal_raw))

    optimized_weights = {
        "poisson": round(float(optimal_w[0]), 4),
        "elo": round(float(optimal_w[1]), 4),
        "xgboost": round(float(optimal_w[2]), 4),
        "sentiment": round(float(optimal_w[3]), 4),
    }

    optimal_brier = brier_score(optimal_raw)

    # Save to settings file
    try:
        settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "app_settings.json")
        try:
            with open(settings_path, "r") as f:
                settings = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            settings = {}

        settings["model_weights"] = optimized_weights
        settings["model_weights_brier"] = round(optimal_brier, 4)
        settings["model_weights_optimized_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        settings["model_weights_sample_size"] = len(rows)

        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)

        logger.info("Optimized weights saved: %s (Brier: %.4f, samples: %d)",
                     optimized_weights, optimal_brier, len(rows))
    except Exception as e:
        logger.error("Failed to save optimized weights: %s", e)

    return {
        "weights": optimized_weights,
        "brier": round(optimal_brier, 4),
        "sample_size": len(rows),
    }


def _optimize_weights_grid(league="PL"):
    """Fallback grid search for weight optimization when scipy is not available."""
    rows = db.fetch_all(
        """SELECT p.poisson_home, p.poisson_draw, p.poisson_away,
                  p.elo_home, p.elo_draw, p.elo_away,
                  p.xgboost_home, p.xgboost_draw, p.xgboost_away,
                  p.sentiment_home, p.sentiment_draw, p.sentiment_away,
                  m.ft_result
           FROM predictions p
           JOIN matches m ON p.league = m.league
                AND p.home_team = m.home_team
                AND p.away_team = m.away_team
                AND p.match_date = m.match_date
           WHERE p.league = ? AND m.ft_result IS NOT NULL
                 AND p.poisson_home IS NOT NULL
                 AND p.elo_home IS NOT NULL""",
        [league]
    )

    if not rows or len(rows) < 20:
        logger.warning("Not enough data for grid search weight optimization")
        return None

    best_brier = 999.0
    best_weights = None

    # Grid search with step 0.05
    step = 0.05
    import numpy as np
    for w_p in np.arange(0.10, 0.60, step):
        for w_e in np.arange(0.10, 0.50, step):
            for w_x in np.arange(0.10, 0.50, step):
                w_s = 1.0 - w_p - w_e - w_x
                if w_s < 0.02 or w_s > 0.30:
                    continue

                total_brier = 0.0
                count = 0
                weight_map = {"poisson": w_p, "elo": w_e, "xgboost": w_x, "sentiment": w_s}

                for row in rows:
                    model_preds = {}
                    if row.get("poisson_home") is not None:
                        model_preds["poisson"] = (row["poisson_home"], row["poisson_draw"], row["poisson_away"])
                    if row.get("elo_home") is not None:
                        model_preds["elo"] = (row["elo_home"], row["elo_draw"], row["elo_away"])
                    if row.get("xgboost_home") is not None:
                        model_preds["xgboost"] = (row["xgboost_home"], row["xgboost_draw"], row["xgboost_away"])
                    if row.get("sentiment_home") is not None:
                        model_preds["sentiment"] = (row["sentiment_home"], row["sentiment_draw"], row["sentiment_away"])

                    if not model_preds:
                        continue

                    avail_w = sum(weight_map[m] for m in model_preds)
                    if avail_w <= 0:
                        continue

                    h = sum(model_preds[m][0] * weight_map[m] / avail_w for m in model_preds)
                    d = sum(model_preds[m][1] * weight_map[m] / avail_w for m in model_preds)
                    a = sum(model_preds[m][2] * weight_map[m] / avail_w for m in model_preds)

                    total_p = h + d + a
                    if total_p > 0:
                        h /= total_p
                        d /= total_p
                        a /= total_p

                    actual = row["ft_result"]
                    total_brier += (h - (1 if actual == "H" else 0)) ** 2 + \
                                   (d - (1 if actual == "D" else 0)) ** 2 + \
                                   (a - (1 if actual == "A" else 0)) ** 2
                    count += 1

                if count > 0:
                    avg_brier = total_brier / count
                    if avg_brier < best_brier:
                        best_brier = avg_brier
                        best_weights = {
                            "poisson": round(float(w_p), 4),
                            "elo": round(float(w_e), 4),
                            "xgboost": round(float(w_x), 4),
                            "sentiment": round(float(w_s), 4),
                        }

    if best_weights:
        # Save to settings
        try:
            settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "app_settings.json")
            try:
                with open(settings_path, "r") as f:
                    settings = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                settings = {}

            settings["model_weights"] = best_weights
            settings["model_weights_brier"] = round(best_brier, 4)
            settings["model_weights_optimized_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            settings["model_weights_sample_size"] = len(rows)

            os.makedirs(os.path.dirname(settings_path), exist_ok=True)
            with open(settings_path, "w") as f:
                json.dump(settings, f, indent=2)

            logger.info("Grid search optimized weights: %s (Brier: %.4f)", best_weights, best_brier)
        except Exception as e:
            logger.error("Failed to save grid search weights: %s", e)

        return {
            "weights": best_weights,
            "brier": round(best_brier, 4),
            "sample_size": len(rows),
        }

    return None
