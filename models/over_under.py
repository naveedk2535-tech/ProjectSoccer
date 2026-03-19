"""
Dedicated Over/Under 2.5 Goals Prediction Model.

Uses a separate XGBoost binary classifier trained on features
specifically relevant to total goals: rolling goals scored/conceded,
corners, over25 historical rates, Pinnacle O/U implied, referee avg goals.

Complements the Poisson model's over/under prediction with a
machine-learning approach.
"""
import logging
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from database import db
from models import elo as elo_model
from models import poisson as poisson_model
import config

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(config.BASE_DIR, "data", "cache", "over_under_model.pkl")


def extract_ou_features(home_team, away_team, league="PL", match_date=None):
    """
    Extract features specifically relevant to total goals prediction.
    """
    features = {}

    # 1-4. Rolling 10-match goals scored and conceded for each team
    for team, prefix in [(home_team, "home"), (away_team, "away")]:
        try:
            rolling = db.fetch_all(
                """SELECT ft_home_goals, ft_away_goals, home_team
                   FROM matches WHERE league = ? AND (home_team = ? OR away_team = ?)
                   AND ft_home_goals IS NOT NULL
                   ORDER BY match_date DESC LIMIT 10""",
                [league, team, team]
            )
            if rolling and len(rolling) >= 3:
                scored = sum(r["ft_home_goals"] if r["home_team"] == team else r["ft_away_goals"] for r in rolling)
                conceded = sum(r["ft_away_goals"] if r["home_team"] == team else r["ft_home_goals"] for r in rolling)
                features[f"{prefix}_rolling10_scored"] = round(scored / len(rolling), 2)
                features[f"{prefix}_rolling10_conceded"] = round(conceded / len(rolling), 2)
            else:
                features[f"{prefix}_rolling10_scored"] = 1.3
                features[f"{prefix}_rolling10_conceded"] = 1.3
        except Exception:
            features[f"{prefix}_rolling10_scored"] = 1.3
            features[f"{prefix}_rolling10_conceded"] = 1.3

    # 5-6. Average corners per team
    for team, prefix in [(home_team, "home"), (away_team, "away")]:
        try:
            corners = db.fetch_one(
                """SELECT AVG(CASE WHEN home_team = ? THEN home_corners
                               WHEN away_team = ? THEN away_corners END) as avg_corners
                   FROM matches WHERE league = ? AND (home_team = ? OR away_team = ?)
                   AND home_corners IS NOT NULL""",
                [team, team, league, team, team]
            )
            features[f"{prefix}_avg_corners"] = round(corners["avg_corners"], 1) if corners and corners["avg_corners"] else 5.0
        except Exception:
            features[f"{prefix}_avg_corners"] = 5.0

    # 7-8. Over 2.5 historical rate per team
    for team, prefix in [(home_team, "home"), (away_team, "away")]:
        try:
            ou = db.fetch_one(
                """SELECT AVG(CASE WHEN (ft_home_goals + ft_away_goals) > 2 THEN 1.0 ELSE 0.0 END) as over25_rate
                   FROM matches WHERE league = ? AND (home_team = ? OR away_team = ?)
                   AND ft_home_goals IS NOT NULL""",
                [league, team, team]
            )
            features[f"{prefix}_over25_rate"] = round(ou["over25_rate"], 3) if ou and ou["over25_rate"] else 0.50
        except Exception:
            features[f"{prefix}_over25_rate"] = 0.50

    # 9. Referee average goals (if available)
    referee_avg_goals = 2.7  # default
    if match_date:
        try:
            fixture = db.fetch_one(
                """SELECT referee FROM fixtures
                   WHERE league = ? AND home_team = ? AND away_team = ?
                   AND DATE(match_date) = DATE(?)""",
                [league, home_team, away_team, match_date]
            )
            referee_name = fixture.get("referee") if fixture else None
            if not referee_name:
                match_row = db.fetch_one(
                    """SELECT referee FROM matches
                       WHERE league = ? AND home_team = ? AND away_team = ?
                       AND match_date = ?""",
                    [league, home_team, away_team, match_date]
                )
                referee_name = match_row.get("referee") if match_row else None

            if referee_name:
                ref_stats = db.fetch_one(
                    """SELECT avg_total_goals FROM referee_stats
                       WHERE referee = ? AND league = ?""",
                    [referee_name, league]
                )
                if ref_stats and ref_stats.get("avg_total_goals") is not None:
                    referee_avg_goals = ref_stats["avg_total_goals"]
        except Exception:
            pass
    features["referee_avg_goals"] = referee_avg_goals

    # 10-11. Poisson attack/defence strengths
    try:
        strengths = poisson_model.calculate_team_strengths(league)
        home_s = strengths.get(home_team, {})
        away_s = strengths.get(away_team, {})
        features["home_attack"] = home_s.get("home_attack", 1.0)
        features["away_attack"] = away_s.get("away_attack", 1.0)
        features["home_defence"] = home_s.get("home_defence", 1.0)
        features["away_defence"] = away_s.get("away_defence", 1.0)
    except Exception:
        features["home_attack"] = 1.0
        features["away_attack"] = 1.0
        features["home_defence"] = 1.0
        features["away_defence"] = 1.0

    # 12-13. Pinnacle O/U implied (from historical averages)
    for team, prefix, side in [(home_team, "home", "home"), (away_team, "away", "away")]:
        try:
            pin = db.fetch_one(
                f"""SELECT AVG(1.0 / pinnacle_{side}) as avg_implied
                   FROM matches WHERE league = ? AND {side}_team = ?
                   AND pinnacle_{side} IS NOT NULL AND pinnacle_{side} > 1""",
                [league, team]
            )
            features[f"{prefix}_pinnacle_strength"] = round(pin["avg_implied"], 4) if pin and pin["avg_implied"] else 0.33
        except Exception:
            features[f"{prefix}_pinnacle_strength"] = 0.33

    # 14. Goals per game for each team
    for team, prefix in [(home_team, "home"), (away_team, "away")]:
        try:
            gpg = db.fetch_one(
                """SELECT AVG(CASE WHEN home_team = ? THEN ft_home_goals
                                   WHEN away_team = ? THEN ft_away_goals END) as avg_goals
                   FROM matches WHERE league = ? AND (home_team = ? OR away_team = ?)""",
                [team, team, league, team, team]
            )
            features[f"{prefix}_goals_pg"] = round(gpg["avg_goals"], 3) if gpg and gpg["avg_goals"] else 1.3
        except Exception:
            features[f"{prefix}_goals_pg"] = 1.3

    # 15. Combined expected total (sum of both teams' goals per game)
    features["combined_goals_pg"] = features.get("home_goals_pg", 1.3) + features.get("away_goals_pg", 1.3)

    # 16. Elo difference (proxy for match competitiveness)
    try:
        ratings = elo_model.build_ratings(league)
        home_elo = ratings.get(home_team, {}).get("elo", 1500)
        away_elo = ratings.get(away_team, {}).get("elo", 1500)
        features["elo_diff"] = abs(home_elo - away_elo)
    except Exception:
        features["elo_diff"] = 0

    return features


def build_training_data(league="PL"):
    """Build feature matrix for over/under 2.5 training."""
    matches = db.fetch_all(
        """SELECT * FROM matches WHERE league = ? AND ft_result IS NOT NULL
           AND ft_home_goals IS NOT NULL
           ORDER BY match_date ASC""",
        [league]
    )
    if len(matches) < 50:
        logger.warning("Not enough matches to train O/U model: %d", len(matches))
        return None, None

    # Use last 60% for training (first 40% for warm-up)
    split_idx = int(len(matches) * 0.4)
    train_matches = matches[split_idx:]

    X_rows = []
    y = []

    for m in train_matches:
        try:
            features = extract_ou_features(
                m["home_team"], m["away_team"], league, m["match_date"]
            )
            X_rows.append(features)

            # Target: 1 = Over 2.5, 0 = Under 2.5
            total_goals = m["ft_home_goals"] + m["ft_away_goals"]
            y.append(1 if total_goals > 2 else 0)
        except Exception as e:
            logger.debug("Skipping match for O/U training: %s", e)
            continue

    if not X_rows:
        return None, None

    X = pd.DataFrame(X_rows)
    return X, np.array(y)


def train(league="PL"):
    """Train the Over/Under 2.5 XGBoost binary classifier."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost not installed")
        return None

    X, y = build_training_data(league)
    if X is None:
        return None

    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
    )

    model.fit(X, y)

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "features": list(X.columns)}, f)

    logger.info("Over/Under 2.5 model trained on %d matches, saved to %s", len(X), MODEL_PATH)
    return model


def predict_over_under(home_team, away_team, league="PL", match_date=None):
    """
    Predict Over/Under 2.5 goals using dedicated model.

    Returns dict with over25_prob, under25_prob, predicted_total, confidence.
    """
    # Load or train model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                saved = pickle.load(f)
            model = saved["model"]
            feature_names = saved["features"]
        except Exception as e:
            logger.warning("Failed to load O/U model: %s", e)
            model = train(league)
            if model is None:
                return None
            feature_names = None
    else:
        model = train(league)
        if model is None:
            return None
        feature_names = None

    features = extract_ou_features(home_team, away_team, league, match_date)
    X = pd.DataFrame([features])

    if feature_names:
        # Ensure column order matches training
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names]

    try:
        probs = model.predict_proba(X)[0]
        over25_prob = float(probs[1])
        under25_prob = float(probs[0])
    except Exception as e:
        logger.error("O/U prediction failed: %s", e)
        return None

    # Estimate predicted total from feature data
    home_scored = features.get("home_rolling10_scored", 1.3)
    away_scored = features.get("away_rolling10_scored", 1.3)
    predicted_total = round(home_scored + away_scored, 2)

    # Confidence: how far from 0.5 the prediction is
    confidence = round(abs(over25_prob - 0.5) * 2, 4)

    return {
        "over25_prob": round(over25_prob, 4),
        "under25_prob": round(under25_prob, 4),
        "predicted_total": predicted_total,
        "confidence": confidence,
    }
