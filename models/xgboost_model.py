"""
XGBoost prediction model.

Trains on 20 features extracted from match data:
Elo ratings, form, attack/defence strengths, days rest,
H2H records, referee tendencies, seasonality, sentiment, streaks.
"""
import logging
import json
import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from database import db
from models import elo as elo_model
from models import poisson as poisson_model
import config

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(config.BASE_DIR, "data", "cache", "xgboost_model.pkl")


def extract_features(home_team, away_team, league="PL", match_date=None):
    """
    Extract all 20 features for a match prediction.
    """
    features = {}

    # 1-2. Elo ratings
    ratings = elo_model.build_ratings(league)
    home_r = ratings.get(home_team, {})
    away_r = ratings.get(away_team, {})
    features["home_elo"] = home_r.get("elo", 1500)
    features["away_elo"] = away_r.get("elo", 1500)

    # 3-6. Attack/Defence strengths
    strengths = poisson_model.calculate_team_strengths(league)
    home_s = strengths.get(home_team, {})
    away_s = strengths.get(away_team, {})
    features["home_attack"] = home_s.get("home_attack", 1.0)
    features["home_defence"] = home_s.get("home_defence", 1.0)
    features["away_attack"] = away_s.get("away_attack", 1.0)
    features["away_defence"] = away_s.get("away_defence", 1.0)

    # 7-8. Form last 5
    features["home_form5"] = home_r.get("form_last5", 7.5)
    features["away_form5"] = away_r.get("form_last5", 7.5)

    # 9-10. Days rest
    if match_date:
        for team, prefix in [(home_team, "home"), (away_team, "away")]:
            last = db.fetch_one(
                """SELECT MAX(match_date) as last_date FROM matches
                   WHERE league = ? AND (home_team = ? OR away_team = ?) AND match_date < ?""",
                [league, team, team, match_date]
            )
            if last and last["last_date"]:
                delta = (datetime.strptime(match_date, "%Y-%m-%d") -
                         datetime.strptime(last["last_date"], "%Y-%m-%d")).days
                features[f"{prefix}_days_rest"] = min(delta, 30)
            else:
                features[f"{prefix}_days_rest"] = 7
    else:
        features["home_days_rest"] = 7
        features["away_days_rest"] = 7

    # 11. Head-to-head win rate (home team perspective)
    h2h = db.fetch_all(
        """SELECT ft_result FROM matches
           WHERE league = ? AND home_team = ? AND away_team = ?
           ORDER BY match_date DESC LIMIT 10""",
        [league, home_team, away_team]
    )
    if h2h:
        h2h_wins = sum(1 for m in h2h if m["ft_result"] == "H")
        features["h2h_home_winrate"] = h2h_wins / len(h2h)
    else:
        features["h2h_home_winrate"] = 0.46  # league average

    # 12-13. Goals per game this season
    for team, prefix in [(home_team, "home"), (away_team, "away")]:
        gpg = db.fetch_one(
            """SELECT AVG(CASE WHEN home_team = ? THEN ft_home_goals
                               WHEN away_team = ? THEN ft_away_goals END) as avg_goals
               FROM matches WHERE league = ? AND (home_team = ? OR away_team = ?)""",
            [team, team, league, team, team]
        )
        features[f"{prefix}_goals_pg"] = round(gpg["avg_goals"], 3) if gpg and gpg["avg_goals"] else 1.3

    # 14. Referee avg total goals
    features["referee_avg_goals"] = 2.7  # default
    # Will be populated when referee is known for upcoming fixtures

    # 15. Month of season (1-12)
    if match_date:
        features["month"] = datetime.strptime(match_date, "%Y-%m-%d").month
    else:
        features["month"] = datetime.utcnow().month

    # 16-17. Sentiment scores (default neutral)
    features["home_sentiment"] = 0.0
    features["away_sentiment"] = 0.0
    # Will be populated by sentiment module

    # 18. Is promoted team (either side)
    features["has_promoted"] = 0  # TODO: detect from season-over-season data

    # 19. Streak length (home team)
    features["home_streak_len"] = home_r.get("streak_length", 0)

    # 20. Pinnacle implied probability (if available)
    features["pinnacle_home_implied"] = 0.0  # populated when odds are available

    return features


def build_training_data(league="PL"):
    """Build feature matrix from historical matches for training."""
    matches = db.fetch_all(
        """SELECT * FROM matches WHERE league = ? AND ft_result IS NOT NULL
           ORDER BY match_date ASC""",
        [league]
    )
    if len(matches) < 50:
        logger.warning("Not enough matches to train XGBoost: %d", len(matches))
        return None, None

    # We need to build features incrementally (only using data available at prediction time)
    # For simplicity, we'll use the last 60% of matches for training (first 40% for warm-up)
    split_idx = int(len(matches) * 0.4)
    train_matches = matches[split_idx:]

    X_rows = []
    y = []

    for m in train_matches:
        features = extract_features(
            m["home_team"], m["away_team"], league, m["match_date"]
        )
        X_rows.append(features)

        # Target: 0=Home, 1=Draw, 2=Away
        result_map = {"H": 0, "D": 1, "A": 2}
        y.append(result_map.get(m["ft_result"], 1))

    X = pd.DataFrame(X_rows)
    return X, np.array(y)


def train(league="PL"):
    """Train the XGBoost model on historical data."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost not installed")
        return None

    X, y = build_training_data(league)
    if X is None:
        return None

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
    )

    model.fit(X, y)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "features": list(X.columns)}, f)

    logger.info("XGBoost model trained on %d matches, saved to %s", len(X), MODEL_PATH)
    return model


def predict(home_team, away_team, league="PL", match_date=None):
    """Generate match prediction using XGBoost."""
    # Load or train model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            saved = pickle.load(f)
        model = saved["model"]
        feature_names = saved["features"]
    else:
        model = train(league)
        if model is None:
            return None
        feature_names = None

    features = extract_features(home_team, away_team, league, match_date)
    X = pd.DataFrame([features])

    if feature_names:
        # Ensure column order matches training
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names]

    probs = model.predict_proba(X)[0]

    # Feature importance for this prediction
    importances = dict(zip(X.columns, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "home_win": round(float(probs[0]), 4),
        "draw": round(float(probs[1]), 4),
        "away_win": round(float(probs[2]), 4),
        "confidence": round(float(max(probs) - sorted(probs)[-2]), 4),
        "details": {
            "features": features,
            "top_features": top_features,
        }
    }
