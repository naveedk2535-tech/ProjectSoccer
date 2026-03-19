"""
XGBoost prediction model.

Trains on 40+ features extracted from match data:
Elo ratings, form (including exponential decay), attack/defence strengths,
days rest, H2H records, referee tendencies, seasonality, sentiment, streaks,
corners, fouls/discipline, Pinnacle implied probabilities, over/under rates,
rolling 10-match metrics, and regression-to-mean conversion luck indicators.
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

    # 14-16. Referee stats (replaces hardcoded 2.7)
    referee_name = None
    # Try to look up referee from fixtures table
    if match_date:
        try:
            fixture = db.fetch_one(
                """SELECT referee FROM fixtures
                   WHERE league = ? AND home_team = ? AND away_team = ?
                   AND DATE(match_date) = DATE(?)""",
                [league, home_team, away_team, match_date]
            )
            if fixture and fixture.get("referee"):
                referee_name = fixture["referee"]
        except Exception:
            pass

        # Also check matches table for historical data
        if not referee_name:
            try:
                match_row = db.fetch_one(
                    """SELECT referee FROM matches
                       WHERE league = ? AND home_team = ? AND away_team = ?
                       AND match_date = ?""",
                    [league, home_team, away_team, match_date]
                )
                if match_row and match_row.get("referee"):
                    referee_name = match_row["referee"]
            except Exception:
                pass

    # Look up referee stats if referee is known
    if referee_name:
        try:
            ref_stats = db.fetch_one(
                """SELECT avg_total_goals, home_win_pct, over25_pct
                   FROM referee_stats WHERE referee = ? AND league = ?""",
                [referee_name, league]
            )
            if ref_stats:
                features["referee_avg_goals"] = ref_stats["avg_total_goals"] if ref_stats["avg_total_goals"] is not None else 2.7
                features["referee_home_win_pct"] = ref_stats["home_win_pct"] if ref_stats["home_win_pct"] is not None else 0.46
                features["referee_over25_pct"] = ref_stats["over25_pct"] if ref_stats["over25_pct"] is not None else 0.50
            else:
                features["referee_avg_goals"] = 2.7
                features["referee_home_win_pct"] = 0.46
                features["referee_over25_pct"] = 0.50
        except Exception:
            features["referee_avg_goals"] = 2.7
            features["referee_home_win_pct"] = 0.46
            features["referee_over25_pct"] = 0.50
    else:
        features["referee_avg_goals"] = 2.7
        features["referee_home_win_pct"] = 0.46
        features["referee_over25_pct"] = 0.50

    # 15. Month of season (1-12)
    if match_date:
        features["month"] = datetime.strptime(match_date, "%Y-%m-%d").month
    else:
        features["month"] = datetime.utcnow().month

    # 16-17. Sentiment scores (default neutral)
    features["home_sentiment"] = 0.0
    features["away_sentiment"] = 0.0
    # Will be populated by sentiment module

    # 18. Is promoted team (either side) — uses Elo model's promoted detection
    home_promoted = 1 if home_r.get("is_promoted", False) else 0
    away_promoted = 1 if away_r.get("is_promoted", False) else 0
    features["has_promoted"] = 1 if (home_promoted or away_promoted) else 0

    # 19. Streak length (home team)
    features["home_streak_len"] = home_r.get("streak_length", 0)

    # 20. Pinnacle implied probability (if available)
    pinnacle_implied = 0.0
    if match_date:
        pinnacle_row = db.fetch_one(
            """SELECT pinnacle_home FROM matches
               WHERE league = ? AND home_team = ? AND away_team = ? AND match_date = ?""",
            [league, home_team, away_team, match_date]
        )
        if pinnacle_row and pinnacle_row.get("pinnacle_home"):
            try:
                pinnacle_implied = round(1.0 / pinnacle_row["pinnacle_home"], 4)
            except (ZeroDivisionError, TypeError):
                pinnacle_implied = 0.0
    features["pinnacle_home_implied"] = pinnacle_implied

    # 21-24. Fixture congestion index
    if match_date:
        for team, prefix in [(home_team, "home"), (away_team, "away")]:
            for days, suffix in [(14, "14d"), (30, "30d")]:
                date_cutoff = (datetime.strptime(match_date, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
                count_row = db.fetch_one(
                    """SELECT COUNT(*) as cnt FROM matches
                       WHERE league = ? AND (home_team = ? OR away_team = ?)
                             AND match_date >= ? AND match_date < ?""",
                    [league, team, team, date_cutoff, match_date]
                )
                features[f"{prefix}_matches_{suffix}"] = count_row["cnt"] if count_row else 0
    else:
        features["home_matches_14d"] = 2
        features["away_matches_14d"] = 2
        features["home_matches_30d"] = 4
        features["away_matches_30d"] = 4

    # 25. Congestion differential (positive = home more fatigued)
    features["congestion_diff"] = features.get("home_matches_14d", 2) - features.get("away_matches_14d", 2)

    # 27-30. Shot accuracy and average shots per game
    for team, prefix in [(home_team, "home"), (away_team, "away")]:
        try:
            shots_data = db.fetch_one(
                """SELECT AVG(CASE WHEN home_team = ? THEN CAST(home_shots_target AS FLOAT) / NULLIF(home_shots, 0)
                               WHEN away_team = ? THEN CAST(away_shots_target AS FLOAT) / NULLIF(away_shots, 0) END) as ratio,
                      AVG(CASE WHEN home_team = ? THEN home_shots
                               WHEN away_team = ? THEN away_shots END) as avg_shots
                   FROM matches WHERE league = ? AND (home_team = ? OR away_team = ?)
                   AND home_shots IS NOT NULL AND home_shots > 0""",
                [team, team, team, team, league, team, team]
            )
            features[f"{prefix}_shot_accuracy"] = round(shots_data["ratio"], 3) if shots_data and shots_data["ratio"] else 0.33
            features[f"{prefix}_avg_shots"] = round(shots_data["avg_shots"], 1) if shots_data and shots_data["avg_shots"] else 12.0
        except Exception:
            features[f"{prefix}_shot_accuracy"] = 0.33
            features[f"{prefix}_avg_shots"] = 12.0

    # 31-32. Opponent-adjusted form
    features["home_opp_adj_form"] = home_r.get("opponent_adjusted_form", 0.5)
    features["away_opp_adj_form"] = away_r.get("opponent_adjusted_form", 0.5)

    # 26. Odds movement (change in best available home odds between earliest and latest fetch)
    features["odds_movement"] = 0.0
    if match_date:
        try:
            odds_rows = db.fetch_all(
                """SELECT home_odds, fetched_at FROM odds
                   WHERE league = ? AND home_team = ? AND away_team = ?
                   ORDER BY fetched_at ASC""",
                [league, home_team, away_team]
            )
            if odds_rows and len(odds_rows) >= 2:
                earliest_odds = odds_rows[0].get("home_odds")
                latest_odds = odds_rows[-1].get("home_odds")
                if earliest_odds and latest_odds and earliest_odds > 0:
                    # Positive movement = odds shortened (more money on that outcome)
                    features["odds_movement"] = round(earliest_odds - latest_odds, 4)
        except Exception:
            features["odds_movement"] = 0.0

    # --- Enhancement 1: Corners as XGBoost Features ---
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

    # --- Enhancement 2: Fouls & Discipline Features ---
    for team, prefix in [(home_team, "home"), (away_team, "away")]:
        try:
            discipline = db.fetch_one(
                """SELECT AVG(CASE WHEN home_team = ? THEN home_fouls WHEN away_team = ? THEN away_fouls END) as avg_fouls,
                          AVG(CASE WHEN home_team = ? THEN home_yellows WHEN away_team = ? THEN away_yellows END) as avg_yellows
                   FROM matches WHERE league = ? AND (home_team = ? OR away_team = ?)
                   AND home_fouls IS NOT NULL""",
                [team, team, team, team, league, team, team]
            )
            features[f"{prefix}_avg_fouls"] = round(discipline["avg_fouls"], 1) if discipline and discipline["avg_fouls"] else 12.0
            features[f"{prefix}_avg_yellows"] = round(discipline["avg_yellows"], 1) if discipline and discipline["avg_yellows"] else 1.5
        except Exception:
            features[f"{prefix}_avg_fouls"] = 12.0
            features[f"{prefix}_avg_yellows"] = 1.5

    # --- Enhancement 3: Historical Pinnacle Odds as Features ---
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
    features["market_spread"] = features["home_pinnacle_strength"] - features["away_pinnacle_strength"]

    # --- Enhancement 4: Over/Under 2.5 Implied Rate ---
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

    # --- Enhancement 5: Exponential Decay Form from Elo ---
    features["home_form5_decay"] = home_r.get("form_last5_decay", 1.5)
    features["away_form5_decay"] = away_r.get("form_last5_decay", 1.5)

    # --- Enhancement 6: Rolling 10-Match Metrics ---
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

    # --- Enhancement 7: Regression to Mean Indicator ---
    for team, prefix in [(home_team, "home"), (away_team, "away")]:
        try:
            reg = db.fetch_one(
                """SELECT AVG(CASE WHEN home_team = ? THEN ft_home_goals WHEN away_team = ? THEN ft_away_goals END) as avg_goals,
                          AVG(CASE WHEN home_team = ? THEN home_shots_target WHEN away_team = ? THEN away_shots_target END) as avg_sot
                   FROM matches WHERE league = ? AND (home_team = ? OR away_team = ?)
                   AND home_shots_target IS NOT NULL""",
                [team, team, team, team, league, team, team]
            )
            if reg and reg["avg_goals"] and reg["avg_sot"] and reg["avg_sot"] > 0:
                # conversion_rate: goals per shot on target. League avg is ~0.30
                conversion = reg["avg_goals"] / reg["avg_sot"]
                # deviation from league average conversion (0.30)
                features[f"{prefix}_conversion_luck"] = round(conversion - 0.30, 3)
            else:
                features[f"{prefix}_conversion_luck"] = 0.0
        except Exception:
            features[f"{prefix}_conversion_luck"] = 0.0

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


def evaluate_model(league="PL"):
    """Run cross-validation on the training data and return the Brier score."""
    try:
        import xgboost as xgb
        from sklearn.model_selection import cross_val_predict
    except ImportError:
        logger.error("xgboost or sklearn not installed for evaluation")
        return None

    X, y = build_training_data(league)
    if X is None or len(X) < 50:
        logger.warning("Not enough data to evaluate model")
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

    try:
        # 5-fold cross-validation to get out-of-sample probabilities
        probs = cross_val_predict(model, X, y, cv=5, method="predict_proba")

        # Calculate Brier score: mean of sum of squared differences
        # For multi-class: Brier = (1/N) * sum_i sum_k (p_ik - o_ik)^2
        n_samples = len(y)
        n_classes = 3
        brier = 0.0
        for i in range(n_samples):
            for k in range(n_classes):
                actual = 1.0 if y[i] == k else 0.0
                brier += (probs[i][k] - actual) ** 2
        brier /= n_samples

        logger.info("XGBoost CV Brier score for %s: %.4f", league, brier)
        return round(brier, 4)
    except Exception as e:
        logger.error("Error evaluating model: %s", e)
        return None


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
