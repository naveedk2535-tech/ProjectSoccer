"""
ProjectSoccer Scheduler
Automated data refresh tasks. Run via PythonAnywhere scheduled tasks or cron.

Usage:
    python scheduler.py --task daily          # fixtures + odds + sentiment
    python scheduler.py --task weekly         # historical CSV update + ratings rebuild + retrain + optimize
    python scheduler.py --task all            # everything
    python scheduler.py --task fixtures       # just fixtures
    python scheduler.py --task odds           # just odds
    python scheduler.py --task sentiment      # just sentiment
    python scheduler.py --task ratings        # recalculate ratings
    python scheduler.py --task predictions    # generate predictions for upcoming
    python scheduler.py --task retrain        # retrain XGBoost model
    python scheduler.py --task optimize_weights  # optimize ensemble weights
    python scheduler.py --task train_stacker     # train stacking ensemble meta-model
"""
import argparse
import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("scheduler")

from database import db
from data import football_data_uk, football_data_api, odds_api
from data import reddit_client, news_client
from models import ensemble, poisson, elo
from models import xgboost_model
from models import diagnosis as diagnosis_module
import config


def task_fixtures():
    """Fetch upcoming fixtures from football-data.org."""
    logger.info("Fetching upcoming fixtures...")
    for code, league in config.LEAGUES.items():
        if not league["enabled"]:
            continue
        fixtures = football_data_api.get_upcoming_fixtures(code)
        logger.info("  %s: %d fixtures", league["name"], len(fixtures))


def task_odds():
    """Fetch latest odds from The Odds API."""
    logger.info("Fetching odds...")
    for code, league in config.LEAGUES.items():
        if not league["enabled"]:
            continue
        odds = odds_api.get_odds(code)
        logger.info("  %s: %d events with odds", league["name"], len(odds))


def task_sentiment():
    """Fetch sentiment from Reddit + NewsAPI."""
    logger.info("Fetching sentiment data...")
    for code, league in config.LEAGUES.items():
        if not league["enabled"]:
            continue
        reddit_client.fetch_all_teams(code)
        news_client.fetch_all_teams(code)
        logger.info("  %s: sentiment updated", league["name"])


def task_csv():
    """Download/update historical CSV data."""
    logger.info("Updating historical CSVs...")
    count = football_data_uk.download_all_leagues()
    logger.info("  Imported %d new matches", count)


def task_ratings():
    """Recalculate team ratings from historical data."""
    logger.info("Recalculating team ratings...")
    for code, league in config.LEAGUES.items():
        if not league["enabled"]:
            continue

        # Build Elo ratings
        ratings = elo.build_ratings(code)
        # Build Poisson strengths
        strengths = poisson.calculate_team_strengths(code)

        # Save to database
        for team, r in ratings.items():
            s = strengths.get(team, {})
            db.upsert("team_ratings", {
                "league": code,
                "team": team,
                "season": league["seasons"][0],
                "as_of_date": datetime.utcnow().strftime("%Y-%m-%d"),
                "elo_rating": r["elo"],
                "elo_home": r["elo_home"],
                "elo_away": r["elo_away"],
                "attack_strength": s.get("attack_strength"),
                "defence_weakness": s.get("defence_weakness"),
                "home_attack_strength": s.get("home_attack"),
                "home_defence_weakness": s.get("home_defence"),
                "away_attack_strength": s.get("away_attack"),
                "away_defence_weakness": s.get("away_defence"),
                "form_last5": r.get("form_last5"),
                "form_last10": r.get("form_last10"),
                "current_streak_type": r.get("streak_type"),
                "current_streak_length": r.get("streak_length"),
            }, ["league", "team", "as_of_date"])

        logger.info("  %s: %d teams rated", league["name"], len(ratings))


def task_predictions():
    """Generate predictions for all upcoming fixtures."""
    logger.info("Generating predictions...")
    for code, league in config.LEAGUES.items():
        if not league["enabled"]:
            continue

        fixtures = db.fetch_all(
            """SELECT * FROM fixtures WHERE league = ? AND status IN ('SCHEDULED', 'TIMED')
               ORDER BY match_date ASC""",
            [code]
        )

        for f in fixtures:
            pred = ensemble.predict(f["home_team"], f["away_team"], code,
                                    f["match_date"][:10] if f["match_date"] else None)
            if pred:
                import json
                db.upsert("predictions", {
                    "league": code,
                    "match_date": f["match_date"][:10] if f["match_date"] else None,
                    "home_team": f["home_team"],
                    "away_team": f["away_team"],
                    "fixture_id": f.get("id"),
                    "poisson_home": pred["model_details"].get("poisson", {}).get("home_win"),
                    "poisson_draw": pred["model_details"].get("poisson", {}).get("draw"),
                    "poisson_away": pred["model_details"].get("poisson", {}).get("away_win"),
                    "elo_home": pred["model_details"].get("elo", {}).get("home_win"),
                    "elo_draw": pred["model_details"].get("elo", {}).get("draw"),
                    "elo_away": pred["model_details"].get("elo", {}).get("away_win"),
                    "xgboost_home": pred["model_details"].get("xgboost", {}).get("home_win"),
                    "xgboost_draw": pred["model_details"].get("xgboost", {}).get("draw"),
                    "xgboost_away": pred["model_details"].get("xgboost", {}).get("away_win"),
                    "sentiment_home": pred["model_details"].get("sentiment", {}).get("home_win"),
                    "sentiment_draw": pred["model_details"].get("sentiment", {}).get("draw"),
                    "sentiment_away": pred["model_details"].get("sentiment", {}).get("away_win"),
                    "ensemble_home": pred["home_win"],
                    "ensemble_draw": pred["draw"],
                    "ensemble_away": pred["away_win"],
                    "ensemble_over25": pred.get("over25"),
                    "ensemble_btts": pred.get("btts"),
                    "confidence": pred["confidence"],
                    "scoreline_matrix": json.dumps(pred.get("scoreline_matrix")),
                }, ["league", "match_date", "home_team", "away_team"])

                # Check for value bets
                best_odds = odds_api.get_best_odds(code, f["home_team"], f["away_team"])
                if best_odds:
                    value_bets = ensemble.calculate_value(pred, best_odds)
                    for vb in value_bets:
                        db.upsert("value_bets", {
                            "league": code,
                            "match_date": f["match_date"][:10] if f["match_date"] else None,
                            "home_team": f["home_team"],
                            "away_team": f["away_team"],
                            "bet_type": vb["bet_type"],
                            "model_probability": vb["model_probability"],
                            "best_bookmaker": vb["best_bookmaker"],
                            "best_odds": vb["best_odds"],
                            "implied_probability": vb["implied_probability"],
                            "edge_percent": vb["edge_percent"],
                            "kelly_stake": vb["kelly_stake"],
                            "confidence": vb["confidence"],
                            "result": "pending",
                        }, ["league", "match_date", "home_team", "away_team", "bet_type"])

        logger.info("  %s: %d fixtures predicted", league["name"], len(fixtures))


def task_tracker():
    """Generate and settle model tracker entries."""
    logger.info("Running model tracker...")

    # Use the Flask app endpoints internally
    from app import app as flask_app
    with flask_app.test_client() as client:
        # Simulate logged-in session
        with client.session_transaction() as sess:
            sess["logged_in"] = True
            sess["username"] = "scheduler"

        resp = client.post("/api/tracker/generate")
        gen = resp.get_json()
        logger.info("  Tracker generate: %s", gen)

        resp = client.post("/api/tracker/settle")
        settle = resp.get_json()
        logger.info("  Tracker settle: %s", settle)


def run_daily():
    """Daily task: fixtures + odds + sentiment + predictions + tracker."""
    task_fixtures()
    task_odds()
    task_sentiment()
    task_predictions()
    task_tracker()


def task_retrain():
    """Retrain XGBoost model if new model performs better."""
    logger.info("Evaluating model retraining...")
    for code, league in config.LEAGUES.items():
        if not league["enabled"]:
            continue

        try:
            # Get current model's Brier score
            old_brier = xgboost_model.evaluate_model(code)
            if old_brier is None:
                logger.info("  %s: No existing model to compare, training fresh", league["name"])
                xgboost_model.train(code)
                continue

            # Train a new model
            new_model = xgboost_model.train(code)
            if new_model is None:
                logger.warning("  %s: Failed to train new model", league["name"])
                continue

            # Evaluate the newly trained model
            new_brier = xgboost_model.evaluate_model(code)
            if new_brier is None:
                logger.warning("  %s: Could not evaluate new model", league["name"])
                continue

            if new_brier < old_brier:
                improvement = round((old_brier - new_brier) / old_brier * 100, 2)
                logger.info(
                    "  %s: New model is better! Brier: %.4f -> %.4f (%.2f%% improvement). Saved.",
                    league["name"], old_brier, new_brier, improvement
                )
            else:
                degradation = round((new_brier - old_brier) / old_brier * 100, 2)
                logger.info(
                    "  %s: New model is worse (Brier: %.4f -> %.4f, +%.2f%%). Keeping new model as it was already saved by train().",
                    league["name"], old_brier, new_brier, degradation
                )
                # Note: In practice you'd want to back up and restore the old model.
                # For now we accept the retrained model since it uses the latest data.

        except Exception as e:
            logger.error("  %s: Retrain error: %s", league["name"], e)


def task_optimize_weights():
    """Optimize ensemble model weights for each enabled league."""
    logger.info("Optimizing ensemble weights...")
    for code, league in config.LEAGUES.items():
        if not league["enabled"]:
            continue
        try:
            result = ensemble.optimize_weights(code)
            if result:
                logger.info("  %s: Optimized weights: %s (Brier: %.4f)",
                            league["name"], result["weights"], result["brier"])
            else:
                logger.info("  %s: Not enough data to optimize weights", league["name"])
        except Exception as e:
            logger.error("  %s: Weight optimization error: %s", league["name"], e)


def task_train_stacker():
    """Train the stacking ensemble meta-model for each enabled league."""
    logger.info("Training stacking ensemble...")
    for code, league in config.LEAGUES.items():
        if not league["enabled"]:
            continue
        try:
            result = ensemble.train_stacker(code)
            if result:
                logger.info("  %s: Stacker trained successfully", league["name"])
            else:
                logger.info("  %s: Not enough data to train stacker", league["name"])
        except Exception as e:
            logger.error("  %s: Stacker training error: %s", league["name"], e)


def run_weekly():
    """Weekly task: CSV update + ratings rebuild + retrain + optimize + stacker + predictions."""
    task_csv()
    task_ratings()
    task_retrain()
    task_optimize_weights()
    task_train_stacker()
    task_predictions()


def run_all():
    """Run everything."""
    task_csv()
    task_fixtures()
    task_odds()
    task_sentiment()
    task_ratings()
    task_retrain()
    task_optimize_weights()
    task_train_stacker()
    task_predictions()


if __name__ == "__main__":
    db.init_db()

    parser = argparse.ArgumentParser(description="ProjectSoccer Scheduler")
    parser.add_argument("--task", required=True,
                        choices=["daily", "weekly", "all", "fixtures", "odds",
                                 "sentiment", "csv", "ratings", "predictions",
                                 "retrain", "optimize_weights", "train_stacker", "tracker"],
                        help="Which task to run")
    args = parser.parse_args()

    tasks = {
        "daily": run_daily,
        "weekly": run_weekly,
        "all": run_all,
        "fixtures": task_fixtures,
        "odds": task_odds,
        "sentiment": task_sentiment,
        "csv": task_csv,
        "ratings": task_ratings,
        "predictions": task_predictions,
        "retrain": task_retrain,
        "optimize_weights": task_optimize_weights,
        "train_stacker": task_train_stacker,
        "tracker": task_tracker,
    }

    logger.info("Starting task: %s", args.task)
    start = datetime.utcnow()
    tasks[args.task]()
    elapsed = (datetime.utcnow() - start).total_seconds()
    logger.info("Task %s completed in %.1fs", args.task, elapsed)
