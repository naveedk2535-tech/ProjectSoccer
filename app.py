"""
ProjectSoccer — Flask Application
Main entry point for the soccer prediction dashboard.
"""
import logging
import json
from datetime import datetime

from flask import Flask, render_template, jsonify, request

import config
from database import db
from data import football_data_uk, football_data_api, odds_api
from data.rate_limiter import get_usage_summary
from models import ensemble, poisson, elo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = config.FLASK_SECRET_KEY


@app.before_request
def ensure_db():
    """Initialize database on first request."""
    if not hasattr(app, "_db_initialized"):
        db.init_db()
        app._db_initialized = True


# --- Dashboard Routes ---

@app.route("/")
def dashboard():
    """Main command center dashboard."""
    league = request.args.get("league", "PL")
    league_config = config.LEAGUES.get(league, config.LEAGUES["PL"])

    # Get upcoming fixtures
    fixtures = db.fetch_all(
        """SELECT * FROM fixtures WHERE league = ? AND status IN ('SCHEDULED', 'TIMED')
           ORDER BY match_date ASC LIMIT 20""",
        [league]
    )

    # Get predictions for upcoming fixtures
    predictions = []
    for f in fixtures:
        pred = db.fetch_one(
            "SELECT * FROM predictions WHERE league = ? AND home_team = ? AND away_team = ?",
            [league, f["home_team"], f["away_team"]]
        )
        predictions.append({
            "fixture": f,
            "prediction": pred,
        })

    # Get value bets
    value_bets = db.fetch_all(
        """SELECT * FROM value_bets WHERE league = ? AND result = 'pending'
           ORDER BY edge_percent DESC LIMIT 10""",
        [league]
    )

    # Get standings from cache (don't hit API on page load)
    try:
        standings = football_data_api.get_standings(league)
    except Exception:
        standings = []

    # API usage
    try:
        api_usage = get_usage_summary()
    except Exception:
        api_usage = {}

    # Match count
    match_count = football_data_uk.get_match_count(league)

    return render_template("dashboard.html",
        league=league,
        league_config=league_config,
        leagues=config.LEAGUES,
        fixtures=fixtures,
        predictions=predictions,
        value_bets=value_bets,
        standings=standings,
        api_usage=api_usage,
        match_count=match_count,
        now=datetime.utcnow(),
    )


@app.route("/match/<league>/<home_team>/<away_team>")
def match_detail(league, home_team, away_team):
    """Detailed match prediction view."""
    home_team = home_team.replace("_", " ")
    away_team = away_team.replace("_", " ")
    league_config = config.LEAGUES.get(league, config.LEAGUES["PL"])

    # Try cached prediction first, fall back to live computation
    cached_pred = db.fetch_one(
        "SELECT * FROM predictions WHERE league = ? AND home_team = ? AND away_team = ?",
        [league, home_team, away_team]
    )

    if cached_pred and cached_pred.get("ensemble_home"):
        # Build prediction dict from cached data
        import json as _json
        scoreline = None
        try:
            scoreline = _json.loads(cached_pred["scoreline_matrix"]) if cached_pred.get("scoreline_matrix") else None
        except Exception:
            pass

        prediction = {
            "home_win": cached_pred["ensemble_home"],
            "draw": cached_pred["ensemble_draw"],
            "away_win": cached_pred["ensemble_away"],
            "over25": cached_pred.get("ensemble_over25"),
            "btts": cached_pred.get("ensemble_btts"),
            "confidence": cached_pred.get("confidence", 0.5),
            "most_likely_score": None,
            "scoreline_matrix": scoreline,
            "predicted_outcome": "H" if (cached_pred["ensemble_home"] or 0) >= max(cached_pred["ensemble_draw"] or 0, cached_pred["ensemble_away"] or 0) else ("A" if (cached_pred["ensemble_away"] or 0) > (cached_pred["ensemble_draw"] or 0) else "D"),
            "implied_odds": {
                "home": round(1 / cached_pred["ensemble_home"], 2) if cached_pred["ensemble_home"] else None,
                "draw": round(1 / cached_pred["ensemble_draw"], 2) if cached_pred["ensemble_draw"] else None,
                "away": round(1 / cached_pred["ensemble_away"], 2) if cached_pred["ensemble_away"] else None,
            },
            "models_used": [],
            "model_details": {},
        }
        # Reconstruct model details from cached columns
        for model_name, prefix in [("poisson", "poisson"), ("elo", "elo"), ("xgboost", "xgboost"), ("sentiment", "sentiment")]:
            h = cached_pred.get(f"{prefix}_home")
            d = cached_pred.get(f"{prefix}_draw")
            a = cached_pred.get(f"{prefix}_away")
            if h is not None:
                prediction["models_used"].append(model_name)
                weight = config.MODEL_WEIGHTS.get(model_name, 0.25)
                prediction["model_details"][model_name] = {
                    "weight": weight, "home_win": h, "draw": d, "away_win": a,
                }
        prediction["models_available"] = len(prediction["models_used"])

        # Find most likely score from matrix
        if scoreline:
            import numpy as np
            matrix = np.array(scoreline)
            idx = np.unravel_index(matrix.argmax(), matrix.shape)
            prediction["most_likely_score"] = f"{idx[0]}-{idx[1]}"
    else:
        prediction = None

    # Get odds from database only (no API call)
    best_odds = odds_api.get_best_odds(league, home_team, away_team)

    # Get cached value bets
    value = db.fetch_all(
        """SELECT * FROM value_bets WHERE league = ? AND home_team = ? AND away_team = ? AND result = 'pending'
           ORDER BY edge_percent DESC""",
        [league, home_team, away_team]
    )

    # Head-to-head history
    h2h_matches = db.fetch_all(
        """SELECT * FROM matches
           WHERE league = ? AND
                 ((home_team = ? AND away_team = ?) OR (home_team = ? AND away_team = ?))
           ORDER BY match_date DESC LIMIT 10""",
        [league, home_team, away_team, away_team, home_team]
    )

    # Recent form (last 5 for each team)
    home_form = db.fetch_all(
        """SELECT * FROM matches
           WHERE league = ? AND (home_team = ? OR away_team = ?) AND ft_result IS NOT NULL
           ORDER BY match_date DESC LIMIT 5""",
        [league, home_team, home_team]
    )
    away_form = db.fetch_all(
        """SELECT * FROM matches
           WHERE league = ? AND (home_team = ? OR away_team = ?) AND ft_result IS NOT NULL
           ORDER BY match_date DESC LIMIT 5""",
        [league, away_team, away_team]
    )

    # Sentiment
    home_sentiment = db.fetch_one(
        "SELECT * FROM sentiment WHERE league = ? AND team = ? ORDER BY score_date DESC LIMIT 1",
        [league, home_team]
    )
    away_sentiment = db.fetch_one(
        "SELECT * FROM sentiment WHERE league = ? AND team = ? ORDER BY score_date DESC LIMIT 1",
        [league, away_team]
    )

    return render_template("match.html",
        league=league,
        league_config=league_config,
        leagues=config.LEAGUES,
        home_team=home_team,
        away_team=away_team,
        prediction=prediction,
        best_odds=best_odds,
        value_bets=value,
        h2h_matches=h2h_matches,
        home_form=home_form,
        away_form=away_form,
        home_sentiment=home_sentiment,
        away_sentiment=away_sentiment,
    )


@app.route("/league/<league>")
def league_view(league):
    """League overview with standings and team ratings."""
    league_config = config.LEAGUES.get(league, config.LEAGUES["PL"])
    standings = football_data_api.get_standings(league)

    # Team ratings
    team_ratings = db.fetch_all(
        """SELECT * FROM team_ratings WHERE league = ?
           ORDER BY elo_rating DESC""",
        [league]
    )

    return render_template("league.html",
        league=league,
        league_config=league_config,
        leagues=config.LEAGUES,
        standings=standings,
        team_ratings=team_ratings,
    )


@app.route("/sentiment")
def sentiment_view():
    """Sentiment tracker page."""
    league = request.args.get("league", "PL")
    league_config = config.LEAGUES.get(league, config.LEAGUES["PL"])

    sentiments = db.fetch_all(
        """SELECT * FROM sentiment WHERE league = ?
           ORDER BY score_date DESC, combined_score DESC""",
        [league]
    )

    # Group by team, get latest
    team_sentiments = {}
    for s in sentiments:
        if s["team"] not in team_sentiments:
            team_sentiments[s["team"]] = s

    return render_template("sentiment.html",
        league=league,
        league_config=league_config,
        leagues=config.LEAGUES,
        team_sentiments=team_sentiments,
    )


@app.route("/performance")
def performance_view():
    """Model backtest and performance page."""
    league = request.args.get("league", "PL")
    league_config = config.LEAGUES.get(league, config.LEAGUES["PL"])

    performance = db.fetch_all(
        "SELECT * FROM model_performance WHERE league = ? ORDER BY season DESC",
        [league]
    )

    return render_template("performance.html",
        league=league,
        league_config=league_config,
        leagues=config.LEAGUES,
        performance=performance,
    )


# --- API Routes ---

@app.route("/api/predict/<league>/<home_team>/<away_team>")
def api_predict(league, home_team, away_team):
    """API endpoint for match prediction. Uses cached predictions when available."""
    home_team = home_team.replace("_", " ")
    away_team = away_team.replace("_", " ")

    # Try cache first
    cached = db.fetch_one(
        "SELECT * FROM predictions WHERE league = ? AND home_team = ? AND away_team = ?",
        [league, home_team, away_team]
    )
    if cached and cached.get("ensemble_home"):
        return jsonify({
            "home_win": cached["ensemble_home"],
            "draw": cached["ensemble_draw"],
            "away_win": cached["ensemble_away"],
            "over25": cached.get("ensemble_over25"),
            "btts": cached.get("ensemble_btts"),
            "confidence": cached.get("confidence"),
            "source": "cached",
        })

    # Fall back to live prediction
    force = request.args.get("force", "false") == "true"
    if force:
        prediction = ensemble.predict(home_team, away_team, league)
        if not prediction:
            return jsonify({"error": "Could not generate prediction"}), 404
        return jsonify(prediction)

    return jsonify({"error": "No cached prediction. Add ?force=true to compute live (slow)."}), 404


@app.route("/api/fixtures/<league>")
def api_fixtures(league):
    """API endpoint for upcoming fixtures."""
    fixtures = football_data_api.get_upcoming_fixtures(league)
    return jsonify(fixtures)


@app.route("/api/odds/<league>")
def api_odds(league):
    """API endpoint for current odds."""
    odds = odds_api.get_odds(league)
    return jsonify(odds)


@app.route("/api/standings/<league>")
def api_standings(league):
    """API endpoint for league standings."""
    standings = football_data_api.get_standings(league)
    return jsonify(standings)


@app.route("/api/usage")
def api_usage():
    """API endpoint for API usage stats."""
    return jsonify(get_usage_summary())


@app.route("/api/refresh-data", methods=["POST"])
def api_refresh_data():
    """Manually trigger data refresh."""
    league = request.json.get("league", "PL") if request.json else "PL"
    results = {}

    action = request.json.get("action", "all") if request.json else "all"

    if action in ("all", "csv"):
        count = football_data_uk.download_all_leagues()
        results["csv_matches_imported"] = count

    if action in ("all", "fixtures"):
        fixtures = football_data_api.get_upcoming_fixtures(league)
        results["fixtures_fetched"] = len(fixtures)

    if action in ("all", "odds"):
        odds = odds_api.get_odds(league)
        results["odds_events"] = len(odds)

    return jsonify({"status": "ok", "results": results})


# --- Template Filters ---

@app.template_filter("pct")
def pct_filter(value):
    """Format as percentage."""
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


@app.template_filter("odds")
def odds_filter(value):
    """Format probability as decimal odds."""
    if not value or value <= 0:
        return "—"
    return f"{1/value:.2f}"


@app.template_filter("edge_color")
def edge_color_filter(edge):
    """Return CSS color class based on edge percentage."""
    if edge is None:
        return "text-gray-500"
    if edge >= 10:
        return "text-green-400"
    if edge >= 5:
        return "text-green-500"
    if edge >= 0:
        return "text-yellow-400"
    return "text-red-400"


@app.template_filter("form_badge")
def form_badge_filter(result):
    """Return badge class for W/D/L."""
    if result == "H" or result == "W":
        return "bg-green-600"
    elif result == "D":
        return "bg-yellow-600"
    elif result == "A" or result == "L":
        return "bg-red-600"
    return "bg-gray-600"


@app.template_filter("tojson_safe")
def tojson_safe_filter(value):
    """Safely convert to JSON for JavaScript."""
    return json.dumps(value, default=str)


if __name__ == "__main__":
    db.init_db()
    app.run(debug=True, port=5050)
