"""
ProjectSoccer — Flask Application
Main entry point for the soccer prediction dashboard.
"""
import logging
import json
import os
from datetime import datetime

from functools import wraps
from flask import Flask, render_template, jsonify, request, session, redirect, url_for

import config
from database import db
from data import football_data_uk, football_data_api, odds_api
from data.rate_limiter import get_usage_summary
from models import ensemble, poisson, elo
from models import diagnosis as diagnosis_module

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = config.FLASK_SECRET_KEY


USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "users.json")
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "app_settings.json")


def load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"bankroll": 0}


def save_settings(settings):
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


def load_users():
    """Load users from JSON file."""
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"admin": "!admin123!"}


def save_users(users):
    """Save users to JSON file."""
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


USERS = load_users()


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


@app.before_request
def ensure_db():
    """Initialize database on first request."""
    if not hasattr(app, "_db_initialized"):
        db.init_db()
        app._db_initialized = True


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username in USERS and USERS[username] == password:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("dashboard"))
        error = "Invalid credentials"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# --- Dashboard Routes ---

@app.route("/")
@login_required
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
@login_required
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
@login_required
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
@login_required
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
@login_required
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


@app.route("/portfolio")
@login_required
def portfolio_view():
    """Bet tracking portfolio — like a stock portfolio."""
    league = request.args.get("league", "PL")
    league_config = config.LEAGUES.get(league, config.LEAGUES["PL"])

    # All bets
    all_bets = db.fetch_all(
        "SELECT * FROM user_bets ORDER BY placed_at DESC"
    )

    # Summary stats
    total_bets = len(all_bets)
    pending = [b for b in all_bets if b["status"] == "pending"]
    settled = [b for b in all_bets if b["status"] in ("won", "lost")]
    won = [b for b in all_bets if b["status"] == "won"]
    lost = [b for b in all_bets if b["status"] == "lost"]

    total_staked = sum(b["stake"] for b in settled) if settled else 0
    total_profit = sum(b["profit_loss"] for b in settled) if settled else 0
    total_payout = sum(b["payout"] for b in settled) if settled else 0
    win_rate = (len(won) / len(settled) * 100) if settled else 0
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
    avg_odds = sum(b["odds"] for b in all_bets) / len(all_bets) if all_bets else 0
    avg_stake = sum(b["stake"] for b in all_bets) / len(all_bets) if all_bets else 0
    best_win = max((b["profit_loss"] for b in won), default=0)
    worst_loss = min((b["profit_loss"] for b in lost), default=0)
    current_streak = 0
    if settled:
        for b in sorted(settled, key=lambda x: x["placed_at"], reverse=True):
            if b["status"] == "won":
                current_streak += 1
            else:
                break

    summary = {
        "total_bets": total_bets,
        "pending_count": len(pending),
        "settled_count": len(settled),
        "won_count": len(won),
        "lost_count": len(lost),
        "win_rate": round(win_rate, 1),
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "total_payout": round(total_payout, 2),
        "roi": round(roi, 1),
        "avg_odds": round(avg_odds, 2),
        "avg_stake": round(avg_stake, 2),
        "best_win": round(best_win, 2),
        "worst_loss": round(worst_loss, 2),
        "current_streak": current_streak,
        "bankroll_change": round(total_profit, 2),
    }

    settings = load_settings()
    bankroll_start = settings.get("bankroll", 0)
    # If bankroll not set, default to total amount wagered (including pending)
    if bankroll_start == 0 and all_bets:
        bankroll_start = round(sum(b["stake"] for b in all_bets), 2)

    return render_template("portfolio.html",
        league=league,
        league_config=league_config,
        leagues=config.LEAGUES,
        bets=all_bets,
        pending=pending,
        summary=summary,
        bankroll_start=bankroll_start,
    )


@app.route("/settings")
@login_required
def settings_view():
    """User management settings page."""
    users = load_users()
    return render_template("settings.html",
        leagues=config.LEAGUES,
        users=users,
        now=datetime.utcnow(),
    )


@app.route("/api/users", methods=["POST"])
@login_required
def api_add_user():
    """Add a new user."""
    global USERS
    data = request.json
    if not data or not data.get("username") or not data.get("password"):
        return jsonify({"error": "Username and password required"}), 400
    username = data["username"].strip()
    if not username:
        return jsonify({"error": "Username cannot be empty"}), 400
    users = load_users()
    if username in users:
        return jsonify({"error": "User already exists"}), 400
    users[username] = data["password"]
    save_users(users)
    USERS = users
    return jsonify({"status": "ok", "message": f"User '{username}' created"})


@app.route("/api/users/<username>/password", methods=["PUT"])
@login_required
def api_change_password(username):
    """Change a user's password."""
    global USERS
    data = request.json
    if not data or not data.get("password"):
        return jsonify({"error": "New password required"}), 400
    users = load_users()
    if username not in users:
        return jsonify({"error": "User not found"}), 404
    users[username] = data["password"]
    save_users(users)
    USERS = users
    return jsonify({"status": "ok", "message": f"Password updated for '{username}'"})


@app.route("/api/users/<username>", methods=["DELETE"])
@login_required
def api_delete_user(username):
    """Delete a user. Cannot delete the last admin."""
    global USERS
    users = load_users()
    if username not in users:
        return jsonify({"error": "User not found"}), 404
    if len(users) <= 1:
        return jsonify({"error": "Cannot delete the last user"}), 400
    del users[username]
    save_users(users)
    USERS = users
    return jsonify({"status": "ok", "message": f"User '{username}' deleted"})


@app.route("/api/bankroll", methods=["PUT"])
@login_required
def api_set_bankroll():
    """Set the starting bankroll amount."""
    data = request.json
    if not data or "bankroll" not in data:
        return jsonify({"error": "Provide bankroll amount"}), 400
    settings = load_settings()
    settings["bankroll"] = float(data["bankroll"])
    save_settings(settings)
    return jsonify({"status": "ok", "bankroll": settings["bankroll"]})


# --- API Routes ---

@app.route("/api/bets", methods=["POST"])
@login_required
def api_place_bet():
    """Place a new tracked bet."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    required = ["league", "match_date", "home_team", "away_team", "bet_type", "stake", "odds"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    db.execute(
        """INSERT INTO user_bets
           (league, match_date, home_team, away_team, bet_type, stake, odds,
            bookmaker, model_probability, edge_percent, status, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
        [data["league"], data["match_date"], data["home_team"], data["away_team"],
         data["bet_type"], float(data["stake"]), float(data["odds"]),
         data.get("bookmaker", ""), data.get("model_probability"),
         data.get("edge_percent"), data.get("notes", "")]
    )
    return jsonify({"status": "ok", "message": "Bet tracked"})


@app.route("/api/bets/<int:bet_id>/settle", methods=["POST"])
@login_required
def api_settle_bet(bet_id):
    """Settle a bet as won, lost, or void."""
    data = request.json
    if not data or "status" not in data:
        return jsonify({"error": "Provide status: won, lost, or void"}), 400

    status = data["status"]
    bet = db.fetch_one("SELECT * FROM user_bets WHERE id = ?", [bet_id])
    if not bet:
        return jsonify({"error": "Bet not found"}), 404

    if status == "won":
        payout = bet["stake"] * bet["odds"]
        profit = payout - bet["stake"]
    elif status == "lost":
        payout = 0
        profit = -bet["stake"]
    else:  # void
        payout = bet["stake"]
        profit = 0

    db.execute(
        """UPDATE user_bets SET status = ?, payout = ?, profit_loss = ?,
           settled_at = datetime('now') WHERE id = ?""",
        [status, round(payout, 2), round(profit, 2), bet_id]
    )
    return jsonify({"status": "ok", "payout": round(payout, 2), "profit": round(profit, 2)})


@app.route("/api/bets/<int:bet_id>", methods=["PUT"])
@login_required
def api_edit_bet(bet_id):
    """Edit a tracked bet."""
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400
    bet = db.fetch_one("SELECT * FROM user_bets WHERE id = ?", [bet_id])
    if not bet:
        return jsonify({"error": "Bet not found"}), 404
    updates = []
    params = []
    for field in ["stake", "odds", "bet_type", "bookmaker", "notes"]:
        if field in data:
            updates.append(f"{field} = ?")
            params.append(data[field])
    if updates:
        params.append(bet_id)
        db.execute(f"UPDATE user_bets SET {', '.join(updates)} WHERE id = ?", params)
    return jsonify({"status": "ok"})


@app.route("/api/bets/<int:bet_id>", methods=["DELETE"])
@login_required
def api_delete_bet(bet_id):
    """Delete a tracked bet."""
    db.execute("DELETE FROM user_bets WHERE id = ?", [bet_id])
    return jsonify({"status": "ok"})

@app.route("/api/predict/<league>/<home_team>/<away_team>")
@login_required
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
@login_required
def api_fixtures(league):
    """API endpoint for upcoming fixtures."""
    fixtures = football_data_api.get_upcoming_fixtures(league)
    return jsonify(fixtures)


@app.route("/api/odds/<league>")
@login_required
def api_odds(league):
    """API endpoint for current odds."""
    odds = odds_api.get_odds(league)
    return jsonify(odds)


@app.route("/api/standings/<league>")
@login_required
def api_standings(league):
    """API endpoint for league standings."""
    standings = football_data_api.get_standings(league)
    return jsonify(standings)


@app.route("/api/diagnosis")
@login_required
def api_diagnosis():
    """API endpoint for model health diagnosis."""
    try:
        health = diagnosis_module.get_model_health()
        retrain_info = diagnosis_module.should_retrain()
        rolling = diagnosis_module.calculate_rolling_performance()

        # Determine overall status
        statuses = [v.get("status", "unknown") for v in health.values()]
        if "critical" in statuses:
            overall = "critical"
        elif "degrading" in statuses:
            overall = "degrading"
        elif all(s == "healthy" for s in statuses):
            overall = "healthy"
        else:
            overall = "unknown"

        return jsonify({
            "overall_status": overall,
            "models": health,
            "retrain": retrain_info,
            "rolling_performance": rolling,
        })
    except Exception as e:
        logger.error("Diagnosis endpoint error: %s", e)
        return jsonify({
            "overall_status": "unknown",
            "models": {},
            "retrain": {"retrain": False, "reason": str(e)},
            "rolling_performance": {},
        })


@app.route("/api/usage")
@login_required
def api_usage():
    """API endpoint for API usage stats."""
    return jsonify(get_usage_summary())


@app.route("/api/refresh-data", methods=["POST"])
@login_required
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
