"""
ProjectSoccer — Flask Application
Main entry point for the soccer prediction dashboard.
"""
import logging
import json
import os
from datetime import datetime
from collections import defaultdict

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
    """Main command center dashboard — shows ALL enabled leagues."""
    enabled_leagues = {code: lg for code, lg in config.LEAGUES.items() if lg.get("enabled")}

    leagues_data = {}
    total_matches = 0
    total_fixtures = 0
    total_value_bets = 0

    for code, lg_config in enabled_leagues.items():
        # Get upcoming fixtures
        fixtures = db.fetch_all(
            """SELECT * FROM fixtures WHERE league = ? AND status IN ('SCHEDULED', 'TIMED')
               ORDER BY match_date ASC LIMIT 20""",
            [code]
        )

        # Get predictions for upcoming fixtures
        predictions = []
        for f in fixtures:
            pred = db.fetch_one(
                "SELECT * FROM predictions WHERE league = ? AND home_team = ? AND away_team = ?",
                [code, f["home_team"], f["away_team"]]
            )
            predictions.append({
                "fixture": f,
                "prediction": pred,
            })

        # Get value bets
        value_bets = db.fetch_all(
            """SELECT * FROM value_bets WHERE league = ? AND result = 'pending'
               ORDER BY edge_percent DESC LIMIT 10""",
            [code]
        )

        # Get standings from cache (don't hit API on page load)
        try:
            standings = football_data_api.get_standings(code)
        except Exception:
            standings = []

        # Match count
        match_count = football_data_uk.get_match_count(code)

        leagues_data[code] = {
            "config": lg_config,
            "fixtures": fixtures,
            "predictions": predictions,
            "value_bets": value_bets,
            "standings": standings,
            "match_count": match_count,
        }

        total_matches += match_count
        total_fixtures += len(fixtures)
        total_value_bets += len(value_bets)

    # API usage
    try:
        api_usage = get_usage_summary()
    except Exception:
        api_usage = {}

    # Data freshness status (aggregated, not per league)
    data_status = {}
    try:
        for source in ["football_data_org", "football_data_uk", "odds_api", "reddit", "newsapi"]:
            last_call = db.fetch_one(
                "SELECT called_at, response_code FROM api_calls WHERE api_name = ? AND cached = 0 ORDER BY called_at DESC LIMIT 1",
                [source]
            )
            data_status[source] = {
                "last_updated": last_call["called_at"][:16].replace("T", " ") if last_call else "Never",
                "status": "ok" if last_call and last_call.get("response_code") == 200 else "unknown",
            }
    except Exception:
        data_status = {}

    return render_template("dashboard.html",
        leagues=config.LEAGUES,
        leagues_data=leagues_data,
        total_matches=total_matches,
        total_fixtures=total_fixtures,
        total_value_bets=total_value_bets,
        api_usage=api_usage,
        data_status=data_status,
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

    # Fetch signal data for display
    signals = {}

    # Referee stats
    referee_name = None
    fixture = db.fetch_one(
        "SELECT referee FROM fixtures WHERE league = ? AND home_team = ? AND away_team = ?",
        [league, home_team, away_team]
    )
    if fixture and fixture.get("referee"):
        referee_name = fixture["referee"]
        ref_stats = db.fetch_one(
            "SELECT * FROM referee_stats WHERE referee = ? AND league = ?",
            [referee_name, league]
        )
        if ref_stats:
            signals["referee"] = {
                "name": referee_name,
                "avg_goals": ref_stats.get("avg_total_goals"),
                "over25_pct": (ref_stats.get("over25_pct", 0) or 0) * 100,
                "avg_cards": ref_stats.get("avg_yellows"),
                "matches": ref_stats.get("matches_officiated"),
                "home_win_pct": (ref_stats.get("home_win_pct", 0) or 0) * 100,
            }

    # Half-time patterns
    from models.poisson import calculate_half_profiles
    try:
        half_profiles = calculate_half_profiles(league)
        hp = half_profiles.get(home_team, {})
        if hp:
            signals["home_half_profile"] = {
                "first_half_goals": hp.get("home_avg_1h_goals", 0),
                "second_half_goals": hp.get("home_avg_2h_goals", 0),
                "comeback_rate": hp.get("comeback_rate", 0) * 100,
                "collapse_rate": hp.get("collapse_rate", 0) * 100,
            }
        ap = half_profiles.get(away_team, {})
        if ap:
            signals["away_half_profile"] = {
                "first_half_goals": ap.get("away_avg_1h_goals", 0),
                "second_half_goals": ap.get("away_avg_2h_goals", 0),
                "comeback_rate": ap.get("comeback_rate", 0) * 100,
                "collapse_rate": ap.get("collapse_rate", 0) * 100,
            }
    except Exception:
        pass

    # Fixture congestion
    from datetime import timedelta
    match_date_str = None
    for fx in db.fetch_all("SELECT match_date FROM fixtures WHERE league = ? AND home_team = ? AND away_team = ?", [league, home_team, away_team]):
        match_date_str = fx["match_date"][:10] if fx.get("match_date") else None
    if match_date_str:
        for team, key in [(home_team, "home"), (away_team, "away")]:
            for days in [14, 30]:
                try:
                    cutoff = (datetime.strptime(match_date_str, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
                    count = db.fetch_one(
                        "SELECT COUNT(*) as cnt FROM matches WHERE league = ? AND (home_team = ? OR away_team = ?) AND match_date >= ? AND match_date < ?",
                        [league, team, team, cutoff, match_date_str]
                    )
                    signals[f"{key}_matches_{days}d"] = count["cnt"] if count else 0
                except Exception:
                    signals[f"{key}_matches_{days}d"] = 0

    # Promoted team status
    from models.elo import build_ratings
    try:
        elo_ratings = build_ratings(league)
        signals["home_promoted"] = elo_ratings.get(home_team, {}).get("is_promoted", False)
        signals["away_promoted"] = elo_ratings.get(away_team, {}).get("is_promoted", False)
    except Exception:
        pass

    # Market movement
    try:
        odds_rows = db.fetch_all(
            "SELECT home_odds, fetched_at FROM odds WHERE league = ? AND home_team = ? AND away_team = ? ORDER BY fetched_at ASC",
            [league, home_team, away_team]
        )
        if odds_rows and len(odds_rows) >= 2:
            signals["odds_opening"] = odds_rows[0].get("home_odds")
            signals["odds_latest"] = odds_rows[-1].get("home_odds")
            signals["odds_movement"] = round(odds_rows[0]["home_odds"] - odds_rows[-1]["home_odds"], 3) if odds_rows[0].get("home_odds") and odds_rows[-1].get("home_odds") else 0
            signals["odds_fetch_count"] = len(odds_rows)
    except Exception:
        pass

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
        signals=signals,
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
    """Sentiment tracker page — shows ALL enabled leagues."""
    enabled_leagues = {code: lg for code, lg in config.LEAGUES.items() if lg.get("enabled")}

    leagues_sentiment = {}
    for code, lg_config in enabled_leagues.items():
        sentiments = db.fetch_all(
            """SELECT * FROM sentiment WHERE league = ?
               ORDER BY score_date DESC, combined_score DESC""",
            [code]
        )
        # Group by team, get latest
        team_sentiments = {}
        for s in sentiments:
            if s["team"] not in team_sentiments:
                team_sentiments[s["team"]] = s
        leagues_sentiment[code] = {
            "config": lg_config,
            "team_sentiments": team_sentiments,
        }

    return render_template("sentiment.html",
        leagues=config.LEAGUES,
        leagues_sentiment=leagues_sentiment,
    )


@app.route("/performance")
@login_required
def performance_view():
    """Model backtest and performance page — shows ALL enabled leagues."""
    enabled_leagues = {code: lg for code, lg in config.LEAGUES.items() if lg.get("enabled")}

    leagues_performance = {}
    for code, lg_config in enabled_leagues.items():
        performance = db.fetch_all(
            "SELECT * FROM model_performance WHERE league = ? ORDER BY season DESC",
            [code]
        )
        leagues_performance[code] = {
            "config": lg_config,
            "performance": performance,
        }

    return render_template("performance.html",
        leagues=config.LEAGUES,
        leagues_performance=leagues_performance,
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

@app.route("/tracker")
@login_required
def tracker_view():
    """Model Tracker -- automated prediction vs result tracking (all leagues)."""
    enabled_leagues = {code: lg for code, lg in config.LEAGUES.items() if lg.get("enabled")}

    # Query all tracker entries across all leagues
    entries = db.fetch_all(
        "SELECT * FROM model_tracker ORDER BY match_date DESC"
    )

    settled = [e for e in entries if e["status"] == "settled"]
    pending = [e for e in entries if e["status"] == "pending"]

    # -- Top pick stats --
    top_correct = sum(1 for e in settled if e["top_pick_correct"] == 1)
    top_total = len(settled)
    top_accuracy = (top_correct / top_total * 100) if top_total > 0 else 0
    top_pnl = sum(e["top_pick_pnl"] or 0 for e in settled)
    top_staked = top_total * 100
    top_roi = (top_pnl / top_staked * 100) if top_staked > 0 else 0

    # -- Value bet stats --
    value_entries = [e for e in settled if e["is_value_bet"] == 1]
    value_correct = sum(1 for e in value_entries if e["value_bet_correct"] == 1)
    value_total = len(value_entries)
    value_accuracy = (value_correct / value_total * 100) if value_total > 0 else 0
    value_pnl = sum(e["value_bet_pnl"] or 0 for e in value_entries)
    value_staked = value_total * 100
    value_roi = (value_pnl / value_staked * 100) if value_staked > 0 else 0
    avg_edge = (sum(e["value_edge"] or 0 for e in value_entries) / value_total) if value_total > 0 else 0

    # -- Current streak (top pick) --
    current_streak = 0
    streak_type = None
    for e in sorted(settled, key=lambda x: x["match_date"], reverse=True):
        if streak_type is None:
            streak_type = "W" if e["top_pick_correct"] == 1 else "L"
            current_streak = 1
        elif (streak_type == "W" and e["top_pick_correct"] == 1) or \
             (streak_type == "L" and e["top_pick_correct"] != 1):
            current_streak += 1
        else:
            break

    # -- Best/worst month --
    monthly_pnl = defaultdict(float)
    for e in settled:
        month_key = e["match_date"][:7] if e["match_date"] else "unknown"
        monthly_pnl[month_key] += e["top_pick_pnl"] or 0
    best_month = max(monthly_pnl.items(), key=lambda x: x[1]) if monthly_pnl else ("N/A", 0)
    worst_month = min(monthly_pnl.items(), key=lambda x: x[1]) if monthly_pnl else ("N/A", 0)

    # -- Breakdown by predicted outcome --
    outcome_stats = {}
    for outcome in ["H", "D", "A"]:
        oc_entries = [e for e in settled if e["predicted_outcome"] == outcome]
        oc_correct = sum(1 for e in oc_entries if e["top_pick_correct"] == 1)
        oc_total = len(oc_entries)
        outcome_stats[outcome] = {
            "total": oc_total,
            "correct": oc_correct,
            "accuracy": round(oc_correct / oc_total * 100, 1) if oc_total > 0 else 0,
        }

    # -- Per-league breakdown --
    league_breakdown = {}
    for code, lg_config in enabled_leagues.items():
        lg_settled = [e for e in settled if e.get("league") == code]
        lg_correct = sum(1 for e in lg_settled if e["top_pick_correct"] == 1)
        lg_total = len(lg_settled)
        lg_pnl = sum(e["top_pick_pnl"] or 0 for e in lg_settled)
        lg_staked = lg_total * 100
        league_breakdown[code] = {
            "config": lg_config,
            "settled": lg_total,
            "correct": lg_correct,
            "accuracy": round(lg_correct / lg_total * 100, 1) if lg_total > 0 else 0,
            "pnl": round(lg_pnl, 2),
            "roi": round(lg_pnl / lg_staked * 100, 1) if lg_staked > 0 else 0,
        }

    summary = {
        "total": len(entries),
        "settled": top_total,
        "pending": len(pending),
        "top_correct": top_correct,
        "top_accuracy": round(top_accuracy, 1),
        "top_pnl": round(top_pnl, 2),
        "top_staked": top_staked,
        "top_roi": round(top_roi, 1),
        "value_total": value_total,
        "value_correct": value_correct,
        "value_accuracy": round(value_accuracy, 1),
        "value_pnl": round(value_pnl, 2),
        "value_staked": value_staked,
        "value_roi": round(value_roi, 1),
        "avg_edge": round(avg_edge, 1),
        "current_streak": current_streak,
        "streak_type": streak_type or "N/A",
        "best_month": best_month[0],
        "best_month_pnl": round(best_month[1], 2),
        "worst_month": worst_month[0],
        "worst_month_pnl": round(worst_month[1], 2),
        "outcome_stats": outcome_stats,
    }

    return render_template("tracker.html",
        leagues=config.LEAGUES,
        entries=entries,
        summary=summary,
        league_breakdown=league_breakdown,
        now=datetime.utcnow(),
    )


@app.route("/api/tracker/settle", methods=["POST"])
@login_required
def api_tracker_settle():
    """Settle pending tracker entries against actual match results."""
    pending = db.fetch_all(
        "SELECT * FROM model_tracker WHERE status = 'pending'"
    )
    settled_count = 0
    for entry in pending:
        # Look for a completed match result
        match = db.fetch_one(
            """SELECT * FROM matches WHERE league = ? AND home_team = ? AND away_team = ?
               AND match_date LIKE ? AND ft_result IS NOT NULL
               ORDER BY match_date DESC LIMIT 1""",
            [entry["league"], entry["home_team"], entry["away_team"],
             entry["match_date"][:10] + "%"]
        )
        if not match:
            continue

        actual_result = match["ft_result"]
        home_goals = match["ft_home_goals"]
        away_goals = match["ft_away_goals"]

        # Top pick P&L
        top_correct = 1 if entry["predicted_outcome"] == actual_result else 0
        predicted_odds = entry["predicted_odds"] or 2.0
        top_pnl = round(100 * predicted_odds - 100, 2) if top_correct else -100.0

        # Value bet P&L
        value_correct = None
        value_pnl = None
        if entry["is_value_bet"] == 1 and entry["value_bet_type"] and entry["value_odds"]:
            vb_type = entry["value_bet_type"]
            # Map value_bet_type to result
            vb_result_map = {"home_win": "H", "draw": "D", "away_win": "A"}
            vb_expected = vb_result_map.get(vb_type)
            if vb_expected:
                value_correct = 1 if actual_result == vb_expected else 0
                value_pnl = round(100 * entry["value_odds"] - 100, 2) if value_correct else -100.0

        db.execute(
            """UPDATE model_tracker SET
                actual_result = ?, actual_home_goals = ?, actual_away_goals = ?,
                top_pick_correct = ?, top_pick_pnl = ?,
                value_bet_correct = ?, value_bet_pnl = ?,
                status = 'settled', settled_at = datetime('now')
               WHERE id = ?""",
            [actual_result, home_goals, away_goals,
             top_correct, top_pnl, value_correct, value_pnl,
             entry["id"]]
        )
        settled_count += 1

    return jsonify({"status": "ok", "settled": settled_count, "checked": len(pending)})


@app.route("/api/tracker/generate", methods=["POST"])
@login_required
def api_tracker_generate():
    """Generate tracker entries from predictions that don't have one yet."""
    predictions = db.fetch_all(
        """SELECT p.* FROM predictions p
           WHERE p.ensemble_home IS NOT NULL
             AND NOT EXISTS (
                 SELECT 1 FROM model_tracker t
                 WHERE t.league = p.league AND t.match_date = p.match_date
                   AND t.home_team = p.home_team AND t.away_team = p.away_team
             )"""
    )
    created = 0
    for pred in predictions:
        home_prob = pred["ensemble_home"] or 0
        draw_prob = pred["ensemble_draw"] or 0
        away_prob = pred["ensemble_away"] or 0

        # Determine predicted outcome
        max_prob = max(home_prob, draw_prob, away_prob)
        if max_prob == 0:
            continue
        if home_prob == max_prob:
            predicted_outcome = "H"
        elif away_prob == max_prob:
            predicted_outcome = "A"
        else:
            predicted_outcome = "D"

        # Implied odds for predicted outcome
        predicted_odds = round(1 / max_prob, 2) if max_prob > 0 else None

        # Check for value bets on this match
        value_bets = db.fetch_all(
            """SELECT * FROM value_bets
               WHERE league = ? AND match_date = ? AND home_team = ? AND away_team = ?
               ORDER BY edge_percent DESC LIMIT 1""",
            [pred["league"], pred["match_date"], pred["home_team"], pred["away_team"]]
        )
        is_value_bet = 0
        value_bet_type = None
        value_edge = None
        value_odds = None
        if value_bets:
            vb = value_bets[0]
            if (vb.get("edge_percent") or 0) >= 5.0:
                is_value_bet = 1
                value_bet_type = vb["bet_type"]
                value_edge = vb["edge_percent"]
                value_odds = vb["best_odds"]

        # Get best bookmaker odds for the predicted outcome
        odds_map = {"H": "home_odds", "D": "draw_odds", "A": "away_odds"}
        odds_col = odds_map.get(predicted_outcome, "home_odds")
        best_odds_row = db.fetch_one(
            f"""SELECT MAX({odds_col}) as best FROM odds
                WHERE league = ? AND home_team = ? AND away_team = ?""",
            [pred["league"], pred["home_team"], pred["away_team"]]
        )
        if best_odds_row and best_odds_row["best"]:
            predicted_odds = best_odds_row["best"]

        db.upsert("model_tracker", {
            "league": pred["league"],
            "match_date": pred["match_date"],
            "home_team": pred["home_team"],
            "away_team": pred["away_team"],
            "predicted_outcome": predicted_outcome,
            "home_prob": home_prob,
            "draw_prob": draw_prob,
            "away_prob": away_prob,
            "predicted_odds": predicted_odds,
            "confidence": pred.get("confidence"),
            "is_value_bet": is_value_bet,
            "value_bet_type": value_bet_type,
            "value_edge": value_edge,
            "value_odds": value_odds,
            "status": "pending",
        }, ["league", "match_date", "home_team", "away_team"])
        created += 1

    return jsonify({"status": "ok", "created": created})


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
    """Manually trigger data refresh for ALL enabled leagues."""
    enabled_leagues = {code: lg for code, lg in config.LEAGUES.items() if lg.get("enabled")}
    results = {}

    action = request.json.get("action", "all") if request.json else "all"

    if action in ("all", "csv"):
        count = football_data_uk.download_all_leagues()
        results["csv_matches_imported"] = count
        if count > 0:
            results["note_csv"] = "New results found — rebuilding ratings"
            try:
                from scheduler import task_ratings
                task_ratings()
                results["ratings_rebuilt"] = True
            except Exception as e:
                logger.error("Rating rebuild failed: %s", e)

    if action in ("all", "fixtures"):
        total_fixtures = 0
        for code in enabled_leagues:
            try:
                fixtures = football_data_api.get_upcoming_fixtures(code)
                total_fixtures += len(fixtures)
            except Exception as e:
                logger.error("Fixtures fetch failed for %s: %s", code, e)
        results["fixtures_fetched"] = total_fixtures

    if action in ("all", "odds"):
        total_odds = 0
        for code in enabled_leagues:
            try:
                odds = odds_api.get_odds(code)
                total_odds += len(odds)
            except Exception as e:
                logger.error("Odds fetch failed for %s: %s", code, e)
        results["odds_events"] = total_odds

    # If we got new fixtures or odds, regenerate predictions
    if action == "all" and (results.get("fixtures_fetched", 0) > 0 or results.get("odds_events", 0) > 0):
        try:
            from scheduler import task_predictions
            task_predictions()
            results["predictions_regenerated"] = True
        except Exception as e:
            logger.error("Prediction generation failed: %s", e)

    # Track what was rate-limited
    from data.rate_limiter import get_usage_summary
    usage = get_usage_summary()
    rate_limited = [name for name, info in usage.items() if info["remaining"] <= 0]
    if rate_limited:
        results["rate_limited"] = rate_limited

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
