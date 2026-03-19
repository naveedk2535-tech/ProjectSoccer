#!/usr/bin/env python3
"""
ProjectSoccer Watchdog — Standalone monitoring script.
Runs health, security, data integrity, API, and model checks.

Usage:
    python watchdog.py

Exit codes:
    0 — all checks passed
    1 — warnings detected
    2 — critical failure
"""
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_PATH = os.path.join(DATA_DIR, "watchdog_config.json")
LOG_PATH = os.path.join(DATA_DIR, "watchdog.log")
LAST_PATH = os.path.join(DATA_DIR, "watchdog_last.json")
ALERT_PATH = os.path.join(DATA_DIR, "watchdog_alert")
DB_PATH = os.path.join(BASE_DIR, "projectsoccer.db")
MODEL_PATH = os.path.join(DATA_DIR, "cache", "xgboost_model.pkl")
ENV_PATH = os.path.join(BASE_DIR, ".env")
GITIGNORE_PATH = os.path.join(BASE_DIR, ".gitignore")

REQUIRED_KEYS = [
    "FOOTBALL_DATA_API_KEY",
    "ODDS_API_KEY",
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "NEWS_API_KEY",
]

KEY_TABLES = ["matches", "predictions", "odds", "fixtures", "team_ratings",
              "sentiment", "value_bets"]

# Patterns that suggest hardcoded secrets
SECRET_PATTERNS = [
    r'["\'](?:sk|pk|api|key|secret|token|password)[_-]?[A-Za-z0-9]{16,}["\']',
    r'(?:api_key|apikey|secret|token)\s*=\s*["\'][A-Za-z0-9]{16,}["\']',
]


def load_config():
    """Load watchdog configuration, returning defaults if missing."""
    defaults = {
        "min_matches": 1000,
        "min_predictions": 5,
        "min_odds": 100,
        "max_db_size_mb": 100,
        "max_model_age_days": 7,
        "count_drop_threshold": 0.5,
        "enabled": True,
        "last_counts": {},
    }
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        # Merge with defaults so new keys are always present
        for k, v in defaults.items():
            cfg.setdefault(k, v)
        return cfg
    except (FileNotFoundError, json.JSONDecodeError):
        return defaults


def save_config(cfg):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def make_result(name, category, status, message):
    return {
        "name": name,
        "category": category,
        "status": status,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


# ===== CHECK FUNCTIONS =====================================================

def check_db_connection():
    """Can we connect and query SQLite?"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT 1")
        conn.close()
        return make_result("Database Connection", "health", "pass",
                           "Successfully connected to SQLite database.")
    except Exception as e:
        return make_result("Database Connection", "health", "critical",
                           f"Cannot connect to database: {e}")


def check_db_size(cfg):
    max_mb = cfg.get("max_db_size_mb", 100)
    if not os.path.exists(DB_PATH):
        return make_result("Database Size", "health", "critical",
                           "Database file does not exist.")
    size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    if size_mb >= max_mb:
        return make_result("Database Size", "health", "critical",
                           f"Database is {size_mb:.1f} MB (limit {max_mb} MB).")
    if size_mb >= max_mb * 0.8:
        return make_result("Database Size", "health", "warn",
                           f"Database is {size_mb:.1f} MB, approaching {max_mb} MB limit.")
    return make_result("Database Size", "health", "pass",
                       f"Database size is {size_mb:.1f} MB.")


def check_key_tables():
    """Verify key tables exist and have data."""
    try:
        conn = sqlite3.connect(DB_PATH)
        missing = []
        empty = []
        for table in KEY_TABLES:
            try:
                cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                if count == 0:
                    empty.append(table)
            except sqlite3.OperationalError:
                missing.append(table)
        conn.close()
        if missing:
            return make_result("Key Tables", "health", "critical",
                               f"Missing tables: {', '.join(missing)}")
        if empty:
            return make_result("Key Tables", "health", "warn",
                               f"Empty tables: {', '.join(empty)}")
        return make_result("Key Tables", "health", "pass",
                           "All key tables exist and contain data.")
    except Exception as e:
        return make_result("Key Tables", "health", "critical", str(e))


def check_app_import():
    """Can we import the app module?"""
    try:
        # Only attempt the import if we haven't already imported it
        sys.path.insert(0, BASE_DIR)
        __import__("config")
        return make_result("App Import", "health", "pass",
                           "config module imports without errors.")
    except Exception as e:
        return make_result("App Import", "health", "critical",
                           f"Cannot import config: {e}")


def check_dependencies():
    """Check that all required Python packages are installed and importable."""
    required = {
        "flask": "Flask (web framework)",
        "pandas": "Pandas (data processing)",
        "numpy": "NumPy (math)",
        "scipy": "SciPy (statistics)",
        "requests": "Requests (HTTP client)",
        "praw": "PRAW (Reddit API)",
        "vaderSentiment.vaderSentiment": "VADER (sentiment scoring)",
        "xgboost": "XGBoost (ML model)",
        "sklearn": "scikit-learn (ML utilities)",
        "dotenv": "python-dotenv (env loading)",
    }
    missing = []
    installed = []
    for module, label in required.items():
        try:
            __import__(module)
            installed.append(label)
        except ImportError:
            missing.append(label)

    if missing:
        return make_result("Dependencies", "health", "critical",
                           f"{len(missing)} packages missing: {', '.join(missing)}")
    return make_result("Dependencies", "health", "pass",
                       f"All {len(installed)} required packages installed.")


def check_scheduled_tasks():
    """Check that scheduled task logs show recent successful runs."""
    log_dir = "/var/log"
    try:
        now = datetime.now(timezone.utc)
        stale_tasks = []
        checked = 0

        # Check PA schedule log files
        for fname in os.listdir(log_dir):
            if fname.startswith("schedule-log-") and fname.endswith(".log"):
                fpath = os.path.join(log_dir, fname)
                mtime = datetime.fromtimestamp(os.path.getmtime(fpath), tz=timezone.utc)
                age_hours = (now - mtime).total_seconds() / 3600
                checked += 1
                if age_hours > 48:
                    stale_tasks.append(f"{fname} ({age_hours:.0f}h old)")

        if checked == 0:
            return make_result("Scheduled Tasks", "health", "warn",
                               "No scheduled task logs found.")
        if stale_tasks:
            return make_result("Scheduled Tasks", "health", "warn",
                               f"{len(stale_tasks)} tasks stale: {', '.join(stale_tasks)}")
        return make_result("Scheduled Tasks", "health", "pass",
                           f"All {checked} scheduled tasks ran recently.")
    except Exception as e:
        return make_result("Scheduled Tasks", "health", "pass",
                           f"Cannot check task logs (local env): {str(e)[:50]}")


def check_disk_space():
    """Check available disk space."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(BASE_DIR)
        free_mb = free / (1024 * 1024)
        used_pct = used / total * 100
        if free_mb < 100:
            return make_result("Disk Space", "health", "critical",
                               f"Low disk space: {free_mb:.0f} MB free ({used_pct:.0f}% used)")
        if free_mb < 500:
            return make_result("Disk Space", "health", "warn",
                               f"Disk getting full: {free_mb:.0f} MB free ({used_pct:.0f}% used)")
        return make_result("Disk Space", "health", "pass",
                           f"{free_mb:.0f} MB free ({used_pct:.0f}% used)")
    except Exception as e:
        return make_result("Disk Space", "health", "warn", str(e))


def check_cache_health():
    """Check cache directory isn't bloated."""
    cache_dir = os.path.join(BASE_DIR, "data", "cache")
    try:
        if not os.path.exists(cache_dir):
            return make_result("Cache Health", "health", "pass", "No cache directory.")
        total_size = 0
        file_count = 0
        for f in os.listdir(cache_dir):
            fp = os.path.join(cache_dir, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
                file_count += 1
        size_mb = total_size / (1024 * 1024)
        if size_mb > 50:
            return make_result("Cache Health", "health", "warn",
                               f"Cache bloated: {size_mb:.1f} MB ({file_count} files)")
        return make_result("Cache Health", "health", "pass",
                           f"Cache: {size_mb:.1f} MB ({file_count} files)")
    except Exception as e:
        return make_result("Cache Health", "health", "warn", str(e))


def check_value_bet_integrity():
    """Check value bets have valid data and no orphaned entries."""
    try:
        conn = sqlite3.connect(DB_PATH)
        # Check for value bets with impossible values
        bad = conn.execute(
            "SELECT COUNT(*) FROM value_bets WHERE edge_percent < 0 OR model_probability <= 0 "
            "OR model_probability >= 1 OR best_odds <= 1"
        ).fetchone()[0]
        # Check for stale pending bets (match date > 3 days ago)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
        stale_pending = conn.execute(
            "SELECT COUNT(*) FROM value_bets WHERE result = 'pending' AND match_date < ?",
            (cutoff,)
        ).fetchone()[0]
        conn.close()

        issues = []
        if bad > 0:
            issues.append(f"{bad} invalid value bets")
        if stale_pending > 0:
            issues.append(f"{stale_pending} stale pending bets (match >3 days ago)")
        if issues:
            return make_result("Value Bet Integrity", "data", "warn",
                               "; ".join(issues))
        return make_result("Value Bet Integrity", "data", "pass",
                           "All value bets valid, no stale entries.")
    except Exception as e:
        return make_result("Value Bet Integrity", "data", "warn", str(e))


# ----- Security -----

def check_env_exists():
    if os.path.exists(ENV_PATH):
        return make_result(".env File", "security", "pass",
                           ".env file exists.")
    return make_result(".env File", "security", "critical",
                       ".env file is missing.")


def check_env_in_gitignore():
    if not os.path.exists(GITIGNORE_PATH):
        return make_result(".gitignore", "security", "warn",
                           ".gitignore file not found.")
    with open(GITIGNORE_PATH, "r") as f:
        content = f.read()
    if ".env" in content:
        return make_result(".gitignore", "security", "pass",
                           ".env is listed in .gitignore.")
    return make_result(".gitignore", "security", "critical",
                       ".env is NOT listed in .gitignore — secrets may be committed.")


def check_hardcoded_keys():
    """Scan .py files for hardcoded API key patterns."""
    flagged = []
    for root, _dirs, files in os.walk(BASE_DIR):
        # Skip venv, __pycache__, .git
        if any(skip in root for skip in ["venv", ".venv", "__pycache__", ".git", "node_modules"]):
            continue
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", errors="ignore") as f:
                    content = f.read()
                for pattern in SECRET_PATTERNS:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        rel = os.path.relpath(fpath, BASE_DIR)
                        flagged.append(rel)
                        break
            except Exception:
                pass
    if flagged:
        return make_result("Hardcoded Keys", "security", "warn",
                           f"Possible hardcoded secrets in: {', '.join(flagged[:5])}")
    return make_result("Hardcoded Keys", "security", "pass",
                       "No hardcoded API key patterns detected.")


def check_suspicious_files():
    """Look for .bak, .tmp, ~ files in project root."""
    suspicious = []
    for item in os.listdir(BASE_DIR):
        if item.endswith((".bak", ".tmp")) or item.endswith("~"):
            suspicious.append(item)
    if suspicious:
        return make_result("Suspicious Files", "security", "warn",
                           f"Found suspicious files: {', '.join(suspicious[:10])}")
    return make_result("Suspicious Files", "security", "pass",
                       "No suspicious .bak/.tmp/~ files found.")


# ----- Data Integrity -----

def _table_count(table):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def check_matches_count(cfg):
    count = _table_count("matches")
    minimum = cfg.get("min_matches", 1000)
    if count < minimum:
        return make_result("Matches Count", "data", "critical",
                           f"matches table has {count} rows (minimum {minimum}).")
    return make_result("Matches Count", "data", "pass",
                       f"matches table has {count} rows.")


def check_predictions_count(cfg):
    count = _table_count("predictions")
    minimum = cfg.get("min_predictions", 5)
    if count < minimum:
        return make_result("Predictions Count", "data", "warn",
                           f"predictions table has {count} rows (minimum {minimum}).")
    return make_result("Predictions Count", "data", "pass",
                       f"predictions table has {count} rows.")


def check_odds_count(cfg):
    count = _table_count("odds")
    minimum = cfg.get("min_odds", 100)
    if count < minimum:
        return make_result("Odds Count", "data", "warn",
                           f"odds table has {count} rows (minimum {minimum}).")
    return make_result("Odds Count", "data", "pass",
                       f"odds table has {count} rows.")


def check_count_drops(cfg):
    """Compare current counts to last saved counts for sudden drops."""
    threshold = cfg.get("count_drop_threshold", 0.5)
    last_counts = cfg.get("last_counts", {})
    if not last_counts:
        return make_result("Count Drops", "data", "pass",
                           "No previous counts to compare (first run).")
    drops = []
    for table in ["matches", "predictions", "odds"]:
        current = _table_count(table)
        previous = last_counts.get(table, 0)
        if previous > 0 and current < previous * threshold:
            drops.append(f"{table}: {previous} -> {current}")
    if drops:
        return make_result("Count Drops", "data", "critical",
                           f"Sudden count drops detected: {'; '.join(drops)}")
    return make_result("Count Drops", "data", "pass",
                       "No sudden count drops detected.")


def check_league_data():
    """Verify all enabled leagues have match data."""
    try:
        sys.path.insert(0, BASE_DIR)
        import config as cfg_mod
        enabled = [code for code, info in cfg_mod.LEAGUES.items() if info.get("enabled")]
        if not enabled:
            return make_result("League Data", "data", "warn",
                               "No leagues are enabled.")
        missing = []
        conn = sqlite3.connect(DB_PATH)
        for code in enabled:
            cur = conn.execute("SELECT COUNT(*) FROM matches WHERE league = ?", (code,))
            if cur.fetchone()[0] == 0:
                missing.append(code)
        conn.close()
        if missing:
            return make_result("League Data", "data", "warn",
                               f"Enabled leagues with no match data: {', '.join(missing)}")
        return make_result("League Data", "data", "pass",
                           f"All {len(enabled)} enabled leagues have data.")
    except Exception as e:
        return make_result("League Data", "data", "warn", f"Could not check leagues: {e}")


# ----- API Keys -----

def check_api_keys():
    """Verify all required API keys are present in environment."""
    # Load .env manually if needed
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH)
    except ImportError:
        pass

    missing = []
    empty = []
    for key in REQUIRED_KEYS:
        val = os.environ.get(key)
        if val is None:
            missing.append(key)
        elif val.strip() == "":
            empty.append(key)

    results = []
    if missing:
        results.append(make_result("API Keys Present", "api", "critical",
                                   f"Missing API keys: {', '.join(missing)}"))
    elif empty:
        results.append(make_result("API Keys Present", "api", "warn",
                                   f"Empty API keys: {', '.join(empty)}"))
    else:
        results.append(make_result("API Keys Present", "api", "pass",
                                   f"All {len(REQUIRED_KEYS)} required API keys are set."))
    return results


# ----- Model Health -----

def check_model_exists():
    if os.path.exists(MODEL_PATH):
        return make_result("XGBoost Model File", "model", "pass",
                           "XGBoost model file exists.")
    return make_result("XGBoost Model File", "model", "warn",
                       "XGBoost model file not found (not yet trained).")


def check_model_staleness(cfg):
    max_days = cfg.get("max_model_age_days", 7)
    if not os.path.exists(MODEL_PATH):
        return make_result("Model Freshness", "model", "warn",
                           "Model file missing — cannot check staleness.")
    mtime = os.path.getmtime(MODEL_PATH)
    age_days = (time.time() - mtime) / 86400
    if age_days > max_days:
        return make_result("Model Freshness", "model", "warn",
                           f"Model is {age_days:.1f} days old (limit {max_days} days).")
    return make_result("Model Freshness", "model", "pass",
                       f"Model is {age_days:.1f} days old.")


def check_prediction_sums():
    """Check that ensemble predictions sum to approximately 1.0."""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT ensemble_home, ensemble_draw, ensemble_away "
            "FROM predictions ORDER BY id DESC LIMIT 20"
        ).fetchall()
        conn.close()
        if not rows:
            return make_result("Prediction Sums", "model", "warn",
                               "No predictions to verify sums.")
        bad = 0
        for h, d, a in rows:
            if h is None or d is None or a is None:
                continue
            total = h + d + a
            if abs(total - 1.0) > 0.05:
                bad += 1
        if bad > 0:
            return make_result("Prediction Sums", "model", "warn",
                               f"{bad}/{len(rows)} recent predictions do not sum to ~1.0.")
        return make_result("Prediction Sums", "model", "pass",
                           "Recent predictions sum to ~1.0.")
    except Exception as e:
        return make_result("Prediction Sums", "model", "warn", str(e))


def check_sentiment_freshness():
    """Check that sentiment data exists and is recent."""
    try:
        conn = sqlite3.connect(DB_PATH)
        # Check each enabled league
        leagues_with_data = 0
        leagues_stale = 0
        details = []
        cutoff = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")

        for league in ["PL", "PD", "BL1"]:
            row = conn.execute(
                "SELECT COUNT(DISTINCT team) as teams, MAX(score_date) as latest "
                "FROM sentiment WHERE league = ?", (league,)
            ).fetchone()
            teams = row[0] if row else 0
            latest = row[1] if row else None

            if teams == 0:
                details.append(f"{league}: no data")
            elif latest and latest < cutoff:
                leagues_stale += 1
                details.append(f"{league}: {teams} teams, stale (last: {latest})")
            else:
                leagues_with_data += 1
                details.append(f"{league}: {teams} teams, fresh")

        conn.close()

        if leagues_with_data == 0:
            return make_result("Sentiment Data", "data", "warn",
                               f"No fresh sentiment data. {'; '.join(details)}")
        if leagues_stale > 0:
            return make_result("Sentiment Data", "data", "warn",
                               f"{leagues_stale} league(s) have stale sentiment. {'; '.join(details)}")
        return make_result("Sentiment Data", "data", "pass",
                           f"Sentiment data fresh. {'; '.join(details)}")
    except Exception as e:
        return make_result("Sentiment Data", "data", "warn", str(e))


def check_api_call_failures():
    """Check for recent API call failures (non-200 responses)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        failures = conn.execute(
            "SELECT api_name, COUNT(*) as cnt FROM api_calls "
            "WHERE called_at > ? AND response_code != 200 AND response_code IS NOT NULL "
            "GROUP BY api_name", (cutoff,)
        ).fetchall()
        conn.close()

        if not failures:
            return make_result("API Call Failures", "api", "pass",
                               "No API failures in last 24 hours.")

        fail_details = [f"{row[0]}: {row[1]} failures" for row in failures]
        total = sum(row[1] for row in failures)
        if total > 10:
            return make_result("API Call Failures", "api", "critical",
                               f"{total} API failures in 24h. {', '.join(fail_details)}")
        return make_result("API Call Failures", "api", "warn",
                           f"{total} API failures in 24h. {', '.join(fail_details)}")
    except Exception as e:
        return make_result("API Call Failures", "api", "warn", str(e))


def check_rate_limit_exhaustion():
    """Check if any API rate limits are currently exhausted."""
    try:
        conn = sqlite3.connect(DB_PATH)
        now = datetime.now(timezone.utc)
        exhausted = []

        # Only monitor APIs where hitting the limit is unexpected
        # Excluded: football_data_uk (3/3 normal), reddit (6/6 normal = 2 scans × 3 leagues)
        # These APIs are expected to hit their limits daily as part of normal operation
        limits = {
            "football_data_org": {"calls": 10, "period": 60},
            "odds_api": {"calls": 12, "period": 86400},
            "newsapi": {"calls": 6, "period": 86400},
        }

        for api_name, limit in limits.items():
            cutoff = (now - timedelta(seconds=limit["period"])).isoformat()
            row = conn.execute(
                "SELECT COUNT(*) FROM api_calls WHERE api_name = ? AND called_at > ?",
                (api_name, cutoff)
            ).fetchone()
            used = row[0] if row else 0
            if used >= limit["calls"]:
                exhausted.append(f"{api_name}: {used}/{limit['calls']}")

        conn.close()

        if not exhausted:
            return make_result("Rate Limits", "api", "pass",
                               "All API rate limits have capacity.")
        if len(exhausted) >= 3:
            return make_result("Rate Limits", "api", "critical",
                               f"{len(exhausted)} APIs exhausted: {', '.join(exhausted)}")
        if len(exhausted) >= 1:
            return make_result("Rate Limits", "api", "critical",
                               f"Rate limit hit: {', '.join(exhausted)}")
        return make_result("Rate Limits", "api", "pass",
                           f"Some APIs at limit (normal): {', '.join(exhausted)}")
    except Exception as e:
        return make_result("Rate Limits", "api", "warn", str(e))


def check_data_freshness():
    """Check that key data sources have been updated recently."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        stale = []

        sources = {
            "football_data_org": "Fixtures & Standings",
            "odds_api": "Bookmaker Odds",
        }

        for api_name, label in sources.items():
            row = conn.execute(
                "SELECT MAX(called_at) as latest FROM api_calls "
                "WHERE api_name = ? AND cached = 0", (api_name,)
            ).fetchone()
            latest = row[0] if row and row[0] else None
            if not latest or latest < cutoff_24h:
                stale.append(f"{label}: {'never' if not latest else latest[:16]}")

        conn.close()

        if not stale:
            return make_result("Data Freshness", "data", "pass",
                               "All key data sources updated within 48 hours.")
        return make_result("Data Freshness", "data", "warn",
                           f"Stale data sources: {'; '.join(stale)}")
    except Exception as e:
        return make_result("Data Freshness", "data", "warn", str(e))


def check_tracker_accuracy():
    """Check model tracker has settled predictions and report accuracy."""
    try:
        conn = sqlite3.connect(DB_PATH)
        settled = conn.execute(
            "SELECT COUNT(*) FROM model_tracker WHERE status = 'settled'"
        ).fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM model_tracker WHERE status = 'pending'"
        ).fetchone()[0]

        if settled == 0:
            conn.close()
            return make_result("Model Tracker", "model", "pass",
                               f"No settled predictions yet. {pending} pending.")

        correct = conn.execute(
            "SELECT COUNT(*) FROM model_tracker WHERE status = 'settled' AND top_pick_correct = 1"
        ).fetchone()[0]
        accuracy = correct / settled * 100
        conn.close()

        if accuracy < 30:
            return make_result("Model Tracker", "model", "warn",
                               f"Low accuracy: {accuracy:.0f}% ({correct}/{settled}). {pending} pending.")
        return make_result("Model Tracker", "model", "pass",
                           f"Accuracy: {accuracy:.0f}% ({correct}/{settled}). {pending} pending.")
    except Exception as e:
        return make_result("Model Tracker", "model", "warn", str(e))


# ===== RUNNER ===============================================================

def run_all_checks():
    cfg = load_config()

    results = []

    # Health
    results.append(check_db_connection())
    results.append(check_db_size(cfg))
    results.append(check_key_tables())
    results.append(check_app_import())
    results.append(check_dependencies())
    results.append(check_disk_space())
    results.append(check_cache_health())
    results.append(check_scheduled_tasks())

    # Security
    results.append(check_env_exists())
    results.append(check_env_in_gitignore())
    results.append(check_hardcoded_keys())
    results.append(check_suspicious_files())

    # Data
    results.append(check_matches_count(cfg))
    results.append(check_predictions_count(cfg))
    results.append(check_odds_count(cfg))
    results.append(check_count_drops(cfg))
    results.append(check_league_data())
    results.append(check_sentiment_freshness())
    results.append(check_data_freshness())
    results.append(check_value_bet_integrity())

    # API
    results.extend(check_api_keys())
    results.append(check_api_call_failures())
    results.append(check_rate_limit_exhaustion())

    # Model
    results.append(check_model_exists())
    results.append(check_model_staleness(cfg))
    results.append(check_prediction_sums())
    results.append(check_tracker_accuracy())

    # Update last counts in config
    cfg["last_counts"] = {
        "matches": _table_count("matches"),
        "predictions": _table_count("predictions"),
        "odds": _table_count("odds"),
    }
    save_config(cfg)

    return results


def determine_overall(results):
    statuses = [r["status"] for r in results]
    if "critical" in statuses:
        return "critical"
    if "warn" in statuses:
        return "warn"
    return "pass"


def main():
    print("ProjectSoccer Watchdog — running checks...")
    results = run_all_checks()
    overall = determine_overall(results)
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    # Build summary
    summary = {
        "last_check": now_iso,
        "overall_status": overall,
        "total_checks": len(results),
        "passed": sum(1 for r in results if r["status"] == "pass"),
        "warnings": sum(1 for r in results if r["status"] == "warn"),
        "critical": sum(1 for r in results if r["status"] == "critical"),
        "results": results,
    }

    # Save last results
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LAST_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    # Append to log (JSON lines)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps({
            "timestamp": now_iso,
            "overall": overall,
            "passed": summary["passed"],
            "warnings": summary["warnings"],
            "critical": summary["critical"],
            "total": summary["total_checks"],
            "details": results,
        }) + "\n")

    # Alert flag
    if overall == "critical":
        with open(ALERT_PATH, "w") as f:
            f.write(now_iso)
        print(f"CRITICAL — {summary['critical']} critical issue(s) detected.")
    else:
        # Remove alert if no critical
        if os.path.exists(ALERT_PATH):
            os.remove(ALERT_PATH)

    # Print summary
    for r in results:
        icon = {"pass": "OK", "warn": "WARN", "critical": "CRIT"}[r["status"]]
        print(f"  [{icon:4s}] [{r['category']:8s}] {r['name']}: {r['message']}")

    print(f"\nOverall: {overall.upper()} "
          f"({summary['passed']} pass, {summary['warnings']} warn, "
          f"{summary['critical']} critical)")

    # Send email alert ONLY for critical issues (not warnings)
    # Warnings are logged and visible on the Watchdog tab
    if overall == "critical":
        try:
            from data.email_util import send_watchdog_alert
            send_watchdog_alert(overall, results,
                f"CRITICAL: {summary['critical']} critical issues found")
        except Exception as e:
            print(f"  Email alert failed: {e}")

    # Exit code
    code = {"pass": 0, "warn": 1, "critical": 2}[overall]
    sys.exit(code)


if __name__ == "__main__":
    main()
