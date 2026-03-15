"""
Centralized rate limiter for all API calls.
All external API calls MUST go through this module.
"""
import time
import logging
import json
import os
from datetime import datetime, timedelta

import config
from database import db

logger = logging.getLogger(__name__)

# In-memory tracking for current session
_call_history = {}


def can_call(api_name):
    """Check if we're allowed to call this API right now."""
    limit = config.RATE_LIMITS.get(api_name)
    if not limit:
        logger.warning("No rate limit configured for %s, allowing", api_name)
        return True

    max_calls = limit["calls"]
    period = limit["period_seconds"]
    cutoff = datetime.utcnow() - timedelta(seconds=period)

    # Check database for persistent tracking
    count_row = db.fetch_one(
        "SELECT COUNT(*) as cnt FROM api_calls WHERE api_name = ? AND called_at > ?",
        [api_name, cutoff.isoformat()]
    )
    count = count_row["cnt"] if count_row else 0

    if count >= max_calls:
        logger.warning(
            "Rate limit reached for %s: %d/%d calls in last %ds",
            api_name, count, max_calls, period
        )
        return False
    return True


def record_call(api_name, endpoint="", response_code=200, cached=False):
    """Record an API call for rate limiting."""
    db.execute(
        "INSERT INTO api_calls (api_name, endpoint, called_at, response_code, cached) VALUES (?, ?, ?, ?, ?)",
        [api_name, endpoint, datetime.utcnow().isoformat(), response_code, int(cached)]
    )


def get_usage_summary():
    """Get API usage summary for dashboard display."""
    summary = {}
    for api_name, limit in config.RATE_LIMITS.items():
        period = limit["period_seconds"]
        cutoff = datetime.utcnow() - timedelta(seconds=period)
        count_row = db.fetch_one(
            "SELECT COUNT(*) as cnt FROM api_calls WHERE api_name = ? AND called_at > ? AND cached = 0",
            [api_name, cutoff.isoformat()]
        )
        count = count_row["cnt"] if count_row else 0
        summary[api_name] = {
            "used": count,
            "limit": limit["calls"],
            "period_hours": period / 3600,
            "remaining": limit["calls"] - count,
        }
    return summary


def check_cache(cache_key, ttl_seconds):
    """Check if cached data exists and is fresh."""
    cache_file = os.path.join(config.CACHE_DIR, f"{cache_key}.json")
    if not os.path.exists(cache_file):
        return None
    age = time.time() - os.path.getmtime(cache_file)
    if age > ttl_seconds:
        return None
    with open(cache_file, "r") as f:
        return json.load(f)


def save_cache(cache_key, data):
    """Save data to cache."""
    cache_file = os.path.join(config.CACHE_DIR, f"{cache_key}.json")
    with open(cache_file, "w") as f:
        json.dump(data, f, default=str)
