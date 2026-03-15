"""
Sentiment analysis model.
Combines Reddit and News sentiment into a team-level signal.
Uses VADER for sentiment scoring (no LLM needed).
"""
import logging
from datetime import datetime, timedelta

from database import db
import config

logger = logging.getLogger(__name__)


def get_team_sentiment(team, league="PL", days=7):
    """Get aggregated sentiment for a team over recent days."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    rows = db.fetch_all(
        """SELECT * FROM sentiment
           WHERE league = ? AND team = ? AND score_date >= ?
           ORDER BY score_date DESC""",
        [league, team, cutoff]
    )

    if not rows:
        return {
            "combined_score": 0.0,
            "trend": 0.0,
            "volume": 0,
            "data_points": 0,
            "signal": "neutral",
        }

    scores = [r["combined_score"] for r in rows if r["combined_score"] is not None]
    volumes = [r.get("reddit_volume", 0) or 0 + (r.get("news_volume", 0) or 0) for r in rows]

    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Trend: compare last 3 days vs previous 4 days
    recent = scores[:3] if len(scores) >= 3 else scores
    older = scores[3:7] if len(scores) >= 4 else []
    trend = (sum(recent)/len(recent)) - (sum(older)/len(older)) if older else 0.0

    # Volume spike detection
    avg_volume = sum(volumes) / len(volumes) if volumes else 0
    latest_volume = volumes[0] if volumes else 0
    volume_spike = latest_volume > avg_volume * 2 if avg_volume > 0 else False

    # Signal classification
    if avg_score > 0.15:
        signal = "positive"
    elif avg_score < -0.15:
        signal = "negative"
    else:
        signal = "neutral"

    if volume_spike:
        signal += "_spike"

    return {
        "combined_score": round(avg_score, 4),
        "trend": round(trend, 4),
        "volume": sum(volumes),
        "latest_volume": latest_volume,
        "volume_spike": volume_spike,
        "data_points": len(rows),
        "signal": signal,
    }


def predict(home_team, away_team, league="PL"):
    """
    Generate prediction adjustment based on sentiment.
    Sentiment is a weak signal — it nudges probabilities slightly.
    """
    home_sent = get_team_sentiment(home_team, league)
    away_sent = get_team_sentiment(away_team, league)

    # Sentiment differential
    diff = home_sent["combined_score"] - away_sent["combined_score"]

    # Convert to small probability adjustment (max ±5%)
    adjustment = max(-0.05, min(0.05, diff * 0.15))

    # Base probabilities (neutral)
    base_home = 0.40
    base_draw = 0.28
    base_away = 0.32

    home_win = base_home + adjustment
    away_win = base_away - adjustment
    draw = 1 - home_win - away_win

    # Ensure valid probabilities
    total = home_win + draw + away_win
    home_win /= total
    draw /= total
    away_win /= total

    return {
        "home_win": round(home_win, 4),
        "draw": round(draw, 4),
        "away_win": round(away_win, 4),
        "confidence": round(abs(diff) * 0.3, 4),  # low confidence for sentiment
        "details": {
            "home_sentiment": home_sent,
            "away_sentiment": away_sent,
            "sentiment_differential": round(diff, 4),
            "adjustment": round(adjustment, 4),
        }
    }
