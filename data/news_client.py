"""
NewsAPI client for headline sentiment analysis.
Fetches news headlines per team and scores with VADER.
"""
import logging
from datetime import datetime, timedelta

import requests

from database import db
from data.rate_limiter import can_call, record_call, check_cache, save_cache
import config

logger = logging.getLogger(__name__)

BASE_URL = "https://newsapi.org/v2"

# Keywords that flag important events
FLAG_KEYWORDS = {
    "negative": ["injury", "injured", "sacked", "fired", "ban", "banned",
                  "suspended", "ruled out", "setback", "crisis", "defeat"],
    "positive": ["signs", "signing", "return", "returns", "fit", "boost",
                 "record", "contract", "extension", "comeback"],
}


def fetch_team_news(team, league="PL"):
    """Fetch news headlines for a team from NewsAPI."""
    cache_key = f"news_{team.replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d')}"
    cached = check_cache(cache_key, config.CACHE_TTL["sentiment"])
    if cached:
        return cached

    # Rate limiting handled at bulk level in fetch_all_teams()

    from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        response = requests.get(
            f"{BASE_URL}/everything",
            params={
                "q": f'"{team}" AND (football OR soccer OR premier league)',
                "from": from_date,
                "sortBy": "relevancy",
                "pageSize": 20,
                "language": "en",
                "apiKey": config.NEWS_API_KEY,
            },
            timeout=15
        )
        # rate limiting handled at bulk level

        if response.status_code != 200:
            logger.error("NewsAPI error %d for %s", response.status_code, team)
            return None

        data = response.json()
        articles = data.get("articles", [])

        # Extract text for sentiment
        texts = []
        flagged_keywords = []
        for article in articles:
            title = article.get("title", "")
            desc = article.get("description", "")
            if title:
                texts.append(title)
            if desc:
                texts.append(desc)

            # Check for flag keywords
            combined = (title + " " + desc).lower()
            for kw_type, keywords in FLAG_KEYWORDS.items():
                for kw in keywords:
                    if kw in combined:
                        flagged_keywords.append({"keyword": kw, "type": kw_type, "headline": title})

        # Score with VADER
        from data.reddit_client import _analyze_sentiment
        sentiment = _analyze_sentiment(texts)

        result = {
            "team": team,
            "news_score": round(sentiment["compound"], 4),
            "news_volume": len(articles),
            "flagged_keywords": flagged_keywords[:10],  # top 10 flags
        }

        save_cache(cache_key, result)
        return result

    except Exception as e:
        logger.error("NewsAPI fetch failed for %s: %s", team, e)
        return None


def fetch_all_teams(league="PL"):
    """Fetch news sentiment for all teams in a league. Counts as 1 API call."""
    if not can_call("newsapi"):
        logger.warning("Rate limit reached for NewsAPI bulk pull")
        return {}
    record_call("newsapi", f"bulk_pull/{league}")

    from data.reddit_client import TEAM_SEARCH_TERMS

    results = {}
    for team in TEAM_SEARCH_TERMS:
        news = fetch_team_news(team, league)
        if news:
            results[team] = news

            # Update sentiment table with news data
            existing = db.fetch_one(
                "SELECT * FROM sentiment WHERE league = ? AND team = ? AND score_date = ?",
                [league, team, datetime.utcnow().strftime("%Y-%m-%d")]
            )
            if existing:
                # Blend reddit + news
                reddit_score = existing.get("reddit_score") or 0
                news_score = news["news_score"]
                combined = reddit_score * 0.6 + news_score * 0.4  # reddit weighted more

                db.execute(
                    """UPDATE sentiment SET news_score = ?, news_volume = ?,
                       news_keywords = ?, combined_score = ?
                       WHERE league = ? AND team = ? AND score_date = ?""",
                    [news_score, news["news_volume"],
                     str(news["flagged_keywords"]), round(combined, 4),
                     league, team, datetime.utcnow().strftime("%Y-%m-%d")]
                )
            else:
                db.upsert("sentiment", {
                    "league": league,
                    "team": team,
                    "score_date": datetime.utcnow().strftime("%Y-%m-%d"),
                    "news_score": news["news_score"],
                    "news_volume": news["news_volume"],
                    "news_keywords": str(news["flagged_keywords"]),
                    "combined_score": news["news_score"],
                }, ["league", "team", "score_date"])

    logger.info("Fetched news sentiment for %d teams", len(results))
    return results
