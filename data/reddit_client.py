"""
Reddit sentiment client using PRAW.
Pulls posts/comments from football subreddits and scores with VADER.
"""
import logging
from datetime import datetime

from database import db
from data.rate_limiter import can_call, record_call, check_cache, save_cache
import config

logger = logging.getLogger(__name__)

# Team search terms (handle nicknames and abbreviations)
TEAM_SEARCH_TERMS = {
    "Arsenal": ["arsenal", "gunners"],
    "Aston Villa": ["aston villa", "villa"],
    "AFC Bournemouth": ["bournemouth", "cherries"],
    "Brentford": ["brentford", "bees"],
    "Brighton & Hove Albion": ["brighton", "seagulls"],
    "Chelsea": ["chelsea", "blues"],
    "Crystal Palace": ["crystal palace", "palace", "eagles"],
    "Everton": ["everton", "toffees"],
    "Fulham": ["fulham", "cottagers"],
    "Ipswich Town": ["ipswich"],
    "Leicester City": ["leicester", "foxes"],
    "Liverpool": ["liverpool", "reds", "lfc"],
    "Manchester City": ["man city", "manchester city", "city", "mcfc"],
    "Manchester United": ["man united", "manchester united", "mufc", "red devils"],
    "Newcastle United": ["newcastle", "magpies", "nufc"],
    "Nottingham Forest": ["nottingham forest", "forest", "nffc"],
    "Southampton": ["southampton", "saints"],
    "Tottenham": ["tottenham", "spurs", "thfc"],
    "West Ham United": ["west ham", "hammers", "whufc"],
    "Wolverhampton": ["wolves", "wolverhampton"],
}


def _get_praw():
    """Initialize PRAW Reddit client."""
    try:
        import praw
        return praw.Reddit(
            client_id=config.REDDIT_CLIENT_ID,
            client_secret=config.REDDIT_CLIENT_SECRET,
            user_agent=config.REDDIT_USER_AGENT,
        )
    except ImportError:
        logger.error("praw not installed")
        return None
    except Exception as e:
        logger.error("Failed to initialize PRAW: %s", e)
        return None


def _analyze_sentiment(texts):
    """Score a list of texts using VADER sentiment analysis."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
    except ImportError:
        logger.error("vaderSentiment not installed")
        return {"compound": 0, "positive": 0, "negative": 0, "neutral": 0, "count": 0}

    if not texts:
        return {"compound": 0, "positive": 0, "negative": 0, "neutral": 0, "count": 0}

    scores = []
    pos = neg = neu = 0
    for text in texts:
        vs = analyzer.polarity_scores(text)
        scores.append(vs["compound"])
        if vs["compound"] > 0.05:
            pos += 1
        elif vs["compound"] < -0.05:
            neg += 1
        else:
            neu += 1

    return {
        "compound": sum(scores) / len(scores),
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "count": len(texts),
    }


def fetch_team_sentiment(team, league="PL"):
    """Fetch and score Reddit sentiment for a specific team."""
    cache_key = f"reddit_{team.replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d')}"
    cached = check_cache(cache_key, config.CACHE_TTL["sentiment"])
    if cached:
        return cached

    # Rate limiting is handled at the bulk level in fetch_all_teams()
    reddit = _get_praw()
    if not reddit:
        return None

    search_terms = TEAM_SEARCH_TERMS.get(team, [team.lower()])
    league_config = config.LEAGUES.get(league, {})
    subreddits = league_config.get("subreddits", ["soccer"])

    texts = []
    try:
        for sub_name in subreddits:
            subreddit = reddit.subreddit(sub_name)
            for term in search_terms[:2]:  # limit search terms
                for post in subreddit.search(term, time_filter="week", limit=10):
                    texts.append(post.title)
                    if post.selftext:
                        texts.append(post.selftext[:500])
                    # Top comments
                    post.comments.replace_more(limit=0)
                    for comment in post.comments[:5]:
                        texts.append(comment.body[:300])

        # rate limiting handled at bulk level
    except Exception as e:
        logger.error("Reddit fetch failed for %s: %s", team, e)
        return None

    sentiment = _analyze_sentiment(texts)
    result = {
        "team": team,
        "reddit_score": round(sentiment["compound"], 4),
        "reddit_volume": sentiment["count"],
        "reddit_positive": sentiment["positive"],
        "reddit_negative": sentiment["negative"],
        "reddit_neutral": sentiment["neutral"],
    }

    save_cache(cache_key, result)
    return result


def fetch_all_teams(league="PL"):
    """Fetch sentiment for all teams in a league. Counts as 1 API call."""
    if not can_call("reddit"):
        logger.warning("Rate limit reached for Reddit bulk pull")
        return {}
    record_call("reddit", f"bulk_pull/{league}")
    results = {}
    for team in TEAM_SEARCH_TERMS:
        sent = fetch_team_sentiment(team, league)
        if sent:
            results[team] = sent
            # Save to database
            db.upsert("sentiment", {
                "league": league,
                "team": team,
                "score_date": datetime.utcnow().strftime("%Y-%m-%d"),
                "reddit_score": sent["reddit_score"],
                "reddit_volume": sent["reddit_volume"],
                "reddit_positive": sent["reddit_positive"],
                "reddit_negative": sent["reddit_negative"],
                "reddit_neutral": sent["reddit_neutral"],
                "combined_score": sent["reddit_score"],  # will blend with news later
            }, ["league", "team", "score_date"])

    logger.info("Fetched sentiment for %d teams", len(results))
    return results
