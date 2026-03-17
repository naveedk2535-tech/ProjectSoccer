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

# Team search terms per league (handle nicknames and abbreviations)
LEAGUE_TEAM_SEARCH = {
    "PL": {
        "Arsenal": ["arsenal", "gunners"],
        "Aston Villa": ["aston villa", "villa"],
        "AFC Bournemouth": ["bournemouth", "cherries"],
        "Brentford": ["brentford"],
        "Brighton & Hove Albion": ["brighton", "seagulls"],
        "Chelsea": ["chelsea"],
        "Crystal Palace": ["crystal palace", "palace"],
        "Everton": ["everton", "toffees"],
        "Fulham": ["fulham"],
        "Ipswich Town": ["ipswich"],
        "Leeds United": ["leeds united", "leeds"],
        "Leicester City": ["leicester", "foxes"],
        "Liverpool": ["liverpool", "lfc"],
        "Manchester City": ["manchester city", "man city"],
        "Manchester United": ["manchester united", "man united"],
        "Newcastle United": ["newcastle", "magpies"],
        "Nottingham Forest": ["nottingham forest", "forest"],
        "Southampton": ["southampton", "saints"],
        "Sunderland": ["sunderland"],
        "Tottenham": ["tottenham", "spurs"],
        "West Ham United": ["west ham", "hammers"],
        "Wolverhampton": ["wolves", "wolverhampton"],
        "Burnley": ["burnley"],
    },
    "PD": {
        "Real Madrid": ["real madrid"],
        "Barcelona": ["barcelona", "barca"],
        "Atletico Madrid": ["atletico madrid", "atletico"],
        "Sevilla": ["sevilla fc"],
        "Real Sociedad": ["real sociedad"],
        "Real Betis": ["real betis", "betis"],
        "Villarreal": ["villarreal"],
        "Ath Bilbao": ["athletic bilbao", "athletic club"],
        "Valencia": ["valencia cf"],
        "Celta Vigo": ["celta vigo", "celta"],
        "Osasuna": ["osasuna"],
        "Getafe": ["getafe"],
        "Mallorca": ["mallorca"],
        "Alaves": ["alaves", "deportivo alaves"],
        "Girona": ["girona fc", "girona"],
        "Rayo Vallecano": ["rayo vallecano"],
        "Espanyol": ["espanyol"],
        "Leganes": ["leganes"],
        "Valladolid": ["valladolid"],
        "Las Palmas": ["las palmas"],
    },
    "BL1": {
        "Bayern Munich": ["bayern munich", "bayern"],
        "Dortmund": ["borussia dortmund", "dortmund", "bvb"],
        "RB Leipzig": ["rb leipzig", "leipzig"],
        "Leverkusen": ["bayer leverkusen", "leverkusen"],
        "Stuttgart": ["vfb stuttgart", "stuttgart"],
        "Ein Frankfurt": ["eintracht frankfurt", "frankfurt"],
        "Freiburg": ["sc freiburg", "freiburg"],
        "Wolfsburg": ["wolfsburg"],
        "Union Berlin": ["union berlin"],
        "Hoffenheim": ["hoffenheim"],
        "Werder Bremen": ["werder bremen"],
        "Augsburg": ["augsburg"],
        "Mainz": ["mainz 05", "mainz"],
        "Bochum": ["bochum"],
        "Heidenheim": ["heidenheim"],
        "St Pauli": ["st pauli"],
        "Holstein Kiel": ["holstein kiel", "kiel"],
        "M'gladbach": ["gladbach", "monchengladbach"],
    },
}

# Flat map for backward compatibility
TEAM_SEARCH_TERMS = {}
for league_teams in LEAGUE_TEAM_SEARCH.values():
    TEAM_SEARCH_TERMS.update(league_teams)


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
    cache_key = f"reddit_{team.replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d_%H')}"
    cached = check_cache(cache_key, 21600)  # 6 hour cache — allows 4 scans/day
    if cached:
        return cached

    # Rate limiting is handled at the bulk level in fetch_all_teams()
    reddit = _get_praw()
    if not reddit:
        return None

    league_teams = LEAGUE_TEAM_SEARCH.get(league, {})
    search_terms = league_teams.get(team, TEAM_SEARCH_TERMS.get(team, [team.lower()]))
    league_config = config.LEAGUES.get(league, {})
    subreddits = league_config.get("subreddits", ["soccer"])

    texts = []
    try:
        for sub_name in subreddits[:1]:  # only primary subreddit to save API calls
            subreddit = reddit.subreddit(sub_name)
            for term in search_terms[:1]:  # only first search term
                for post in subreddit.search(term, time_filter="week", limit=5):
                    texts.append(post.title)
                    if post.selftext:
                        texts.append(post.selftext[:300])
                    # Skip comments — titles are enough for sentiment

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
    league_teams = LEAGUE_TEAM_SEARCH.get(league, LEAGUE_TEAM_SEARCH.get("PL", {}))
    for team in league_teams:
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
