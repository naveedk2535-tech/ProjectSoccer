# ProjectSoccer — CLAUDE.md

## Project Overview
A full end-to-end soccer prediction engine with a modern dark-themed dashboard. Predicts match outcomes using statistical models (Poisson, Dixon-Coles, Elo, XGBoost) combined with sentiment analysis, then compares predictions against bookmaker odds to identify value bets.

## Tech Stack
- **Backend:** Python 3.11+ / Flask
- **Database:** SQLite (lightweight, no config needed)
- **Frontend:** Tailwind CSS + Alpine.js + Chart.js (no build step required)
- **Deployment:** PythonAnywhere (path: /home/zziai38)
- **Source Control:** GitHub (naveedk2535-tech)

## Key Principles
1. **Start with Premier League** — get it working end-to-end, then expand to other leagues
2. **Rate limit everything** — never exceed free tier API limits
3. **Respect the data** — clean, validate, and cache aggressively
4. **Be honest about uncertainty** — football is high-variance, show confidence intervals
5. **Dashboard-first** — no CSV exports, everything is visual and interactive
6. **Dark theme, trading terminal aesthetic** — think Bloomberg meets FotMob

## API Rate Limits (ENFORCE THESE)
| API | Limit | Our Target | Cache Duration |
|-----|-------|------------|----------------|
| football-data.org | 10 req/min | 2 calls/day | 12 hours |
| The Odds API | 500 req/month | ~48/month | 4 hours on match days |
| Reddit (PRAW) | 100 req/min | 1 bulk pull/day | 24 hours |
| NewsAPI | 100 req/day | 1 call/day | 24 hours |
| football-data.co.uk | Unlimited | 1x/3 days | 3 days |

## Data Refresh Schedule
- **Historical CSVs:** Download once at setup, refresh weekly after matchday
- **Fixtures:** Check 2x/day (morning + evening)
- **Odds:** 3x on match days only (morning, pre-match, just before kickoff)
- **Sentiment:** 1x/day
- **Team ratings:** Recalculate after each completed matchday

## Directory Structure
```
ProjectSoccer/
├── CLAUDE.md              # This file — project rules
├── SKILLS.md              # Expert persona definition
├── MEMORY.md              # Memory index
├── .env                   # API keys (NEVER commit)
├── .gitignore
├── requirements.txt
├── app.py                 # Flask main application
├── config.py              # Settings, rate limits, constants
├── models/
│   ├── __init__.py
│   ├── poisson.py         # Poisson + Dixon-Coles model
│   ├── elo.py             # Elo rating system
│   ├── xgboost_model.py   # XGBoost classifier
│   ├── sentiment.py       # Sentiment scoring model
│   └── ensemble.py        # Weighted ensemble combiner
├── data/
│   ├── __init__.py
│   ├── football_data_uk.py   # CSV downloader (football-data.co.uk)
│   ├── football_data_api.py  # Fixtures API (football-data.org)
│   ├── odds_api.py           # The Odds API client
│   ├── reddit_client.py      # Reddit sentiment via PRAW
│   ├── news_client.py        # NewsAPI client
│   ├── rate_limiter.py       # Centralized rate limiting
│   └── cache/                # Local data cache (gitignored)
├── database/
│   ├── __init__.py
│   ├── db.py              # Database connection + helpers
│   └── schema.sql         # Table definitions
├── templates/
│   ├── base.html           # Layout with nav, dark theme
│   ├── dashboard.html      # Command center home
│   ├── match.html          # Match detail deep dive
│   ├── league.html         # League overview
│   ├── sentiment.html      # Sentiment tracker
│   ├── performance.html    # Model backtest results
│   └── components/         # Reusable template fragments
├── static/
│   ├── css/
│   ├── js/
│   └── img/
├── scheduler.py            # Automated data refresh tasks
└── backtest.py             # Historical validation runner
```

## Coding Standards
- All API calls go through `rate_limiter.py` — never call an API directly
- Cache all API responses locally with TTL
- Every model must expose: `predict(home_team, away_team) -> dict` returning P(H), P(D), P(A)
- Use logging, not print statements
- Keep functions small and testable
- Handle API failures gracefully — use cached data as fallback

## Models — Expected Interface
```python
# Every model must implement:
def predict(home_team: str, away_team: str) -> dict:
    """
    Returns:
        {
            "home_win": 0.45,    # probability
            "draw": 0.28,
            "away_win": 0.27,
            "confidence": 0.72,  # model confidence 0-1
            "details": {}        # model-specific breakdown
        }
    """
```

## Build Phases
1. **Phase 1 — Foundation:** Data pipeline + Poisson model + dashboard shell
2. **Phase 2 — Core Models:** Dixon-Coles + Elo + XGBoost + ensemble
3. **Phase 3 — Odds & Value:** Odds API + value bet engine + Kelly staking
4. **Phase 4 — Sentiment:** Reddit + NewsAPI + VADER scoring
5. **Phase 5 — Polish:** Backtest page, mobile responsive, PythonAnywhere deploy

## Security
- API keys in `.env` only — NEVER hardcode, NEVER commit
- `.env` is in `.gitignore`
- No user authentication needed (personal tool)
- Sanitize all external data before database insertion

## Notes
- Football is high-variance. A 60% prediction will lose 40% of the time. That's correct, not broken.
- The model finds VALUE, not winners. A team at 40% to win with odds implying 25% is a value bet even though they'll probably lose.
- Fractional Kelly (quarter-Kelly) for all stake suggestions — full Kelly is too aggressive.
