---
name: API Keys & Services
description: External API services used in ProjectSoccer with their keys and rate limits
type: project
---

**Services configured:**
- football-data.org API — fixtures, standings, schedules (token stored in .env)
- Reddit API via PRAW — sentiment analysis (client ID + secret stored in .env)
- NewsAPI.org — headline sentiment (key stored in .env)
- football-data.co.uk — historical CSVs (no auth needed)
- The Odds API — bookmaker odds for upcoming matches (key stored in .env)

**Why:** All free-tier services selected to keep the project zero-cost.

**How to apply:** All keys live in .env file only. Never hardcode. All API calls go through rate_limiter.py.
