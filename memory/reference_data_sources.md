---
name: Data Sources
description: External data sources for ProjectSoccer with URLs, refresh schedules, and usage notes
type: reference
---

| Source | URL | Data | Refresh |
|--------|-----|------|---------|
| football-data.co.uk | https://www.football-data.co.uk/data.php | Historical match results + odds CSVs | Weekly after matchday |
| football-data.org | https://api.football-data.org/v4/ | Upcoming fixtures, standings, live scores | 2x/day |
| The Odds API | https://api.the-odds-api.com/v4/ | Bookmaker odds for upcoming matches | 3x on match days |
| Reddit (PRAW) | https://www.reddit.com/dev/api/ | Fan sentiment from r/soccer, r/PremierLeague | 1x/day |
| NewsAPI | https://newsapi.org/v2/ | News headlines for sentiment | 1x/day |
| FBref | https://fbref.com/ | xG and advanced stats (scrape carefully) | Weekly |

**Premier League CSV direct link:** https://www.football-data.co.uk/mmz4281/{season}/E0.csv
- Season format: 2425 for 2024-25, 2324 for 2023-24, etc.
