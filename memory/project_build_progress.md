---
name: Build Progress
description: Tracks what has been built in ProjectSoccer and current phase status
type: project
---

## Current Status: Phase 1-4 COMPLETE, Phase 5-6 Remaining

### Completed [2026-03-15]
**Phase 1 — Foundation**
- 1,051 PL matches loaded (3 seasons: 22/23, 23/24, 24/25)
- SQLite database with 10 tables, all populated
- Rate limiter with per-API tracking and caching
- Team name standardization across all data sources
- Git repo: https://github.com/naveedk2535-tech/ProjectSoccer

**Phase 2 — Core Models (ALL 4 RUNNING)**
- Poisson + Dixon-Coles: scoreline matrix, O2.5, BTTS
- Elo: ratings with time decay, form, streaks, home/away splits
- XGBoost: 20 features, trained on 631 matches (H2H 33% importance)
- Ensemble: weighted blend (Poisson 35%, XGBoost 30%, Elo 25%, Sentiment 10%)

**Phase 3 — Odds & Value**
- The Odds API: 552 odds rows, 19 events, 40 bookmakers
- Margin stripping + implied probability calculation
- 5 value bets identified for upcoming gameweek
- Kelly Criterion (quarter-Kelly) stake suggestions

**Phase 4 — Sentiment**
- Reddit sentiment for 23 PL teams via PRAW + VADER
- NewsAPI headlines with keyword flagging (injury, sacked, ban, etc.)
- Combined sentiment score per team

**Phase 5 — Backtest & Validation**
- Backtest on 631 matches: 54.4% accuracy, Brier 0.574
- Calibration: well calibrated (50% predicted → 51.9% observed)
- Value bet ROI: +31.3% on historical data
- 25 referee profiles, 298 H2H records

**Dashboard — ALL 5 PAGES WORKING**
- Command Center: fixtures + predictions + value bets + standings
- Match Detail: 4-model breakdown, scoreline chart, H2H, form, odds
- League View: standings table + Elo ratings
- Sentiment Tracker: per-team gauges
- Model Performance: backtest results

### Remaining
- Phase 6: Deploy to PythonAnywhere (/home/zziai38)
- Set up scheduled tasks (daily/weekly cron)
- Add remaining 6 leagues
- Mobile responsive polish
