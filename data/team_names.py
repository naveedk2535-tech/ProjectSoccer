"""
Centralized team name standardization.
All data sources must pass team names through standardise() before storage.
"""

# Maps all known variations to a canonical name
# Canonical names match what football-data.co.uk CSVs use
NAME_MAP = {
    # Premier League
    "Man United": "Manchester United",
    "Manchester United FC": "Manchester United",
    "Man City": "Manchester City",
    "Manchester City FC": "Manchester City",
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham": "Nottingham Forest",
    "Nottingham Forest FC": "Nottingham Forest",
    "Sheffield Utd": "Sheffield United",
    "Sheffield United FC": "Sheffield United",
    "Wolves": "Wolverhampton",
    "Wolverhampton Wanderers": "Wolverhampton",
    "Wolverhampton Wanderers FC": "Wolverhampton",
    "West Ham": "West Ham United",
    "West Ham United FC": "West Ham United",
    "Newcastle": "Newcastle United",
    "Newcastle United FC": "Newcastle United",
    "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham Hotspur FC": "Tottenham",
    "Leeds": "Leeds United",
    "Leeds United FC": "Leeds United",
    "Leicester": "Leicester City",
    "Leicester City FC": "Leicester City",
    "Brighton": "Brighton & Hove Albion",
    "Brighton and Hove Albion": "Brighton & Hove Albion",
    "Brighton & Hove Albion FC": "Brighton & Hove Albion",
    "Brighton and Hove Albion FC": "Brighton & Hove Albion",
    "West Brom": "West Bromwich Albion",
    "West Bromwich Albion FC": "West Bromwich Albion",
    "Ipswich": "Ipswich Town",
    "Ipswich Town FC": "Ipswich Town",
    "Luton": "Luton Town",
    "Luton Town FC": "Luton Town",
    "Norwich": "Norwich City",
    "Norwich City FC": "Norwich City",
    "Bournemouth": "AFC Bournemouth",
    "AFC Bournemouth FC": "AFC Bournemouth",
    "Arsenal FC": "Arsenal",
    "Chelsea FC": "Chelsea",
    "Liverpool FC": "Liverpool",
    "Everton FC": "Everton",
    "Aston Villa FC": "Aston Villa",
    "Fulham FC": "Fulham",
    "Crystal Palace FC": "Crystal Palace",
    "Brentford FC": "Brentford",
    "Burnley FC": "Burnley",
    "Southampton FC": "Southampton",
    "Watford FC": "Watford",
    "Sunderland AFC": "Sunderland",
    # Scottish
    "Celtic FC": "Celtic",
    "Rangers FC": "Rangers",
    # La Liga
    "Club Atlético de Madrid": "Atletico Madrid",
    "Atletico de Madrid": "Atletico Madrid",
    "Real Madrid CF": "Real Madrid",
    "FC Barcelona": "Barcelona",
    "Sevilla FC": "Sevilla",
    "Real Betis Balompié": "Real Betis",
    "Valencia CF": "Valencia",
    "Villarreal CF": "Villarreal",
    "Athletic Club": "Ath Bilbao",
    "Real Sociedad de Fútbol": "Real Sociedad",
    # Bundesliga
    "FC Bayern München": "Bayern Munich",
    "Bayern Munich": "Bayern Munich",
    "Borussia Dortmund": "Dortmund",
    "RB Leipzig": "RB Leipzig",
    "Bayer 04 Leverkusen": "Leverkusen",
    "VfB Stuttgart": "Stuttgart",
    # Serie A
    "FC Internazionale Milano": "Inter",
    "AC Milan": "AC Milan",
    "Juventus FC": "Juventus",
    "SSC Napoli": "Napoli",
    "AS Roma": "Roma",
    "SS Lazio": "Lazio",
    # Ligue 1
    "Paris Saint-Germain FC": "Paris SG",
    "Paris Saint-Germain": "Paris SG",
    "Olympique de Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "AS Monaco FC": "Monaco",
}


def standardise(name):
    """Standardise a team name to canonical format."""
    if not name:
        return name
    name = name.strip()
    # Direct lookup
    if name in NAME_MAP:
        return NAME_MAP[name]
    # Try stripping " FC", " AFC" suffix
    for suffix in [" FC", " AFC", " CF"]:
        stripped = name.rstrip(suffix) if name.endswith(suffix) else None
        if stripped and stripped in NAME_MAP:
            return NAME_MAP[stripped]
    return name
