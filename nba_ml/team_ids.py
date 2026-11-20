"""NBA ↔ ESPN team ID / abbreviation maps (static; no network)."""

from __future__ import annotations

# ESPN site API team id → NBA stats TEAM_ID
ESPN_TO_NBA: dict[int, int] = {
    1: 1610612737,  # ATL
    2: 1610612738,  # BOS
    17: 1610612751,  # BKN
    30: 1610612766,  # CHA
    4: 1610612741,  # CHI
    5: 1610612739,  # CLE
    6: 1610612742,  # DAL
    7: 1610612743,  # DEN
    8: 1610612765,  # DET
    9: 1610612744,  # GSW
    10: 1610612745,  # HOU
    11: 1610612754,  # IND
    12: 1610612746,  # LAC
    13: 1610612747,  # LAL
    29: 1610612763,  # MEM
    14: 1610612748,  # MIA
    15: 1610612749,  # MIL
    16: 1610612750,  # MIN
    3: 1610612740,  # NOP
    18: 1610612752,  # NYK
    25: 1610612760,  # OKC
    19: 1610612753,  # ORL
    20: 1610612755,  # PHI
    21: 1610612756,  # PHX
    22: 1610612757,  # POR
    23: 1610612758,  # SAC
    24: 1610612759,  # SAS
    28: 1610612761,  # TOR
    26: 1610612762,  # UTA
    27: 1610612764,  # WAS
}

NBA_TO_ESPN: dict[int, int] = {v: k for k, v in ESPN_TO_NBA.items()}

# ESPN abbreviations that differ from nba_api
ESPN_ABBR_TO_NBA: dict[str, str] = {
    "GS": "GSW",
    "NY": "NYK",
    "NO": "NOP",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
}

NBA_ID_TO_ABBR: dict[int, str] = {
    1610612737: "ATL",
    1610612738: "BOS",
    1610612739: "CLE",
    1610612740: "NOP",
    1610612741: "CHI",
    1610612742: "DAL",
    1610612743: "DEN",
    1610612744: "GSW",
    1610612745: "HOU",
    1610612746: "LAC",
    1610612747: "LAL",
    1610612748: "MIA",
    1610612749: "MIL",
    1610612750: "MIN",
    1610612751: "BKN",
    1610612752: "NYK",
    1610612753: "ORL",
    1610612754: "IND",
    1610612755: "PHI",
    1610612756: "PHX",
    1610612757: "POR",
    1610612758: "SAC",
    1610612759: "SAS",
    1610612760: "OKC",
    1610612761: "TOR",
    1610612762: "UTA",
    1610612763: "MEM",
    1610612764: "WAS",
    1610612765: "DET",
    1610612766: "CHA",
}

NBA_ID_TO_NAME: dict[int, str] = {
    1610612737: "Atlanta Hawks",
    1610612738: "Boston Celtics",
    1610612739: "Cleveland Cavaliers",
    1610612740: "New Orleans Pelicans",
    1610612741: "Chicago Bulls",
    1610612742: "Dallas Mavericks",
    1610612743: "Denver Nuggets",
    1610612744: "Golden State Warriors",
    1610612745: "Houston Rockets",
    1610612746: "LA Clippers",
    1610612747: "Los Angeles Lakers",
    1610612748: "Miami Heat",
    1610612749: "Milwaukee Bucks",
    1610612750: "Minnesota Timberwolves",
    1610612751: "Brooklyn Nets",
    1610612752: "New York Knicks",
    1610612753: "Orlando Magic",
    1610612754: "Indiana Pacers",
    1610612755: "Philadelphia 76ers",
    1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers",
    1610612758: "Sacramento Kings",
    1610612759: "San Antonio Spurs",
    1610612760: "Oklahoma City Thunder",
    1610612761: "Toronto Raptors",
    1610612762: "Utah Jazz",
    1610612763: "Memphis Grizzlies",
    1610612764: "Washington Wizards",
    1610612765: "Detroit Pistons",
    1610612766: "Charlotte Hornets",
}


def espn_team_id_to_nba(espn_id: int | str) -> int | None:
    try:
        return ESPN_TO_NBA.get(int(espn_id))
    except (TypeError, ValueError):
        return None


def nba_abbr_from_espn(abbr: str | None) -> str | None:
    if not abbr:
        return None
    a = str(abbr).upper().strip()
    return ESPN_ABBR_TO_NBA.get(a, a)


def team_name(nba_team_id: int) -> str:
    return NBA_ID_TO_NAME.get(int(nba_team_id), f"Team {nba_team_id}")


def team_abbr(nba_team_id: int) -> str | None:
    return NBA_ID_TO_ABBR.get(int(nba_team_id))
