import re
from datetime import datetime

import pandas as pd
import requests

from .Dictionaries import team_index_current

REQUEST_TIMEOUT = 30

games_header = {
    'user-agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/57.0.2987.133 Safari/537.36',
    'Dnt': '1',
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept-Language': 'en',
    'origin': 'http://stats.nba.com',
    'Referer': 'https://github.com'
}

data_headers = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Origin": "https://www.nba.com",
    "Priority": "u=3, i",
    "Referer": "https://www.nba.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0.1 Safari/605.1.15"
}

TODAYS_GAMES_URL = 'https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json'
LEGACY_TODAYS_GAMES_URL = 'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/{year}/scores/00_todays_scores.json'

TEAM_STATS_URL = "https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom=&DateTo=&Division=&" \
                 "GameScope=&GameSegment=&Height=&ISTRound=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&" \
                 "Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&" \
                 "PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season={season}&SeasonSegment=&" \
                 "SeasonType=Regular%20Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision="


def current_season(today=None):
    """NBA season string for a date, e.g. November 2026 -> '2026-27'."""
    today = today or datetime.now()
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def team_stats_url(season=None):
    return TEAM_STATS_URL.format(season=season or current_season())


def normalize_team_name(name):
    """Every data source must use the same names as stats.nba.com."""
    return (name or '').strip().replace('Los Angeles Clippers', 'LA Clippers')


def get_json_data(url):
    try:
        raw_data = requests.get(url, headers=data_headers, timeout=REQUEST_TIMEOUT)
        json = raw_data.json()
    except Exception as e:
        print(e)
        return {}
    return json.get('resultSets')


def get_todays_games():
    """Today's games as [[home, away], ...], trying the current NBA feed and then the legacy one."""
    try:
        games = requests.get(TODAYS_GAMES_URL, headers=games_header, timeout=REQUEST_TIMEOUT).json()
        return create_todays_games(games.get('scoreboard', {}).get('games', []))
    except Exception:
        url = LEGACY_TODAYS_GAMES_URL.format(year=current_season()[:4])
        games = requests.get(url, headers=games_header, timeout=REQUEST_TIMEOUT).json()
        return create_todays_games(games.get('gs', {}).get('g', []))


def to_data_frame(data):
    try:
        data_list = data[0]
    except Exception as e:
        print(e)
        return pd.DataFrame(data={})
    return pd.DataFrame(data=data_list.get('rowSet'), columns=data_list.get('headers'))


def create_todays_games(input_list):
    """Accepts games from either NBA feed (cdn.nba.com or the legacy data.nba.com format)."""
    games = []
    for game in input_list:
        if 'homeTeam' in game:
            home, away = game['homeTeam'], game['awayTeam']
            names = [f"{team.get('teamCity', '')} {team.get('teamName', '')}" for team in (home, away)]
        else:
            home, away = game.get('h', {}), game.get('v', {})
            names = [f"{team.get('tc', '')} {team.get('tn', '')}" for team in (home, away)]
        home_team, away_team = (normalize_team_name(name) for name in names)
        if home_team not in team_index_current or away_team not in team_index_current:
            continue
        games.append([home_team, away_team])
    return games


def create_todays_games_from_odds(input_dict):
    games = []
    for game in input_dict.keys():
        home_team, away_team = game.split(":")
        if home_team not in team_index_current or away_team not in team_index_current:
            continue
        games.append([home_team, away_team])
    return games


def get_date(date_string):
    year1, month, day = re.search(r'(\d+)-\d+-(\d\d)(\d\d)', date_string).groups()
    year = year1 if int(month) > 8 else int(year1) + 1
    return datetime.strptime(f"{year}-{month}-{day}", '%Y-%m-%d')
