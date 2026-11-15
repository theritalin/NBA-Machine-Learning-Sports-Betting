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
    'Accept': 'application/json, text/plain, */*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nba.com/',
    'Connection': 'keep-alive'
}

TODAYS_GAMES_URL = 'https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json'

TEAM_STATS_URL = 'https://stats.nba.com/stats/leaguedashteamstats?' \
                 'Conference=&DateFrom=&DateTo=&Division=&GameScope=&' \
                 'GameSegment=&LastNGames=0&LeagueID=00&Location=&' \
                 'MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&' \
                 'PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&' \
                 'PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&' \
                 'Season={season}&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&' \
                 'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='


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


def get_todays_games_json(url=TODAYS_GAMES_URL):
    raw_data = requests.get(url, headers=games_header, timeout=REQUEST_TIMEOUT)
    return raw_data.json().get('scoreboard', {}).get('games', [])


def to_data_frame(data):
    try:
        data_list = data[0]
    except Exception as e:
        print(e)
        return pd.DataFrame(data={})
    return pd.DataFrame(data=data_list.get('rowSet'), columns=data_list.get('headers'))


def create_todays_games(input_list):
    games = []
    for game in input_list:
        home = game.get('homeTeam', {})
        away = game.get('awayTeam', {})
        home_team = normalize_team_name(f"{home.get('teamCity', '')} {home.get('teamName', '')}")
        away_team = normalize_team_name(f"{away.get('teamCity', '')} {away.get('teamName', '')}")
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
