"""Elo and recent-form features derived from game scores.

The single module used by the training set builder (Create_Games.py) AND live prediction (main.py),
so both sides produce identical numbers. Every feature of a game is computed from games played
BEFORE that game's date only; games on the same date never influence each other.
"""
import re
import sqlite3
from datetime import date as date_type
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# Old names of a franchise -> the current name. "LA Clippers" matches tools.normalize_team_name.
FRANCHISE = {
    'Los Angeles Clippers': 'LA Clippers',
    'New Jersey Nets': 'Brooklyn Nets',
    'Seattle SuperSonics': 'Oklahoma City Thunder',
    'Charlotte Bobcats': 'Charlotte Hornets',
    'New Orleans Hornets': 'New Orleans Pelicans',
}

FEATURE_COLUMNS = [
    'Elo-Home', 'Elo-Away', 'Elo-Diff', 'Elo-Prob-Home',
    'Form10-Margin-Home', 'Form10-Margin-Away', 'Form5-Margin-Home', 'Form5-Margin-Away',
    'Form10-WinPct-Home', 'Form10-WinPct-Away', 'HomeSplit-Margin-Home', 'AwaySplit-Margin-Away',
    'Games7-Home', 'Games7-Away', 'B2B-Home', 'B2B-Away',
]
HISTORY_COLUMNS = ['date', 'home', 'away', 'home_pts', 'away_pts', 'season']

ELO_START = 1500.0
ELO_HOME_ADVANTAGE = 100.0
ELO_MEAN = 1505.0
ELO_CARRY = 0.75
NEW_SEASON_GAP_DAYS = 90
NBA_TIMEZONE = ZoneInfo('America/New_York')  # same rule as main.rest_days
SEASON_TABLE_PATTERN = re.compile(r'(\d{4})-(\d{2})')


def franchise(name):
    name = (name or '').strip()
    return FRANCHISE.get(name, name)


def season_label(year):
    return f"{year}-{(year + 1) % 100:02d}"


def season_of(day):
    """Fallback season label from the calendar (NBA seasons start in October)."""
    return season_label(day.year if day.month >= 10 else day.year - 1)


def _to_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date_type):
        return value
    return datetime.strptime(str(value)[:10], '%Y-%m-%d').date()


def elo_expected(elo_home, elo_away):
    """Home win probability; the home advantage is added here."""
    return 1.0 / (1.0 + 10 ** (-(elo_home + ELO_HOME_ADVANTAGE - elo_away) / 400.0))


def _mean(values, empty):
    return sum(values) / len(values) if values else empty


class GameFeatureBuilder:
    """Processes games in date order. features_for() only sees games of EARLIER dates:
    update() holds a day's results back until a later date is asked for or updated."""

    def __init__(self):
        self.elo = {}
        self.results = {}  # team -> [(margin, won)] this season
        self.home_margins = {}  # team -> [margin] home games this season
        self.away_margins = {}  # team -> [margin] away games this season
        self.game_days = {}  # team -> sorted list of dates ever played
        self.season = None
        self.last_date = None  # newest date already applied
        self.latest_date = None  # newest date seen (applied or pending)
        self._pending = []

    # -- public API

    def known(self, team):
        return franchise(team) in self.game_days

    def features_for(self, date, home, away, season=None):
        day = _to_date(date)
        self._flush(before=day)
        self._roll_season(day, season)
        home, away = franchise(home), franchise(away)
        elo_home, elo_away = self.elo.get(home, ELO_START), self.elo.get(away, ELO_START)
        home_form, away_form = self.results.get(home, []), self.results.get(away, [])
        return {
            'Elo-Home': elo_home,
            'Elo-Away': elo_away,
            'Elo-Diff': elo_home + ELO_HOME_ADVANTAGE - elo_away,
            'Elo-Prob-Home': elo_expected(elo_home, elo_away),
            'Form10-Margin-Home': _mean([m for m, _ in home_form[-10:]], 0.0),
            'Form10-Margin-Away': _mean([m for m, _ in away_form[-10:]], 0.0),
            'Form5-Margin-Home': _mean([m for m, _ in home_form[-5:]], 0.0),
            'Form5-Margin-Away': _mean([m for m, _ in away_form[-5:]], 0.0),
            'Form10-WinPct-Home': _mean([w for _, w in home_form[-10:]], 0.5),
            'Form10-WinPct-Away': _mean([w for _, w in away_form[-10:]], 0.5),
            'HomeSplit-Margin-Home': _mean(self.home_margins.get(home, []), 0.0),
            'AwaySplit-Margin-Away': _mean(self.away_margins.get(away, []), 0.0),
            'Games7-Home': self._games_within(home, day, 7),
            'Games7-Away': self._games_within(away, day, 7),
            'B2B-Home': int(self._games_within(home, day, 1) > 0),
            'B2B-Away': int(self._games_within(away, day, 1) > 0),
        }

    def update(self, date, home, away, home_pts, away_pts, season=None):
        day = _to_date(date)
        self._flush(before=day)
        self._pending.append((day, franchise(home), franchise(away), float(home_pts), float(away_pts), season))
        self.latest_date = day if self.latest_date is None else max(self.latest_date, day)

    def replay(self, history):
        """Walk a history frame in order; yields (row, features) with the features known BEFORE the game."""
        for row in history.itertuples(index=False):
            season = getattr(row, 'season', None)
            season = None if season is None or pd.isna(season) else season
            features = self.features_for(row.date, row.home, row.away, season)
            self.update(row.date, row.home, row.away, row.home_pts, row.away_pts, season)
            yield row, features

    # -- internals

    def _games_within(self, team, day, days):
        return sum(1 for d in self.game_days.get(team, []) if 0 < (day - d).days <= days)

    def _roll_season(self, day, season):
        """Regress Elo to the mean and clear this-season windows when a new season begins."""
        label = season if season is not None else season_of(day)
        new = self.season is not None and (
            label != self.season or (self.last_date is not None and (day - self.last_date).days > NEW_SEASON_GAP_DAYS))
        if new:
            self.elo = {t: ELO_CARRY * e + (1 - ELO_CARRY) * ELO_MEAN for t, e in self.elo.items()}
            self.results, self.home_margins, self.away_margins = {}, {}, {}
        self.season = label
        if self.last_date is None or day > self.last_date:
            self.last_date = day  # reference for the 90-day gap rule: a season rolls over only once

    def _flush(self, before):
        """Apply every pending game dated before `before`."""
        due = [g for g in self._pending if g[0] < before]
        if not due:
            return
        self._pending = [g for g in self._pending if g[0] >= before]
        for day, home, away, home_pts, away_pts, season in sorted(due, key=lambda g: g[0]):
            self._roll_season(day, season)
            self._apply(day, home, away, home_pts, away_pts)

    def _apply(self, day, home, away, home_pts, away_pts):
        elo_home, elo_away = self.elo.get(home, ELO_START), self.elo.get(away, ELO_START)
        margin = home_pts - away_pts
        expected = elo_expected(elo_home, elo_away)
        if margin == 0:
            outcome, winner_gap = 0.5, 0.0
        else:
            outcome = 1.0 if margin > 0 else 0.0
            gap = elo_home + ELO_HOME_ADVANTAGE - elo_away
            winner_gap = gap if margin > 0 else -gap
        k = 20.0 * (abs(margin) + 3) ** 0.8 / (7.5 + 0.006 * winner_gap)
        delta = k * (outcome - expected)
        self.elo[home], self.elo[away] = elo_home + delta, elo_away - delta
        won = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        self.results.setdefault(home, []).append((margin, won))
        self.results.setdefault(away, []).append((-margin, 1.0 - won))
        self.home_margins.setdefault(home, []).append(margin)
        self.away_margins.setdefault(away, []).append(-margin)
        for team in (home, away):
            self.game_days.setdefault(team, []).append(day)
        self.last_date = max(self.last_date, day) if self.last_date else day


def feature_vector(features):
    """Feature dict -> list in FEATURE_COLUMNS order (the order the models are trained on)."""
    return [float(features[c]) for c in FEATURE_COLUMNS]


def select_odds_table(odds_con, season_key):
    candidates = [f"odds_{season_key}_new", f"odds_{season_key}", f"{season_key}_new", f"{season_key}"]
    for table_name in candidates:
        found = odds_con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                                 (table_name,)).fetchone()
        if found:
            return table_name
    return None


def _schedule_games(path, after):
    """Finished games of a fixture CSV dated after `after` (US Eastern dates)."""
    schedule = pd.read_csv(path)
    schedule = schedule[schedule['Result'].notna() & (schedule['Result'].astype(str).str.strip() != '')]
    if schedule.empty:
        return []
    played_on = pd.to_datetime(schedule['Date'], format='%d/%m/%Y %H:%M').dt.tz_localize('UTC') \
        .dt.tz_convert(NBA_TIMEZONE).dt.date
    match = re.search(r'nba-(\d{4})', str(path))
    season = season_label(int(match.group(1))) if match else None
    rows = []
    for day, home, away, result in zip(played_on, schedule['Home Team'], schedule['Away Team'], schedule['Result']):
        if after is not None and day <= after:
            continue
        home_pts, away_pts = (float(x) for x in str(result).split('-'))
        rows.append((day, franchise(home), franchise(away), home_pts, away_pts, season))
    return rows


def load_game_history(odds_db_path, schedule_csv_paths=()):
    """All games of every season in the odds database (2007-08 on, for Elo warm-up) plus finished
    games of the fixture CSVs dated AFTER the last odds date. Columns: HISTORY_COLUMNS, date-sorted,
    (date, home) unique."""
    rows = []
    with sqlite3.connect(odds_db_path) as con:
        names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        keys = sorted({f"{m.group(1)}-{m.group(2)}" for m in map(SEASON_TABLE_PATTERN.search, names) if m})
        for key in keys:
            table = select_odds_table(con, key)
            if not table:
                continue
            frame = pd.read_sql_query(f'SELECT Date, Home, Away, Points, Win_Margin FROM "{table}"', con)
            frame = frame.dropna(subset=['Date', 'Home', 'Away', 'Points', 'Win_Margin'])
            for d, home, away, points, margin in zip(frame['Date'], frame['Home'], frame['Away'],
                                                     frame['Points'], frame['Win_Margin']):
                points, margin = float(points), float(margin)
                rows.append((_to_date(d), franchise(home), franchise(away),
                             (points + margin) / 2, (points - margin) / 2, key))
    last_odds_date = max((r[0] for r in rows), default=None)
    for path in schedule_csv_paths:
        if path and Path(path).exists():
            rows.extend(_schedule_games(path, last_odds_date))
    history = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    history = history.sort_values('date', kind='stable').drop_duplicates(['date', 'home'], keep='first')
    return history.reset_index(drop=True)


def latest_game_date(history):
    """Newest game date in a history frame (a date), or None when empty."""
    return None if history is None or len(history) == 0 else _to_date(history['date'].max())


def feature_table(history):
    """Pre-game features of every game in the history: DataFrame [date, home] + FEATURE_COLUMNS."""
    builder = GameFeatureBuilder()
    records = []
    for row, features in builder.replay(history):
        records.append({'date': row.date.isoformat(), 'home': row.home, **features})
    return pd.DataFrame(records, columns=['date', 'home'] + FEATURE_COLUMNS)
