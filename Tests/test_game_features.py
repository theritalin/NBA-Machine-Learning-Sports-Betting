import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.Utils import Game_Features as GF  # noqa: E402

D = date


def play(builder, day, home, away, home_pts, away_pts, season=None):
    """features_for then update, the way every caller walks a history."""
    features = builder.features_for(day, home, away, season)
    builder.update(day, home, away, home_pts, away_pts, season)
    return features


class LeakageTest(unittest.TestCase):
    GAMES = [(D(2024, 1, 1), 'A', 'B', 100, 90), (D(2024, 1, 3), 'B', 'C', 95, 100),
             (D(2024, 1, 5), 'A', 'C', 110, 108), (D(2024, 1, 8), 'C', 'A', 99, 101)]

    def run_games(self, games):
        builder = GF.GameFeatureBuilder()
        return [play(builder, *g) for g in games]

    def test_own_score_never_changes_own_features_but_changes_later_ones(self):
        base = self.run_games(self.GAMES)
        changed = list(self.GAMES)
        changed[2] = (D(2024, 1, 5), 'A', 'C', 70, 120)  # swing the third game
        other = self.run_games(changed)
        self.assertEqual(base[:3], other[:3])  # the swung game's OWN features (index 2) did not move
        self.assertNotEqual(base[3], other[3])  # the next game of A and C sees the new score

    def test_asking_after_the_update_still_gives_pre_game_features(self):
        # even a caller that updates BEFORE asking for the same game's features must not see its result
        honest = GF.GameFeatureBuilder()
        for g in self.GAMES[:2]:
            play(honest, *g)
        expected = honest.features_for(*self.GAMES[2][:3])
        careless = GF.GameFeatureBuilder()
        for g in self.GAMES[:2]:
            play(careless, *g)
        careless.update(*self.GAMES[2])
        self.assertEqual(careless.features_for(*self.GAMES[2][:3]), expected)

    def test_first_game_is_neutral(self):
        first = self.run_games(self.GAMES)[0]
        self.assertEqual(first['Elo-Home'], 1500)
        self.assertEqual(first['Elo-Diff'], 100)
        self.assertEqual(first['Form10-WinPct-Home'], 0.5)
        self.assertEqual(first['Games7-Home'], 0)

    def test_same_day_games_do_not_influence_each_other(self):
        day = D(2024, 1, 10)
        solo = GF.GameFeatureBuilder()
        play(solo, D(2024, 1, 8), 'A', 'B', 100, 90)
        play(solo, D(2024, 1, 8), 'C', 'D', 90, 100)
        expected_ca = solo.features_for(day, 'C', 'A')
        expected_db = solo.features_for(day, 'D', 'B')

        both = GF.GameFeatureBuilder()
        play(both, D(2024, 1, 8), 'A', 'B', 100, 90)
        play(both, D(2024, 1, 8), 'C', 'D', 90, 100)
        first = both.features_for(day, 'C', 'A')
        both.update(day, 'C', 'A', 120, 80)  # a result of the same day ...
        second = both.features_for(day, 'D', 'B')  # ... must not reach a game of that day
        self.assertEqual(first, expected_ca)
        self.assertEqual(second, expected_db)
        # the next day sees it
        self.assertNotEqual(both.features_for(D(2024, 1, 11), 'C', 'A'), first)


class EloTest(unittest.TestCase):
    def test_equal_teams_home_probability(self):
        features = GF.GameFeatureBuilder().features_for(D(2024, 1, 1), 'A', 'B')
        self.assertAlmostEqual(features['Elo-Prob-Home'], 0.64, places=2)
        self.assertAlmostEqual(features['Elo-Prob-Home'], 1 / (1 + 10 ** -0.25))

    def test_winner_gains_and_total_is_conserved(self):
        builder = GF.GameFeatureBuilder()
        play(builder, D(2024, 1, 1), 'A', 'B', 110, 100)
        builder.features_for(D(2024, 1, 2), 'A', 'B')  # applies the pending day
        self.assertGreater(builder.elo['A'], 1500)
        self.assertLess(builder.elo['B'], 1500)
        self.assertAlmostEqual(builder.elo['A'] + builder.elo['B'], 3000)
        play(builder, D(2024, 1, 3), 'B', 'A', 90, 120)  # away team wins
        builder.features_for(D(2024, 1, 4), 'A', 'B')
        self.assertAlmostEqual(sum(builder.elo.values()), 3000)

    def test_k_formula_for_first_game(self):
        builder = GF.GameFeatureBuilder()
        play(builder, D(2024, 1, 1), 'A', 'B', 110, 100)  # margin 10, winner gap +100 (home advantage)
        builder.features_for(D(2024, 1, 2), 'A', 'B')
        k = 20 * 13 ** 0.8 / (7.5 + 0.006 * 100)
        expected = 1 / (1 + 10 ** -0.25)
        self.assertAlmostEqual(builder.elo['A'], 1500 + k * (1 - expected))

    def test_upset_moves_more_than_expected_win(self):
        favourite, upset = GF.GameFeatureBuilder(), GF.GameFeatureBuilder()
        for b in (favourite, upset):
            b.elo.update({'A': 1700.0, 'B': 1400.0})
        play(favourite, D(2024, 1, 1), 'A', 'B', 110, 100)
        play(upset, D(2024, 1, 1), 'A', 'B', 100, 110)
        for b in (favourite, upset):
            b.features_for(D(2024, 1, 2), 'A', 'B')
        self.assertGreater(abs(upset.elo['A'] - 1700), abs(favourite.elo['A'] - 1700))

    def test_new_season_pulls_toward_the_mean_once(self):
        builder = GF.GameFeatureBuilder()
        for i in range(10):
            play(builder, D(2024, 1, 1 + 2 * i), 'A', 'B', 120, 100)
        builder.features_for(D(2024, 1, 25), 'A', 'B')
        a, b = builder.elo['A'], builder.elo['B']
        self.assertGreater(a, 1550)
        features = builder.features_for(D(2024, 10, 25), 'A', 'B')  # > 90 days later
        self.assertAlmostEqual(features['Elo-Home'], 0.75 * a + 0.25 * 1505)
        self.assertAlmostEqual(features['Elo-Away'], 0.75 * b + 0.25 * 1505)
        self.assertEqual(builder.features_for(D(2024, 10, 25), 'A', 'B'), features)  # asking again: no 2nd pull
        builder.update(D(2024, 10, 25), 'A', 'B', 100, 99)
        after = builder.features_for(D(2024, 10, 27), 'A', 'B')
        self.assertAlmostEqual(after['Elo-Home'] + after['Elo-Away'], features['Elo-Home'] + features['Elo-Away'])
        self.assertLess(abs(after['Elo-Home'] - features['Elo-Home']), 20)  # no hidden second regression

    def test_season_label_change_regresses_even_within_90_days(self):
        builder = GF.GameFeatureBuilder()
        for i in range(6):
            play(builder, D(2020, 10, 1 + i), 'A', 'B', 120, 100, season='2019-20')
        builder.features_for(D(2020, 10, 8), 'A', 'B', '2019-20')
        a = builder.elo['A']
        features = builder.features_for(D(2020, 12, 22), 'A', 'B', '2020-21')  # 77 days later
        self.assertAlmostEqual(features['Elo-Home'], 0.75 * a + 0.25 * 1505)


class FranchiseTest(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(GF.franchise('New Jersey Nets'), 'Brooklyn Nets')
        self.assertEqual(GF.franchise('Seattle SuperSonics'), 'Oklahoma City Thunder')
        self.assertEqual(GF.franchise('Charlotte Bobcats'), 'Charlotte Hornets')
        self.assertEqual(GF.franchise('New Orleans Hornets'), 'New Orleans Pelicans')
        self.assertEqual(GF.franchise('Los Angeles Clippers'), 'LA Clippers')
        self.assertEqual(GF.franchise('LA Clippers'), 'LA Clippers')
        self.assertEqual(GF.franchise('Boston Celtics'), 'Boston Celtics')

    def test_agrees_with_normalize_team_name(self):
        from src.Utils.tools import normalize_team_name
        self.assertEqual(GF.franchise('Los Angeles Clippers'), normalize_team_name('Los Angeles Clippers'))

    def test_old_and_new_name_share_one_elo(self):
        builder = GF.GameFeatureBuilder()
        play(builder, D(2010, 1, 1), 'New Jersey Nets', 'Boston Celtics', 100, 90)
        old = builder.features_for(D(2010, 1, 3), 'New Jersey Nets', 'Chicago Bulls')
        new = builder.features_for(D(2010, 1, 3), 'Brooklyn Nets', 'Chicago Bulls')
        self.assertEqual(old, new)
        self.assertGreater(new['Elo-Home'], 1500)
        self.assertTrue(builder.known('Brooklyn Nets'))
        self.assertFalse(builder.known('Brooklyn Netz'))  # a typo is detectable


class FormTest(unittest.TestCase):
    def test_windows_and_splits(self):
        builder = GF.GameFeatureBuilder()
        for i in range(12):  # A beats B at home by i+1, every 2 days
            play(builder, D(2024, 1, 1 + 2 * i), 'A', 'B', 100 + i + 1, 100)
        features = builder.features_for(D(2024, 2, 1), 'A', 'B')
        self.assertAlmostEqual(features['Form10-Margin-Home'], np.mean(range(3, 13)))
        self.assertAlmostEqual(features['Form5-Margin-Home'], np.mean(range(8, 13)))
        self.assertAlmostEqual(features['Form10-Margin-Away'], -np.mean(range(3, 13)))
        self.assertEqual(features['Form10-WinPct-Home'], 1.0)
        self.assertEqual(features['Form10-WinPct-Away'], 0.0)
        self.assertAlmostEqual(features['HomeSplit-Margin-Home'], np.mean(range(1, 13)))
        self.assertAlmostEqual(features['AwaySplit-Margin-Away'], -np.mean(range(1, 13)))

    def test_fewer_games_than_window_uses_what_exists(self):
        builder = GF.GameFeatureBuilder()
        play(builder, D(2024, 1, 1), 'A', 'B', 110, 100)  # A +10 at home
        play(builder, D(2024, 1, 3), 'C', 'A', 100, 106)  # A +6 away
        features = builder.features_for(D(2024, 1, 5), 'A', 'C')
        self.assertAlmostEqual(features['Form10-Margin-Home'], (10 + 6) / 2)
        self.assertEqual(features['Form10-WinPct-Home'], 1.0)
        self.assertEqual(features['HomeSplit-Margin-Home'], 10)
        self.assertEqual(features['AwaySplit-Margin-Away'], 0.0)  # C has not played away yet
        self.assertEqual(features['Form10-Margin-Away'], -6)

    def test_windows_do_not_cross_season_boundary(self):
        builder = GF.GameFeatureBuilder()
        for i in range(5):
            play(builder, D(2024, 3, 1 + 2 * i), 'A', 'B', 130, 100, season='2023-24')
        late = builder.features_for(D(2024, 4, 15), 'A', 'B', '2023-24')
        self.assertEqual(late['Form10-Margin-Home'], 30)
        new_season = builder.features_for(D(2024, 10, 25), 'A', 'B', '2024-25')
        for column in ('Form10-Margin-Home', 'Form5-Margin-Home', 'Form10-Margin-Away', 'HomeSplit-Margin-Home',
                       'AwaySplit-Margin-Away'):
            self.assertEqual(new_season[column], 0.0, column)
        self.assertEqual(new_season['Form10-WinPct-Home'], 0.5)

    def test_back_to_back_and_games_in_last_7_days(self):
        builder = GF.GameFeatureBuilder()
        play(builder, D(2024, 1, 1), 'A', 'B', 100, 90)
        play(builder, D(2024, 1, 3), 'A', 'C', 100, 90)
        play(builder, D(2024, 1, 4), 'A', 'D', 100, 90)
        play(builder, D(2024, 1, 4), 'E', 'F', 100, 90)
        features = builder.features_for(D(2024, 1, 5), 'A', 'B')
        self.assertEqual(features['B2B-Home'], 1)
        self.assertEqual(features['Games7-Home'], 3)
        self.assertEqual(features['B2B-Away'], 0)
        self.assertEqual(features['Games7-Away'], 1)
        later = builder.features_for(D(2024, 1, 9), 'A', 'B')  # Jan 1 is 8 days back, Jan 3 is 6
        self.assertEqual(later['Games7-Home'], 2)
        self.assertEqual(later['B2B-Home'], 0)
        same_day = builder.features_for(D(2024, 1, 4), 'A', 'E')  # games of the same day never count
        self.assertEqual(same_day['Games7-Home'], 2)

    def test_feature_columns_match_dict_and_order(self):
        features = GF.GameFeatureBuilder().features_for(D(2024, 1, 1), 'A', 'B')
        self.assertEqual(list(features), GF.FEATURE_COLUMNS)
        self.assertEqual(len(GF.FEATURE_COLUMNS), 16)
        self.assertEqual(GF.feature_vector(features)[:4], [1500.0, 1500.0, 100.0, features['Elo-Prob-Home']])


class LoadHistoryTest(unittest.TestCase):
    def test_csv_results_only_after_last_odds_date_and_utc_to_eastern(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / 'odds.sqlite'
            with sqlite3.connect(db) as con:
                pd.DataFrame([
                    {'Date': '2025-01-09', 'Home': 'Los Angeles Clippers', 'Away': 'New Jersey Nets',
                     'Points': 200, 'Win_Margin': 10},
                    {'Date': '2025-01-10', 'Home': 'Boston Celtics', 'Away': 'Miami Heat',
                     'Points': 210, 'Win_Margin': -4},
                ]).to_sql('odds_2024-25', con, index=False)
            csv = Path(tmp) / 'nba-2024-UTC.csv'
            pd.DataFrame([  # 02:00 UTC on the 11th is still the 10th in New York: not after the last odds date
                {'Date': '11/01/2025 02:00', 'Home Team': 'Boston Celtics', 'Away Team': 'Miami Heat',
                 'Result': '1 - 2'},
                {'Date': '12/01/2025 01:30', 'Home Team': 'Boston Celtics', 'Away Team': 'LA Clippers',
                 'Result': '120 - 100'},  # 20:30 on the 11th in New York
                {'Date': '13/01/2025 00:00', 'Home Team': 'Miami Heat', 'Away Team': 'Boston Celtics',
                 'Result': ''},
                {'Date': '14/01/2025 00:00', 'Home Team': 'Miami Heat', 'Away Team': 'Boston Celtics',
                 'Result': None},
            ]).to_csv(csv, index=False)
            history = GF.load_game_history(db, [csv, Path(tmp) / 'missing.csv'])
        self.assertEqual(list(history.columns), GF.HISTORY_COLUMNS)
        self.assertEqual(len(history), 3)
        self.assertEqual(list(history['date']), [D(2025, 1, 9), D(2025, 1, 10), D(2025, 1, 11)])
        first = history.iloc[0]
        self.assertEqual((first.home, first.away), ('LA Clippers', 'Brooklyn Nets'))
        self.assertEqual((first.home_pts, first.away_pts), (105.0, 95.0))  # (200+10)/2, (200-10)/2
        csv_game = history.iloc[2]
        self.assertEqual((csv_game.home, csv_game.away, csv_game.home_pts, csv_game.away_pts),
                         ('Boston Celtics', 'LA Clippers', 120.0, 100.0))
        self.assertEqual(GF.latest_game_date(history), D(2025, 1, 11))
        self.assertIsNone(GF.latest_game_date(history.iloc[0:0]))

    def test_unique_per_date_and_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / 'odds.sqlite'
            row = {'Date': '2025-01-09', 'Home': 'A', 'Away': 'B', 'Points': 200, 'Win_Margin': 10}
            with sqlite3.connect(db) as con:
                pd.DataFrame([row, row]).to_sql('odds_2024-25', con, index=False)
            self.assertEqual(len(GF.load_game_history(db)), 1)


class DatasetMatchesLiveTest(unittest.TestCase):
    """Counterpart test: the row Create_Games writes and what main.py computes for the same game."""

    def test_create_games_and_main_give_identical_features(self):
        spec = importlib.util.spec_from_file_location(
            'create_games_parity', BASE_DIR / 'src' / 'Process-Data' / 'Create_Games.py')
        create_games = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(create_games)
        import main as live

        teams = ['Team A', 'Team B', 'Team C', 'Team D']
        games = [('2024-01-02', 'Team A', 'Team B', 210, 6), ('2024-01-02', 'Team C', 'Team D', 200, -4),
                 ('2024-01-04', 'Team B', 'Team C', 190, 12), ('2024-01-05', 'Team A', 'Team C', 220, -2),
                 ('2024-01-07', 'Team D', 'Team A', 205, 9), ('2024-01-09', 'Team C', 'Team A', 215, 3)]
        odds_rows = [{'Date': d, 'Home': h, 'Away': a, 'OU': 205.5, 'Points': p, 'Win_Margin': m,
                      'Days_Rest_Home': 2, 'Days_Rest_Away': 2} for d, h, a, p, m in games]
        team_df = pd.DataFrame({'TEAM_ID': range(30), 'TEAM_NAME': [f'T{i}' for i in range(30)],
                                'STAT_A': range(30)})
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with sqlite3.connect(tmp / 'odds.sqlite') as con:
                pd.DataFrame(odds_rows).to_sql('2023-24', con, index=False)
            with sqlite3.connect(tmp / 'teams.sqlite') as con:
                for d, *_ in games:
                    previous = (pd.Timestamp(d) - pd.Timedelta(days=1)).date().isoformat()
                    team_df.to_sql(previous, con, if_exists='replace', index=False)
            with sqlite3.connect(tmp / 'truncated.sqlite') as con:
                pd.DataFrame(odds_rows[:-1]).to_sql('2023-24', con, index=False)
            (tmp / 'config.toml').write_text('[create-games.2023-24]\n')

            with mock.patch.multiple(create_games, CONFIG_PATH=tmp / 'config.toml', ODDS_DB_PATH=tmp / 'odds.sqlite',
                                     TEAMS_DB_PATH=tmp / 'teams.sqlite', OUTPUT_DB_PATH=tmp / 'out.sqlite',
                                     OUTPUT_TABLE='dataset_test',
                                     TEAM_INDEX_BY_SEASON={'2023-24': {t: i for i, t in enumerate(teams)}}):
                create_games.main()
            with sqlite3.connect(tmp / 'out.sqlite') as con:
                frame = pd.read_sql_query('SELECT * FROM "dataset_test"', con)

            self.assertEqual(len(frame), len(games))
            self.assertEqual(list(frame.columns[-16:]), GF.FEATURE_COLUMNS)
            self.assertEqual(list(frame.columns[-18:-16]), ['Days-Rest-Home', 'Days-Rest-Away'])
            # live: history holds every game BEFORE the last one; ask for the last game on its own date
            with mock.patch.object(live, 'ODDS_DB_PATH', tmp / 'truncated.sqlite'), \
                    mock.patch.object(live, 'SCHEDULE_DIR', tmp):
                extra = live.elo_features([('Team C', 'Team A')], '2023-24', today=D(2024, 1, 9))
        np.testing.assert_array_equal(extra[0], frame[GF.FEATURE_COLUMNS].iloc[-1].values)
        self.assertGreater(frame['Elo-Home'].nunique(), 2)  # rows differ, so the equality is not vacuous


if __name__ == '__main__':
    unittest.main()
