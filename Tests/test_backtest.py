import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import backtest
from src.Predict.Recommender import GOOD_EV, MODEL_WEIGHT


class TestMetrics(unittest.TestCase):
    def test_log_loss_known_values(self):
        self.assertAlmostEqual(backtest.log_loss([0.5, 0.5], [1, 0]), np.log(2))
        self.assertAlmostEqual(backtest.log_loss([0.8], [1]), -np.log(0.8))
        self.assertAlmostEqual(backtest.log_loss([0.8], [0]), -np.log(0.2))

    def test_log_loss_survives_certain_wrong_prediction(self):
        self.assertTrue(np.isfinite(backtest.log_loss([0.0], [1])))

    def test_brier(self):
        self.assertAlmostEqual(backtest.brier([0.8, 0.4], [1, 0]), (0.04 + 0.16) / 2)


class TestProfits(unittest.TestCase):
    def test_home_bet_win_and_loss(self):
        # fair .7 at decimal 2.0 -> EV .4; away side EV -0.4 -> no bet
        win = backtest.moneyline_profits([0.7], [1], [2.0], [2.0], GOOD_EV)
        loss = backtest.moneyline_profits([0.7], [0], [2.0], [2.0], GOOD_EV)
        self.assertEqual(list(win), [1.0])
        self.assertEqual(list(loss), [-1.0])

    def test_away_bet(self):
        p = backtest.moneyline_profits([0.2], [0], [2.0], [2.5], GOOD_EV)  # away fair .8*2.5-1=1.0
        self.assertEqual(list(p), [1.5])

    def test_threshold_and_no_bets(self):
        p = backtest.moneyline_profits([0.5], [1], [1.9], [1.9], GOOD_EV)
        self.assertEqual(len(p), 0)
        self.assertIsNone(backtest.mean_or_none(p))

    def test_ev_threshold_is_inclusive_and_matches_formula(self):
        # EV exactly .25 at fair .5, decimal 2.5
        self.assertEqual(len(backtest.moneyline_profits([0.5], [1], [2.5], [1.2], 0.25)), 1)
        self.assertEqual(len(backtest.moneyline_profits([0.5], [1], [2.5], [1.2], 0.26)), 0)

    def test_totals_blend_needs_strong_confidence(self):
        # confidence .6 -> fair .53 -> EV .012 at -110: no bet
        self.assertEqual(len(backtest.total_profits([0.6], [1])), 0)
        # confidence 1.0 -> fair .3+.35=.65 -> EV .65*1.909-1 = .241: bet
        fair = MODEL_WEIGHT * 1.0 + (1 - MODEL_WEIGHT) * 0.5
        self.assertGreater(fair * (1 + 100 / 110) - 1, GOOD_EV)
        over_win = backtest.total_profits([1.0], [1])
        under_win = backtest.total_profits([0.0], [0])
        lose = backtest.total_profits([1.0], [0])
        self.assertAlmostEqual(over_win[0], 100 / 110)
        self.assertAlmostEqual(under_win[0], 100 / 110)
        self.assertEqual(list(lose), [-1.0])


class TestBootstrap(unittest.TestCase):
    def test_deterministic(self):
        a = backtest.bootstrap_ratio_ci([1, -2, 3, 0.5], [10, 10, 10, 10])
        b = backtest.bootstrap_ratio_ci([1, -2, 3, 0.5], [10, 10, 10, 10])
        self.assertEqual(a, b)

    def test_constant_seasons_give_degenerate_interval(self):
        ci = backtest.bootstrap_ratio_ci([5, 5, 5], [10, 10, 10])
        self.assertAlmostEqual(ci['low'], 0.5)
        self.assertAlmostEqual(ci['high'], 0.5)
        self.assertFalse(ci['contains_zero'])

    def test_interval_brackets_point_and_zero_flag(self):
        ci = backtest.bootstrap_ratio_ci([3, -3, 2, -2, 1, -1], [10] * 6)
        self.assertLessEqual(ci['low'], ci['point'])
        self.assertGreaterEqual(ci['high'], ci['point'])
        self.assertTrue(ci['contains_zero'])

    def test_clearly_positive_excludes_zero(self):
        ci = backtest.bootstrap_ratio_ci([4, 5, 6, 5, 4, 6], [10] * 6)
        self.assertFalse(ci['contains_zero'])
        self.assertGreater(ci['low'], 0)

    def test_resamples_seasons_not_games(self):
        # one season is +1 per bet, another -1: block resample must reach both extremes
        ci = backtest.bootstrap_ratio_ci([10, -10], [10, 10])
        self.assertAlmostEqual(ci['low'], -1.0)
        self.assertAlmostEqual(ci['high'], 1.0)

    def test_no_bets_does_not_crash_or_fake_zero(self):
        ci = backtest.bootstrap_ratio_ci([0, 0], [0, 0])
        self.assertIsNone(ci['low'])
        self.assertIsNone(ci['contains_zero'])
        ci = backtest.bootstrap_ratio_ci([], [])
        self.assertIsNone(ci['point'])

    def test_some_empty_seasons(self):
        ci = backtest.bootstrap_ratio_ci([2, 0, 3], [4, 0, 6])
        self.assertAlmostEqual(ci['point'], 0.5)
        self.assertAlmostEqual(ci['low'], 0.5)

    def test_four_seasons_with_intermediate_values_resample_across_seasons(self):
        # Per-season ratios .1 .3 .5 .7. A real season-block bootstrap mixes seasons, so its 95% interval is
        # strictly inside [.1, .7] (all four draws equal has probability 1/256 < 2.5%). A bug that collapses
        # every resample to a single season would reach .1 and .7 exactly.
        ci = backtest.bootstrap_ratio_ci([1, 3, 5, 7], [10, 10, 10, 10])
        self.assertAlmostEqual(ci['point'], 0.4)
        self.assertGreater(ci['low'], 0.1 + 0.02)
        self.assertLess(ci['high'], 0.7 - 0.02)
        self.assertLess(ci['low'], ci['point'])
        self.assertGreater(ci['high'], ci['point'])

    def test_at_least_2000_reps_by_default(self):
        self.assertGreaterEqual(backtest.BOOTSTRAP_REPS, 2000)


class TestRecipe(unittest.TestCase):
    def test_default_when_no_meta(self):
        with tempfile.TemporaryDirectory() as d:
            params, rounds, source = backtest.load_recipe('ML', d)
        self.assertEqual(rounds, backtest.ROUNDS)
        self.assertEqual(params['objective'], 'binary:logistic')
        self.assertEqual(params['max_depth'], backtest.PARAMS['max_depth'])
        self.assertEqual(source, 'varsayılan')

    def test_real_models_dir_without_meta_falls_back(self):
        # existing model files are not *_meta.json
        if not any(f.endswith('_meta.json') for f in os.listdir(backtest.MODELS_DIR)):
            self.assertEqual(backtest.load_recipe('UO')[2], 'varsayılan')

    def _write(self, d, name, meta, mtime):
        path = os.path.join(d, name)
        with open(path, 'w') as f:
            json.dump(meta, f)
        os.utime(path, (mtime, mtime))

    def test_newest_meta_of_right_kind_wins_and_multiclass_is_converted(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, 'XGB_60%_ML_old_meta.json', {'params': {'max_depth': 2}, 'num_boost_round': 10}, 1000)
            self._write(d, 'XGB_61%_ML_new_meta.json',
                        {'params': {'max_depth': 7, 'objective': 'multi:softprob', 'num_class': 2,
                                    'eval_metric': 'mlogloss', 'eta': 0.1}, 'num_boost_round': 55}, 2000)
            self._write(d, 'XGB_50%_UO_x_meta.json', {'params': {'max_depth': 9}, 'num_boost_round': 99}, 3000)
            params, rounds, source = backtest.load_recipe('ML', d)
            self.assertEqual(source, 'XGB_61%_ML_new_meta.json')
            self.assertEqual(rounds, 55)
            self.assertEqual(params, {'max_depth': 7, 'objective': 'binary:logistic', 'eta': 0.1})
            self.assertEqual(backtest.load_recipe('UO', d)[1], 99)

    def test_bad_kind(self):
        with self.assertRaises(ValueError):
            backtest.load_recipe('XX')


class TestFeatures(unittest.TestCase):
    def test_all_numeric_non_target_columns_including_new_ones(self):
        df = pd.DataFrame({'index': [1.0], 'TEAM_NAME': ['A'], 'Date': ['2020-01-01'], 'Score': [200.0],
                           'Home-Team-Win': [1.0], 'OU': [210.5], 'OU-Cover': [1.0], 'Home': ['A'],
                           'ML_Home': [-110], 'ML_Away': [-110], 'season': [2019],
                           'W_PCT': [0.5], 'NEW_ELO': [1500.0], 'Days-Rest-Home': [2.0]})
        self.assertEqual(backtest.select_features(df.dtypes), ['W_PCT', 'NEW_ELO', 'Days-Rest-Home'])
        self.assertEqual(backtest.select_features(df.dtypes, include_ou=True),
                         ['OU', 'W_PCT', 'NEW_ELO', 'Days-Rest-Home'])


class TestDropFeaturesAndFewBets(unittest.TestCase):
    def test_drop_pattern_removes_only_matching_features(self):
        df = pd.DataFrame({'W_PCT': [0.5], 'Elo-Home': [1500.0], 'Form10-Margin-Home': [1.0],
                           'HomeSplit-Margin-Home': [1.0], 'B2B-Away': [0.0], 'Days-Rest-Home': [2.0],
                           'Score': [200.0]})
        pattern = '^(Elo|Form|HomeSplit|AwaySplit|Games7|B2B)'
        self.assertEqual(backtest.select_features(df.dtypes, drop_pattern=pattern), ['W_PCT', 'Days-Rest-Home'])
        self.assertEqual(len(backtest.select_features(df.dtypes)), 6)

    def test_ci_needs_at_least_30_bets(self):
        few = [np.array([1.0, -1.0] * 7)]  # 14 bets
        self.assertIsNone(backtest.ci_or_none(few, 14))
        enough = [np.array([1.0, -1.0] * 8), np.array([1.0, -1.0] * 8)]  # 32 bets
        self.assertIsNotNone(backtest.ci_or_none(enough, 32))
        self.assertIsNotNone(backtest.ci_or_none(enough, backtest.MIN_BETS_FOR_CI))
        self.assertIsNone(backtest.ci_or_none(enough, backtest.MIN_BETS_FOR_CI - 1))

    def test_few_bets_print_says_insufficient_and_json_is_null(self):
        import contextlib
        import io
        result = {'satirlar': [], 'toplam': {'mac': 0, 'bahis': 5, 'guven_araligi': {'getiri': None}}}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            backtest.print_totals(result)
        text = out.getvalue()
        self.assertIn('yetersiz bahis', text)
        self.assertNotIn('%95 GA', text)
        self.assertIsNone(json.loads(json.dumps(backtest._clean(result)))['toplam']['guven_araligi']['getiri'])


if __name__ == '__main__':
    unittest.main()
