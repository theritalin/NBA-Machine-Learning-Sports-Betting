import unittest

from src.Predict import Recommender
from src.Utils import Expected_Value


class TestOddsMath(unittest.TestCase):

    def test_decimal_conversion(self):
        self.assertAlmostEqual(Expected_Value.american_to_decimal(-200), 1.5)
        self.assertAlmostEqual(Expected_Value.american_to_decimal(150), 2.5)

    def test_no_vig_sums_to_one(self):
        home, away = Expected_Value.no_vig_probabilities(-110, -110)
        self.assertAlmostEqual(home, 0.5)
        self.assertAlmostEqual(home + away, 1)

    def test_kelly(self):
        self.assertAlmostEqual(Expected_Value.kelly_fraction(0.6, 100), 0.2)
        self.assertEqual(Expected_Value.kelly_fraction(0.4, 100), 0)

    def test_parse_odds(self):
        self.assertEqual(Expected_Value.parse_american_odds('-118'), -118)
        self.assertEqual(Expected_Value.parse_american_odds('2.50'), 150)
        self.assertEqual(Expected_Value.parse_american_odds('1,50'), -200)
        self.assertIsNone(Expected_Value.parse_american_odds(''))
        self.assertIsNone(Expected_Value.parse_american_odds(None))


class TestRecommender(unittest.TestCase):
    games = [['Boston Celtics', 'Atlanta Hawks']]

    def analyze(self, home_prob, over_prob, **odds):
        return Recommender.analyze(self.games, [home_prob], [over_prob], [odds], bankroll=1000)

    def test_value_bet_is_recommended(self):
        analysis = self.analyze(0.80, None, home_odds=100, away_odds=-120, total=None)
        bets = Recommender.recommended_bets(analysis)
        self.assertEqual([b['selection'] for b in bets], ['Boston Celtics kazanır'])
        self.assertGreater(bets[0]['stake'], 0)
        self.assertLessEqual(bets[0]['stake'], 1000 * Recommender.MAX_STAKE_FRACTION)

    def test_agreeing_with_market_is_a_pass(self):
        analysis = self.analyze(0.5, None, home_odds=-110, away_odds=-110, total=None)
        self.assertEqual(Recommender.recommended_bets(analysis), [])

    def test_missing_odds_gives_prediction_without_bets(self):
        game = self.analyze(0.6, 0.9, home_odds=None, away_odds=None, total=None)[0]
        self.assertEqual(game['bets'], [])
        self.assertEqual(game['winner'], 'Boston Celtics')

    def test_only_confident_totals_are_recommended(self):
        weak = self.analyze(0.5, 0.60, home_odds=None, away_odds=None, total=220.5)
        strong = self.analyze(0.5, 0.10, home_odds=None, away_odds=None, total=220.5)
        self.assertEqual(Recommender.recommended_bets(weak), [])
        self.assertEqual([b['selection'] for b in Recommender.recommended_bets(strong)], ['ALT 220.5'])


if __name__ == '__main__':
    unittest.main()
