"""Turns raw model probabilities + sportsbook odds into ranked bet recommendations.

In every honest test (see backtest.py and README) the bookmaker's odds predicted games
better than the models did, so a model probability is never trusted on its own: it is
blended with the bookmaker's no-vig probability, and a bet is only recommended when the
blended probability still leaves a clear expected profit at the offered price.
"""
from src.Utils.Expected_Value import (american_to_decimal, kelly_fraction, no_vig_probabilities,
                                      parse_american_odds)

MODEL_WEIGHT = 0.3          # weight of the model vs. the market in the "fair" probability
STRONG_EV = 0.15            # EV per unit staked needed for a GÜÇLÜ rating
GOOD_EV = 0.07              # ... for an ORTA rating
KELLY_MULTIPLIER = 0.25     # stake a quarter of full Kelly
MAX_STAKE_FRACTION = 0.03   # never more than 3% of bankroll on one bet
DEFAULT_TOTAL_PRICE = -110

STRONG, GOOD, PASS = 'GÜÇLÜ', 'ORTA', 'PAS'


def _rating(ev):
    if ev >= STRONG_EV:
        return STRONG
    if ev >= GOOD_EV:
        return GOOD
    return PASS


def _bet(game, kind, selection, odds, win_prob, model_prob, market_prob, bankroll):
    decimal = american_to_decimal(odds)
    ev = win_prob * decimal - 1
    rating = _rating(ev)
    stake_fraction = min(kelly_fraction(win_prob, odds) * KELLY_MULTIPLIER, MAX_STAKE_FRACTION)
    return {
        'game': f"{game['away']} @ {game['home']}",
        'type': kind,
        'selection': selection,
        'odds': odds,
        'decimal_odds': round(decimal, 2),
        'win_prob': round(win_prob, 4),
        'model_prob': round(model_prob, 4),
        'market_prob': None if market_prob is None else round(market_prob, 4),
        'ev_pct': round(ev * 100, 1),
        'rating': rating,
        'stake': round(bankroll * stake_fraction, 2) if rating != PASS else 0.0,
    }


def analyze(games, home_probs, over_probs, odds_rows, bankroll=1000.0):
    """One dict per game with the model's opinion and every bet that could be placed.

    odds_rows[i] = {'home_odds', 'away_odds', 'total', 'over_price', 'under_price'}
    (American prices, any of them may be None).
    """
    results = []
    for (home, away), home_prob, over_prob, row in zip(games, home_probs, over_probs, odds_rows):
        game = {'home': home, 'away': away, 'model_home_prob': round(home_prob, 4),
                'home_odds': parse_american_odds(row.get('home_odds')),
                'away_odds': parse_american_odds(row.get('away_odds')),
                'total': row.get('total'), 'bets': []}

        fair_home = home_prob
        market_home = market_away = None
        if game['home_odds'] and game['away_odds']:
            market_home, market_away = no_vig_probabilities(game['home_odds'], game['away_odds'])
            fair_home = MODEL_WEIGHT * home_prob + (1 - MODEL_WEIGHT) * market_home
        game['market_home_prob'] = None if market_home is None else round(market_home, 4)
        game['home_prob'] = round(fair_home, 4)
        game['winner'] = home if fair_home >= 0.5 else away
        game['winner_prob'] = round(max(fair_home, 1 - fair_home), 4)

        if game['home_odds'] and game['away_odds']:
            game['bets'].append(_bet(game, 'Maç Sonucu', f"{home} kazanır", game['home_odds'],
                                     fair_home, home_prob, market_home, bankroll))
            game['bets'].append(_bet(game, 'Maç Sonucu', f"{away} kazanır", game['away_odds'],
                                     1 - fair_home, 1 - home_prob, market_away, bankroll))

        game['ou_pick'] = game['ou_confidence'] = None
        if over_prob is not None and game['total'] is not None:
            over = over_prob >= 0.5
            confidence = max(over_prob, 1 - over_prob)
            game['ou_pick'] = 'ÜST' if over else 'ALT'
            game['ou_confidence'] = round(confidence, 4)
            over_price = parse_american_odds(row.get('over_price')) or DEFAULT_TOTAL_PRICE
            under_price = parse_american_odds(row.get('under_price')) or DEFAULT_TOTAL_PRICE
            market_over, market_under = no_vig_probabilities(over_price, under_price)
            market = market_over if over else market_under
            fair = MODEL_WEIGHT * confidence + (1 - MODEL_WEIGHT) * market
            game['bets'].append(_bet(game, 'Alt/Üst', f"{game['ou_pick']} {game['total']}",
                                     over_price if over else under_price, fair, confidence, market, bankroll))
        results.append(game)
    return results


def recommended_bets(analysis):
    """Bets worth placing, best first."""
    bets = [bet for game in analysis for bet in game['bets'] if bet['rating'] != PASS]
    return sorted(bets, key=lambda bet: bet['ev_pct'], reverse=True)
