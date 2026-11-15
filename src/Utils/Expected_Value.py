"""Betting math helpers. All odds are American (e.g. -150, +130) unless noted."""


def payout(odds):
    """Profit on a winning 100-unit stake."""
    if odds > 0:
        return odds
    return (100 / (-1 * odds)) * 100


def expected_value(Pwin, odds):
    """Expected profit (in units) of a 100-unit stake."""
    Ploss = 1 - Pwin
    Mwin = payout(odds)
    return round((Pwin * Mwin) - (Ploss * 100), 2)


def american_to_decimal(odds):
    """-150 -> 1.67, +130 -> 2.30 (the format used by European/Turkish bookmakers)."""
    return 1 + payout(odds) / 100


def implied_probability(odds):
    """Break-even win probability of a price (includes the bookmaker's margin)."""
    return 1 / american_to_decimal(odds)


def no_vig_probabilities(odds_a, odds_b):
    """Bookmaker's 'true' opinion for a two-way market, with the margin removed."""
    a = implied_probability(odds_a)
    b = implied_probability(odds_b)
    return a / (a + b), b / (a + b)


def kelly_fraction(Pwin, odds):
    """Fraction of bankroll the Kelly criterion would stake (0 if the bet has no edge)."""
    b = payout(odds) / 100
    return max(0.0, (Pwin * b - (1 - Pwin)) / b)


def parse_american_odds(value):
    """Turn user/scraper input into an int American price, or None if unusable.

    Decimal prices (e.g. '1.85') are accepted too and converted.
    """
    if value is None:
        return None
    try:
        number = float(str(value).strip().replace(',', '.'))
    except ValueError:
        return None
    if 1 < number < 100:  # decimal odds
        if number >= 2:
            return int(round((number - 1) * 100))
        return int(round(-100 / (number - 1)))
    if abs(number) < 100:
        return None
    return int(round(number))
