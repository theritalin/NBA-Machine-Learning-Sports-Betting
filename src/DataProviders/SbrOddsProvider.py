from sbrscrape import Scoreboard

from src.Utils.tools import normalize_team_name


class SbrOddsProvider:
    """Fetches today's NBA odds for one sportsbook from sportsbookreview."""

    def __init__(self, sportsbook="fanduel"):
        sb = Scoreboard(sport="NBA")
        self.games = sb.games if hasattr(sb, 'games') else []
        self.sportsbook = sportsbook

    def get_odds(self):
        """Returns
        {home + ':' + away: {
            home: {'money_line_odds': val}, away: {'money_line_odds': val},
            'under_over_odds': total line, 'over_price': val, 'under_price': val}}
        Any value is None when the sportsbook has not posted it.
        """
        dict_res = {}
        for game in self.games:
            home_team_name = normalize_team_name(game['home_team'])
            away_team_name = normalize_team_name(game['away_team'])

            def book_value(key):
                return (game.get(key) or {}).get(self.sportsbook)

            dict_res[home_team_name + ':' + away_team_name] = {
                'under_over_odds': book_value('total'),
                'over_price': book_value('over_odds'),
                'under_price': book_value('under_odds'),
                home_team_name: {'money_line_odds': book_value('home_ml')},
                away_team_name: {'money_line_odds': book_value('away_ml')},
            }
        return dict_res
