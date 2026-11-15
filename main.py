import argparse
import json
import sys

import numpy as np
from colorama import Fore, Style, init

from src.Predict import Recommender, Report
from src.Utils.tools import (create_todays_games, create_todays_games_from_odds, get_json_data,
                             get_todays_games_json, normalize_team_name, team_stats_url, to_data_frame)

# The 52 per-team inputs the models were trained on, in training order (home team first, then away).
BASE_STATS = ['GP', 'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
              'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS']
STAT_COLUMNS = BASE_STATS + [f"{stat}_RANK" for stat in BASE_STATS]
MIN_GAMES_PLAYED = 10


def fail(message):
    print(Fore.RED + message + Style.RESET_ALL, file=sys.stderr)
    sys.exit(1)


def warn(message):
    print(Fore.YELLOW + message + Style.RESET_ALL, file=sys.stderr)


def fetch_games_and_odds(sportsbook):
    """Returns (games, odds_rows); odds_rows is None when odds must be typed in."""
    if not sportsbook:
        return create_todays_games(get_todays_games_json()), None

    from src.DataProviders.SbrOddsProvider import SbrOddsProvider
    odds = SbrOddsProvider(sportsbook=sportsbook).get_odds()
    games = create_todays_games_from_odds(odds)
    rows = []
    for home, away in games:
        game_odds = odds[f"{home}:{away}"]
        rows.append({'home_odds': game_odds[home]['money_line_odds'],
                     'away_odds': game_odds[away]['money_line_odds'],
                     'total': game_odds['under_over_odds'],
                     'over_price': game_odds.get('over_price'),
                     'under_price': game_odds.get('under_price')})
    return games, rows


def ask_odds(games):
    print("Oranları girin (ondalık 1.85 ya da Amerikan -118; bilinmiyorsa boş bırakın).")
    rows = []
    for home, away in games:
        print(Style.BRIGHT + f"\n{away} @ {home}" + Style.RESET_ALL)
        total = input("  Alt/Üst barajı (örn. 224.5): ").strip().replace(',', '.')
        try:
            total = float(total)
        except ValueError:
            total = None
        rows.append({'home_odds': input(f"  {home} kazanır oranı: "),
                     'away_odds': input(f"  {away} kazanır oranı: "),
                     'total': total})
    return rows


def build_features(games, stats):
    """One row per game: home team's stats followed by away team's stats."""
    missing = [col for col in ['TEAM_NAME'] + STAT_COLUMNS if col not in stats.columns]
    if missing:
        fail(f"stats.nba.com verisinde beklenen sütunlar yok: {missing[:5]}")
    by_name = stats.assign(TEAM_NAME=stats['TEAM_NAME'].map(normalize_team_name)).set_index('TEAM_NAME')
    features = []
    for home, away in games:
        for team in (home, away):
            if by_name.loc[team, 'GP'] < MIN_GAMES_PLAYED:
                warn(f"Uyarı: {team} henüz {int(by_name.loc[team, 'GP'])} maç oynadı; sezon başında tahminler güvenilmezdir.")
        features.append(np.concatenate([by_name.loc[home, STAT_COLUMNS].values,
                                        by_name.loc[away, STAT_COLUMNS].values]).astype(float))
    return np.array(features)


def main(args):
    init()
    games, odds_rows = fetch_games_and_odds(args.odds)
    if not games:
        fail("Bugün için maç bulunamadı." + (f" ({args.odds} henüz oran açmamış olabilir.)" if args.odds else ""))

    stats = to_data_frame(get_json_data(team_stats_url(args.season)))
    if stats.empty:
        fail("Takım istatistikleri stats.nba.com'dan alınamadı (site erişimi engelliyor ya da sezon başlamadı).")
    known = set(stats['TEAM_NAME'].map(normalize_team_name))
    keep = [i for i, game in enumerate(games) if game[0] in known and game[1] in known]
    games = [games[i] for i in keep]
    odds_rows = [odds_rows[i] for i in keep] if odds_rows is not None else ask_odds(games)

    if args.odds:
        no_price = [f"{away} @ {home}" for (home, away), row in zip(games, odds_rows)
                    if row['home_odds'] is None or row['away_odds'] is None]
        if no_price:
            warn(f"Uyarı: {args.odds} şu maçlara oran vermemiş, bahis önerilemez: {', '.join(no_price)}")

    features = build_features(games, stats)
    totals = [row.get('total') for row in odds_rows]

    runners = []
    if args.xgb or args.A or not args.nn:
        from src.Predict import XGBoost_Runner
        runners.append(('XGBoost', XGBoost_Runner))
    if args.nn or args.A:
        from src.Predict import NN_Runner
        runners.append(('Sinir Ağı', NN_Runner))

    output = {}
    for name, runner in runners:
        try:
            home_probs, over_probs = runner.predict(features, totals)
        except RuntimeError as e:
            warn(str(e))
            continue
        analysis = Recommender.analyze(games, home_probs, over_probs, odds_rows, args.kasa)
        output[name] = {'games': analysis, 'recommended': Recommender.recommended_bets(analysis)}
        if not args.json:
            Report.print_report(analysis, name, args.odds, args.kasa)
    if args.json:
        print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA maçları için bahis önerileri")
    parser.add_argument('-xgb', action='store_true', help='XGBoost modeli (varsayılan)')
    parser.add_argument('-nn', action='store_true', help='Sinir ağı modeli (TensorFlow gerekir)')
    parser.add_argument('-A', action='store_true', help='Bütün modeller')
    parser.add_argument('-odds', help='Oranların çekileceği site: fanduel, draftkings, betmgm, pointsbet, '
                                      'caesars, wynn, bet_rivers_ny. Verilmezse oranlar elle girilir.')
    parser.add_argument('-kasa', type=float, default=1000, help='Toplam bahis bütçeniz; önerilen tutarlar buna göre '
                                                                'hesaplanır (varsayılan 1000)')
    parser.add_argument('-season', help='Sezon, örn. 2026-27 (varsayılan: içinde bulunulan sezon)')
    parser.add_argument('-json', action='store_true', help='Sonuçları JSON olarak yazdır')
    main(parser.parse_args())
