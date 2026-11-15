"""Honest (walk-forward) test of the money line strategy on past seasons.

For each season the model is trained only on earlier seasons, then every bet the
recommender would have rated ORTA or GÜÇLÜ is "placed" at the historical closing
odds with a flat 1-unit stake.

    python3 backtest.py
"""
import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb

from src.Predict.Recommender import GOOD_EV, MODEL_WEIGHT, STRONG_EV
from src.Utils.Expected_Value import american_to_decimal
from src.Utils.tools import normalize_team_name

DATASET = 'dataset_2012-26'
NON_FEATURES = ['index', 'Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'index.1', 'TEAM_NAME.1', 'Date.1',
                'OU-Cover', 'OU']
FIRST_TEST_SEASON = 2017
PARAMS = {'max_depth': 3, 'eta': 0.02, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5,
          'objective': 'binary:logistic', 'seed': 1}
ROUNDS = 300


def load():
    with sqlite3.connect('Data/dataset.sqlite') as con:
        data = pd.read_sql_query(f'select * from "{DATASET}"', con)
    with sqlite3.connect('Data/OddsData.sqlite') as con:
        tables = pd.read_sql_query("select name from sqlite_master where type='table'", con).name
        odds = pd.concat([pd.read_sql_query(f'select Date, Home, ML_Home, ML_Away from "{table}"', con)
                          for table in tables])
    for col in ['ML_Home', 'ML_Away']:
        odds[col] = pd.to_numeric(odds[col], errors='coerce')
    odds['Date'] = odds['Date'].astype(str).str[:10]
    odds['Home'] = odds['Home'].map(normalize_team_name)
    data['Home'] = data['TEAM_NAME'].map(normalize_team_name)
    merged = data.drop_duplicates(['Date', 'Home']).merge(odds.drop_duplicates(['Date', 'Home']), on=['Date', 'Home'])
    dates = pd.to_datetime(merged['Date'])
    merged['season'] = np.where(dates.dt.month >= 9, dates.dt.year, dates.dt.year - 1)
    valid = (merged.ML_Home.abs() >= 100) & (merged.ML_Away.abs() >= 100)
    return merged[valid].sort_values('Date').reset_index(drop=True)


def main():
    games = load()
    features = [c for c in games.columns if c not in NON_FEATURES + ['Home', 'ML_Home', 'ML_Away', 'season']]
    assert len(features) == 106, features
    home_won = games['Home-Team-Win'].astype(int)
    dec_home = games.ML_Home.map(american_to_decimal)
    dec_away = games.ML_Away.map(american_to_decimal)
    market = (1 / dec_home) / (1 / dec_home + 1 / dec_away)

    print(f"{'Sezon':8} {'Maç':>5} {'Model isabet':>13} {'Piyasa isabet':>14} | "
          f"{'ORTA+ bahis':>11} {'Getiri':>7} | {'GÜÇLÜ bahis':>11} {'Getiri':>7}")
    totals = {GOOD_EV: [], STRONG_EV: []}
    for season in sorted(s for s in games.season.unique() if s >= FIRST_TEST_SEASON):
        label = f"{season}-{str(season + 1)[-2:]}"
        train, test = games.season < season, games.season == season
        model = xgb.train(PARAMS, xgb.DMatrix(games.loc[train, features].values.astype(float),
                                              label=home_won[train]), ROUNDS)
        model_prob = model.predict(xgb.DMatrix(games.loc[test, features].values.astype(float)))
        fair = MODEL_WEIGHT * model_prob + (1 - MODEL_WEIGHT) * market[test]
        won = home_won[test]
        cells = []
        for threshold in (GOOD_EV, STRONG_EV):
            bet_home = fair * dec_home[test] - 1 >= threshold
            bet_away = (1 - fair) * dec_away[test] - 1 >= threshold
            profit = np.concatenate([np.where(won[bet_home] == 1, dec_home[test][bet_home] - 1, -1),
                                     np.where(won[bet_away] == 0, dec_away[test][bet_away] - 1, -1)])
            totals[threshold].append(profit)
            cells.append(f"{len(profit):>11} {profit.mean() * 100:>+6.1f}%")
        print(f"{label:8} {test.sum():>5} {((model_prob > .5) == won).mean() * 100:>12.1f}% "
              f"{((market[test] > .5) == won).mean() * 100:>13.1f}% | {cells[0]} | {cells[1]}")
    cells = [f"{len(np.concatenate(p)):>11} {np.concatenate(p).mean() * 100:>+6.1f}%" for p in totals.values()]
    print(f"{'TOPLAM':8} {'':>5} {'':>13} {'':>14} | {cells[0]} | {cells[1]}")
    print("\nGetiri = yatırılan her 100 birim için kâr/zarar (sabit tutarla oynanırsa).")


if __name__ == '__main__':
    main()
