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

NON_FEATURES = ['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU']
FIRST_TEST_SEASON = '2017-18'


def load():
    with sqlite3.connect('Data/dataset.sqlite') as con:
        data = pd.read_sql_query('select * from "dataset_2012-23"', con, index_col='index')
    odds = []
    with sqlite3.connect('Data/odds.sqlite') as con:
        tables = pd.read_sql_query("select name from sqlite_master where type='table'", con).name
        for table in tables:
            frame = pd.read_sql_query(f'select * from "{table}"', con)
            if 'Date' not in frame:  # one table was saved without a header row
                continue
            parts = frame['Date'].str.split('-', expand=True)  # '2022-23-1019'
            month, day = parts[2].str[:2].astype(int), parts[2].str[2:].astype(int)
            year = parts[0].astype(int) + (month < 9)
            frame['Date'] = pd.to_datetime(dict(year=year, month=month, day=day)).dt.strftime('%Y-%m-%d')
            frame['season'] = table.replace('odds_', '')
            odds.append(frame[['Date', 'Home', 'ML_Home', 'ML_Away', 'season']])
    odds = pd.concat(odds)
    for col in ['ML_Home', 'ML_Away']:
        odds[col] = pd.to_numeric(odds[col], errors='coerce')
    merged = data.drop_duplicates(['Date', 'TEAM_NAME']).merge(
        odds, left_on=['Date', 'TEAM_NAME'], right_on=['Date', 'Home'])
    valid = (merged.ML_Home.abs() >= 100) & (merged.ML_Away.abs() >= 100)
    return merged[valid].reset_index(drop=True)


def main():
    games = load()
    features = [c for c in games.columns if c not in NON_FEATURES + ['Home', 'ML_Home', 'ML_Away', 'season']]
    home_won = games['Home-Team-Win'].astype(int)
    dec_home = games.ML_Home.map(american_to_decimal)
    dec_away = games.ML_Away.map(american_to_decimal)
    market = (1 / dec_home) / (1 / dec_home + 1 / dec_away)

    print(f"{'Sezon':8} {'Maç':>5} {'Model isabet':>13} {'Piyasa isabet':>14} | "
          f"{'ORTA+ bahis':>11} {'Getiri':>7} | {'GÜÇLÜ bahis':>11} {'Getiri':>7}")
    totals = {GOOD_EV: [], STRONG_EV: []}
    for season in sorted(s for s in games.season.unique() if s >= FIRST_TEST_SEASON):
        train, test = games.season < season, games.season == season
        model = xgb.train({'max_depth': 2, 'eta': 0.01, 'objective': 'binary:logistic', 'seed': 1},
                          xgb.DMatrix(games.loc[train, features].values.astype(float), label=home_won[train]), 500)
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
        print(f"{season:8} {test.sum():>5} {((model_prob > .5) == won).mean() * 100:>12.1f}% "
              f"{((market[test] > .5) == won).mean() * 100:>13.1f}% | {cells[0]} | {cells[1]}")
    cells = [f"{len(np.concatenate(p)):>11} {np.concatenate(p).mean() * 100:>+6.1f}%" for p in totals.values()]
    print(f"{'TOPLAM':8} {'':>5} {'':>13} {'':>14} | {cells[0]} | {cells[1]}")
    print("\nGetiri = yatırılan her 100 birim için kâr/zarar (sabit tutarla oynanırsa).")


if __name__ == '__main__':
    main()
