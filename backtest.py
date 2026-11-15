"""Honest (walk-forward) test of the money line and over/under strategies on past seasons.

For each season the model is trained only on earlier seasons, then every bet the
recommender would have rated ORTA or GÜÇLÜ is "placed" at the historical closing
odds with a flat 1-unit stake. Besides hit rate and profit the report shows log-loss and
Brier score against the bookmaker's no-vig probability, and a season-block bootstrap
95% confidence interval for the TOTAL row.

    python3 backtest.py
    python3 backtest.py --seasons 3 --json /tmp/sonuc.json
    python3 backtest.py --first-season 2015

The model recipe (xgboost params + rounds) comes from the newest
Models/XGBoost_Models/*_{ML|UO}_*_meta.json if one exists, else from PARAMS / ROUNDS below.
"""
import argparse
import glob
import json
import os
import re
import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb

from src.Predict.Recommender import DEFAULT_TOTAL_PRICE, GOOD_EV, MODEL_WEIGHT, STRONG_EV
from src.Utils.Expected_Value import american_to_decimal
from src.Utils.tools import normalize_team_name

DATASET = 'dataset_2012-26'
# Target / identity columns. Everything else that is numeric is a feature
# (OU is a feature only for the over/under model).
NON_FEATURES = ['index', 'Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'index.1', 'TEAM_NAME.1', 'Date.1',
                'OU-Cover', 'OU']
EXTRA_COLUMNS = ['Home', 'ML_Home', 'ML_Away', 'season']  # added by load(), never features
FIRST_TEST_SEASON = 2017
PARAMS = {'max_depth': 3, 'eta': 0.02, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5,
          'objective': 'binary:logistic', 'seed': 1}
ROUNDS = 300
MODELS_DIR = 'Models/XGBoost_Models'
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 12345
BREAK_EVEN_110 = 110 / 210  # %52,4: hit rate needed at -110


# ---------------------------------------------------------------- pure helpers

def binary_params(params):
    """Turn any recipe (e.g. multi:softprob) into this script's binary setup."""
    out = dict(params)
    out['objective'] = 'binary:logistic'
    out.pop('num_class', None)
    out.pop('eval_metric', None)
    return out


def load_recipe(kind, models_dir=MODELS_DIR):
    """(params, num_boost_round, source) for kind 'ML' or 'UO'.

    Newest `*_{kind}_*_meta.json` holding {"params": {...}, "num_boost_round": int} wins;
    otherwise the module defaults PARAMS / ROUNDS are used.
    """
    if kind not in ('ML', 'UO'):
        raise ValueError(f"kind 'ML' ya da 'UO' olmalı: {kind!r}")
    files = glob.glob(os.path.join(models_dir, f'*_{kind}_*_meta.json'))
    if not files:
        return binary_params(PARAMS), ROUNDS, 'varsayılan'
    newest = max(files, key=lambda f: (os.path.getmtime(f), f))
    with open(newest) as handle:
        meta = json.load(handle)
    return binary_params(meta['params']), int(meta['num_boost_round']), os.path.basename(newest)


MIN_BETS_FOR_CI = 30


def select_features(columns_dtypes, include_ou=False, drop_pattern=None):
    """All numeric columns that are not target/identity columns. `columns_dtypes` is df.dtypes.
    Columns matching the regex `drop_pattern` (re.search) are left out (for A/B comparisons)."""
    excluded = set(NON_FEATURES) | set(EXTRA_COLUMNS)
    if include_ou:
        excluded.discard('OU')
    features = [c for c, t in columns_dtypes.items() if c not in excluded and pd.api.types.is_numeric_dtype(t)]
    if drop_pattern:
        features = [c for c in features if not re.search(drop_pattern, c)]
    return features


def log_loss(prob, outcome, eps=1e-15):
    prob = np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)
    outcome = np.asarray(outcome, dtype=float)
    return float(-np.mean(outcome * np.log(prob) + (1 - outcome) * np.log(1 - prob)))


def brier(prob, outcome):
    return float(np.mean((np.asarray(prob, dtype=float) - np.asarray(outcome, dtype=float)) ** 2))


def moneyline_profits(fair_home, home_won, dec_home, dec_away, threshold):
    """Flat 1-unit profits of every side whose EV (fair * decimal - 1) >= threshold (Recommender formula)."""
    fair_home, home_won = np.asarray(fair_home, float), np.asarray(home_won)
    dec_home, dec_away = np.asarray(dec_home, float), np.asarray(dec_away, float)
    bet_home = fair_home * dec_home - 1 >= threshold
    bet_away = (1 - fair_home) * dec_away - 1 >= threshold
    return np.concatenate([np.where(home_won[bet_home] == 1, dec_home[bet_home] - 1, -1.0),
                           np.where(home_won[bet_away] == 0, dec_away[bet_away] - 1, -1.0)])


def total_profits(over_prob, went_over, threshold=GOOD_EV, price=DEFAULT_TOTAL_PRICE):
    """Over/under bets Recommender-style: pick the model's side, blend its confidence with a 0.5 market."""
    over_prob, went_over = np.asarray(over_prob, float), np.asarray(went_over)
    pick_over = over_prob >= 0.5
    confidence = np.maximum(over_prob, 1 - over_prob)
    fair = MODEL_WEIGHT * confidence + (1 - MODEL_WEIGHT) * 0.5
    decimal = american_to_decimal(price)
    bet = fair * decimal - 1 >= threshold
    won = pick_over[bet] == (went_over[bet] == 1)
    return np.where(won, decimal - 1, -1.0)


def bootstrap_ratio_ci(numerators, denominators, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED, level=0.95):
    """Season-block bootstrap CI of sum(num)/sum(den) (e.g. profit per bet, log-loss diff per game).

    Seasons are resampled with replacement. Resamples whose denominator is 0 are discarded.
    Returns {'low', 'high', 'point', 'contains_zero', 'valid_reps'}; NaN/None when nothing is valid.
    """
    num, den = np.asarray(numerators, float), np.asarray(denominators, float)
    if len(num) == 0 or den.sum() == 0:
        return {'low': None, 'high': None, 'point': None, 'contains_zero': None, 'valid_reps': 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(reps, len(num)))
    n, d = num[idx].sum(axis=1), den[idx].sum(axis=1)
    ok = d > 0
    stats = n[ok] / d[ok]
    alpha = (1 - level) / 2
    low, high = np.quantile(stats, [alpha, 1 - alpha])
    return {'low': float(low), 'high': float(high), 'point': float(num.sum() / den.sum()),
            'contains_zero': bool(low <= 0 <= high), 'valid_reps': int(ok.sum())}


def ci_or_none(per_season_profits, n_bets):
    """Bootstrap CI of profit per bet, or None when there are too few bets for it to mean anything."""
    if n_bets < MIN_BETS_FOR_CI:
        return None
    return bootstrap_ratio_ci([p.sum() for p in per_season_profits], [len(p) for p in per_season_profits])


def mean_or_none(values):
    values = np.asarray(values, float)
    return float(values.mean()) if len(values) else None


# ---------------------------------------------------------------- data

def load():
    """(games, stats). stats counts rows lost in the odds merge / odds validity filter."""
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
    unique_data = data.drop_duplicates(['Date', 'Home'])
    merged = unique_data.merge(odds.drop_duplicates(['Date', 'Home']), on=['Date', 'Home'])
    dates = pd.to_datetime(merged['Date'])
    merged['season'] = np.where(dates.dt.month >= 9, dates.dt.year, dates.dt.year - 1)
    valid = (merged.ML_Home.abs() >= 100) & (merged.ML_Away.abs() >= 100)
    stats = {'veri_seti_maci': int(len(unique_data)), 'oranla_eslesen': int(len(merged)),
             'oransiz_dusen': int(len(unique_data) - len(merged)),
             'gecersiz_oran_dusen': int((~valid).sum())}
    return merged[valid].sort_values('Date').reset_index(drop=True), stats


def train_predict(frame, features, label, train_mask, test_mask, recipe):
    params, rounds, _ = recipe
    params = {'nthread': 2, **params}
    model = xgb.train(params, xgb.DMatrix(frame.loc[train_mask, features].values.astype(float),
                                          label=label[train_mask]), rounds)
    return model.predict(xgb.DMatrix(frame.loc[test_mask, features].values.astype(float)))


# ---------------------------------------------------------------- runs

def run_moneyline(games, seasons, recipe, drop_pattern=None):
    features = select_features(games.dtypes, drop_pattern=drop_pattern)
    home_won = games['Home-Team-Win'].astype(int).values
    dec_home = games.ML_Home.map(american_to_decimal).values
    dec_away = games.ML_Away.map(american_to_decimal).values
    market = (1 / dec_home) / (1 / dec_home + 1 / dec_away)
    rows, profits = [], {GOOD_EV: [], STRONG_EV: []}
    ll_diff_sum, count = [], []
    for season in seasons:
        train, test = (games.season < season).values, (games.season == season).values
        if train.sum() == 0 or test.sum() == 0:
            continue
        prob = train_predict(games, features, home_won, train, test, recipe)
        won, mk = home_won[test], market[test]
        fair = MODEL_WEIGHT * prob + (1 - MODEL_WEIGHT) * mk
        row = {'sezon': f"{season}-{str(season + 1)[-2:]}", 'mac': int(test.sum()),
               'model_isabet': float(((prob > .5) == won).mean()), 'piyasa_isabet': float(((mk > .5) == won).mean()),
               'model_logloss': log_loss(prob, won), 'piyasa_logloss': log_loss(mk, won),
               'model_brier': brier(prob, won), 'piyasa_brier': brier(mk, won)}
        for name, threshold in (('orta', GOOD_EV), ('guclu', STRONG_EV)):
            p = moneyline_profits(fair, won, dec_home[test], dec_away[test], threshold)
            profits[threshold].append(p)
            row[f'{name}_bahis'] = int(len(p))
            row[f'{name}_getiri'] = mean_or_none(p)
        rows.append(row)
        ll_diff_sum.append((row['model_logloss'] - row['piyasa_logloss']) * row['mac'])
        count.append(row['mac'])
    total = _totals(rows, profits, ll_diff_sum, count)
    return {'satirlar': rows, 'toplam': total, 'ozellik_sayisi': len(features)}


def _totals(rows, profits, ll_diff_sum, count):
    total = {'mac': int(sum(count))}
    if not rows:
        return total
    w = np.array(count, float)
    for key in ('model_isabet', 'piyasa_isabet', 'model_logloss', 'piyasa_logloss', 'model_brier', 'piyasa_brier'):
        total[key] = float(np.average([r[key] for r in rows], weights=w))
    ci = {}
    for name, threshold in (('orta', GOOD_EV), ('guclu', STRONG_EV)):
        per_season = profits[threshold]
        allp = np.concatenate(per_season)
        total[f'{name}_bahis'] = int(len(allp))
        total[f'{name}_getiri'] = mean_or_none(allp)
        ci[f'{name}_getiri'] = ci_or_none(per_season, len(allp))
    ci['logloss_fark'] = bootstrap_ratio_ci(ll_diff_sum, count)
    total['guven_araligi'] = ci
    return total


def run_totals(games, seasons, recipe, drop_pattern=None):
    ou_games = games[games['OU-Cover'].isin([0, 1])].reset_index(drop=True)
    features = select_features(ou_games.dtypes, include_ou=True, drop_pattern=drop_pattern)
    went_over = ou_games['OU-Cover'].astype(int).values
    rows, profits, ll_sum, count, hits = [], [], [], [], []
    for season in seasons:
        train, test = (ou_games.season < season).values, (ou_games.season == season).values
        if train.sum() == 0 or test.sum() == 0:
            continue
        prob = train_predict(ou_games, features, went_over, train, test, recipe)
        y = went_over[test]
        p = total_profits(prob, y)
        profits.append(p)
        row = {'sezon': f"{season}-{str(season + 1)[-2:]}", 'mac': int(test.sum()),
               'isabet': float(((prob >= .5) == y).mean()), 'logloss': log_loss(prob, y),
               'bahis': int(len(p)), 'getiri': mean_or_none(p)}
        rows.append(row)
        ll_sum.append(row['logloss'] * row['mac'])
        count.append(row['mac'])
        hits.append(row['isabet'] * row['mac'])
    total = {'mac': int(sum(count))}
    if rows:
        allp = np.concatenate(profits)
        total.update({'isabet': float(sum(hits) / sum(count)), 'logloss': float(sum(ll_sum) / sum(count)),
                      'bahis': int(len(allp)), 'getiri': mean_or_none(allp),
                      'guven_araligi': {'getiri': ci_or_none(profits, len(allp))}})
    return {'satirlar': rows, 'toplam': total, 'ozellik_sayisi': len(features)}


# ---------------------------------------------------------------- printing

def pct(value, signed=False, digits=1):
    if value is None:
        return '-'
    return f"{value * 100:{'+' if signed else ''}.{digits}f}%"


def num(value, digits=4):
    return '-' if value is None else f"{value:.{digits}f}"


def ci_line(label, ci, signed=True, unit='%'):
    if ci is None:
        return f"  {label}: yetersiz bahis (<{MIN_BETS_FOR_CI}); güven aralığı hesaplanmadı"
    if ci['low'] is None:
        return f"  {label}: hesaplanamadı (bahis yok)"
    fmt = (lambda v: pct(v, signed)) if unit == '%' else (lambda v: f"{v:+.4f}")
    verdict = "0'ı İÇERİYOR (şanstan ayırt edilemez)" if ci['contains_zero'] else "0'ı İÇERMİYOR"
    return f"  {label}: %95 GA [{fmt(ci['low'])}, {fmt(ci['high'])}] (nokta {fmt(ci['point'])}) -> {verdict}"


def print_moneyline(result):
    print("\n=== MAÇ SONUCU (moneyline) ===")
    print(f"{'Sezon':8} {'Maç':>5} | {'İsabet M':>8} {'Piy.':>6} | {'LogLoss M':>9} {'Piy.':>7} | "
          f"{'Brier M':>7} {'Piy.':>7} | {'ORTA+ bahis':>11} {'Getiri':>7} | {'GÜÇLÜ bahis':>11} {'Getiri':>7}")
    for r in result['satirlar'] + [dict(result['toplam'], sezon='TOPLAM')]:
        if 'model_isabet' not in r:
            continue
        print(f"{r['sezon']:8} {r['mac']:>5} | {pct(r['model_isabet']):>8} {pct(r['piyasa_isabet']):>6} | "
              f"{num(r['model_logloss']):>9} {num(r['piyasa_logloss']):>7} | "
              f"{num(r['model_brier']):>7} {num(r['piyasa_brier']):>7} | "
              f"{r['orta_bahis']:>11} {pct(r['orta_getiri'], True):>7} | "
              f"{r['guclu_bahis']:>11} {pct(r['guclu_getiri'], True):>7}")
    ci = result['toplam'].get('guven_araligi')
    if ci:
        print("\nTOPLAM için sezon bloklu bootstrap (2000 tekrar, sabit tohum):")
        print(ci_line('ORTA+ getirisi', ci['orta_getiri']))
        print(ci_line('GÜÇLÜ getirisi', ci['guclu_getiri']))
        print(ci_line('Log-loss farkı (model - piyasa; negatif = model iyi)', ci['logloss_fark'], unit='x'))


def print_totals(result):
    print("\n=== ALT/ÜST ===")
    print(f"{'Sezon':8} {'Maç':>5} | {'İsabet':>7} {'LogLoss':>8} | {'Önerilen bahis':>14} {'Getiri':>7}")
    for r in result['satirlar'] + [dict(result['toplam'], sezon='TOPLAM')]:
        if 'isabet' not in r:
            continue
        print(f"{r['sezon']:8} {r['mac']:>5} | {pct(r['isabet']):>7} {num(r['logloss']):>8} | "
              f"{r['bahis']:>14} {pct(r['getiri'], True):>7}")
    ci = result['toplam'].get('guven_araligi')
    if ci:
        print("\nTOPLAM için sezon bloklu bootstrap:")
        print(ci_line('Alt/üst getirisi', ci['getiri']))
        if result['toplam']['bahis'] < MIN_BETS_FOR_CI:
            print(f"  UYARI: yalnızca {result['toplam']['bahis']} bahis var; sonuç çıkarılamaz.")
    print(f"Not: -110 fiyatta başabaş isabeti {pct(BREAK_EVEN_110)}; yazı-tura {pct(0.5)}. Hiç bahis önerilmeyen "
          "sezonda getiri '-' gösterilir (0 değildir).")


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# ---------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--json', metavar='YOL', help='tüm sayıları JSON olarak bu dosyaya yaz')
    parser.add_argument('--first-season', type=int, default=FIRST_TEST_SEASON, metavar='YIL',
                        help=f'ilk test sezonu (varsayılan {FIRST_TEST_SEASON})')
    parser.add_argument('--drop-features', metavar='REGEX',
                        help="adı bu düzenli ifadeye uyan özellik sütunlarını modele verme (karşılaştırma için), "
                             "örn. '^(Elo|Form|HomeSplit|AwaySplit|Games7|B2B)'")
    parser.add_argument('--seasons', type=int, metavar='N', help='yalnız son N sezonu test et')
    args = parser.parse_args(argv)

    games, stats = load()
    seasons = sorted(s for s in games.season.unique() if s >= args.first_season)
    if args.seasons:
        seasons = seasons[-args.seasons:]
    if not seasons:
        raise SystemExit('Test edilecek sezon yok (--first-season çok büyük olabilir).')
    ml_recipe, ou_recipe = load_recipe('ML'), load_recipe('UO')

    print(f"Veri: {stats['veri_seti_maci']} maç; oranla eşleşen {stats['oranla_eslesen']}, "
          f"oran bulunamayıp DÜŞEN {stats['oransiz_dusen']}, geçersiz oran yüzünden DÜŞEN "
          f"{stats['gecersiz_oran_dusen']}; kullanılan {len(games)}.")
    print(f"Test sezonları: {seasons[0]}-{seasons[-1]} ({len(seasons)} sezon). Tarif ML: {ml_recipe[2]} "
          f"({ml_recipe[1]} tur), UO: {ou_recipe[2]} ({ou_recipe[1]} tur).")

    if args.drop_features:
        print(f"Atılan özellik deseni: {args.drop_features}")
    moneyline = run_moneyline(games, seasons, ml_recipe, args.drop_features)
    print(f"Maç sonucu modelinde kullanılan özellik sayısı: {moneyline['ozellik_sayisi']}")
    print_moneyline(moneyline)
    totals = run_totals(games, seasons, ou_recipe, args.drop_features)
    print(f"\nAlt/üst modelinde kullanılan özellik sayısı (OU çizgisi dahil): {totals['ozellik_sayisi']}")
    print_totals(totals)
    print("\nGetiri = yatırılan her 100 birim için kâr/zarar (sabit tutarla oynanırsa). "
          f"Harman: %{MODEL_WEIGHT * 100:.0f} model / %{(1 - MODEL_WEIGHT) * 100:.0f} piyasa; "
          f"eşikler EV >= {GOOD_EV} (ORTA), >= {STRONG_EV} (GÜÇLÜ).")

    if args.json:
        payload = {'veri': stats, 'sezonlar': [int(s) for s in seasons],
                   'tarif': {'ML': {'params': ml_recipe[0], 'num_boost_round': ml_recipe[1], 'kaynak': ml_recipe[2]},
                             'UO': {'params': ou_recipe[0], 'num_boost_round': ou_recipe[1], 'kaynak': ou_recipe[2]}},
                   'atilan_ozellik_deseni': args.drop_features,
                   'bootstrap': {'tekrar': BOOTSTRAP_REPS, 'tohum': BOOTSTRAP_SEED},
                   'mac_sonucu': moneyline, 'alt_ust': totals}
        with open(args.json, 'w') as handle:
            json.dump(_clean(payload), handle, ensure_ascii=False, indent=2)
        print(f"JSON yazıldı: {args.json}")


if __name__ == '__main__':
    main()
