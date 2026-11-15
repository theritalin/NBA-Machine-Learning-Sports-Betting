import os

import numpy as np
import xgboost as xgb

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Models', 'XGBoost_Models')
ML_MODEL = 'XGBoost_67.8%_ML-2.json'
UO_MODEL = 'XGBoost_52.5%_UO-6.json'

_models = {}


def _load(name):
    if name not in _models:
        booster = xgb.Booster()
        booster.load_model(os.path.join(MODEL_DIR, name))
        _models[name] = booster
    return _models[name]


def predict(features, totals):
    """features: (n_games, 104) float array. totals: O/U line per game, None if unknown.

    Returns (home_win_probs, over_probs); over_probs[i] is None when there is no line.
    """
    features = np.asarray(features, dtype=float)
    home_probs = [float(p) for p in _load(ML_MODEL).predict(xgb.DMatrix(features))[:, 1]]

    over_probs = [None] * len(features)
    rows = [i for i, total in enumerate(totals) if total is not None]
    if rows:
        uo_features = np.column_stack([features[rows], [float(totals[i]) for i in rows]])
        # classes: 0 = under, 1 = over, 2 = push
        preds = _load(UO_MODEL).predict(xgb.DMatrix(uo_features))
        for i, p in zip(rows, preds):
            over_probs[i] = float(p[1] / (p[0] + p[1]))
    return home_probs, over_probs
