import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin

from src.Utils import Calibration

MODEL_DIR = Path(__file__).resolve().parents[2] / "Models" / "XGBoost_Models"
ACCURACY_PATTERN = re.compile(r"XGBoost_(\d+(?:\.\d+)?)%_")

AUXILIARY_SUFFIXES = ("_calibration", "_meta")

_models = {}


def _warn(message):
    print(f"Uyarı: {message}", file=sys.stderr)


class BoosterWrapper(ClassifierMixin, BaseEstimator):
    """Stand-in for the class older training scripts wrapped the booster in before pickling."""

    def __init__(self, booster, num_class):
        self.booster = booster
        self.classes_ = np.arange(num_class)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return self.booster.predict(xgb.DMatrix(X))


def _select_model_path(kind):
    candidates = [path for path in MODEL_DIR.glob(f"*_{kind}_*.json")
                  if not path.stem.endswith(AUXILIARY_SUFFIXES)]
    if not candidates:
        raise FileNotFoundError(f"No XGBoost {kind} model found in {MODEL_DIR}")

    def score(path):
        match = ACCURACY_PATTERN.search(path.name)
        return (path.stat().st_mtime, float(match.group(1)) if match else 0.0)

    return max(candidates, key=score)


def _load_calibrator(model_path):
    """Returns a function calibrating raw booster probabilities, or None."""
    json_path = model_path.with_name(f"{model_path.stem}_calibration.json")
    if json_path.exists():
        try:
            calibration = Calibration.load(json_path)
            n_classes = len(calibration["classes"])
            probe = np.array([0.5]) if calibration.get("binary") else np.full((1, n_classes), 1 / n_classes)
            Calibration.apply(probe, calibration)
        except Exception as e:  # unreadable or malformed JSON
            _warn(f"kalibrasyon dosyası kullanılamadı ({json_path.name}: {type(e).__name__}), "
                  "ham model olasılıkları kullanılacak")
            return None
        return lambda probs, data: Calibration.apply(probs, calibration)

    legacy_path = model_path.with_name(f"{model_path.stem}_calibration.pkl")
    if not legacy_path.exists():
        _warn(f"{model_path.name} için kalibrasyon dosyası yok, ham model olasılıkları kullanılacak")
        return None
    try:
        # Legacy scikit-learn calibrators were pickled by training scripts run as __main__,
        # so the wrapper class is recorded as __main__.BoosterWrapper and must exist there.
        main_module = sys.modules["__main__"]
        if not hasattr(main_module, "BoosterWrapper"):
            main_module.BoosterWrapper = BoosterWrapper
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calibrator = joblib.load(legacy_path)
        calibrator.predict_proba(np.zeros((1, calibrator.estimator.booster.num_features())))
    except Exception as e:  # pickle from an incompatible scikit-learn/xgboost
        print(f"Uyarı: kalibrasyon dosyası kullanılamadı, ham model olasılıkları kullanılacak ({type(e).__name__})",
              file=sys.stderr)
        return None
    return lambda probs, data: calibrator.predict_proba(data)


def _load(kind):
    """Returns a function mapping a feature matrix to class probabilities."""
    if kind not in _models:
        path = _select_model_path(kind)
        booster = xgb.Booster()
        booster.load_model(str(path))
        calibrate = _load_calibrator(path)

        def predict_probs(data):
            probs = booster.predict(xgb.DMatrix(data))
            if calibrate:
                probs = calibrate(probs, data)
            if probs.ndim == 1:  # binary:logistic -> [1-p, p], same layout as old softprob models
                probs = np.column_stack([1 - probs, probs])
            return probs

        predict_probs.num_features = booster.num_features()
        _models[kind] = predict_probs
    return _models[kind]


def _with_extra(kind, base, extra):
    """Appends the Elo/form columns when (and only when) the model was trained with them.

    base: the old-layout matrix (ML: stats + rest; UO: stats + OU + rest). The model decides:
    it expects either base.shape[1] columns (extra is ignored) or base + extra columns.
    Anything else is an error: silently feeding wrong columns would give wrong probabilities.
    """
    expected = _load(kind).num_features
    old_width = base.shape[1]
    if expected == old_width:
        return base
    extra_width = 0 if extra is None else extra.shape[1]
    if extra is not None and extra_width and expected == old_width + extra_width:
        return np.column_stack([base, extra])
    raise RuntimeError(
        f"{kind} modeli {expected} özellik bekliyor; verilen {old_width} sütun"
        + (f" + {extra_width} Elo/form sütunu" if extra is not None else " (Elo/form sütunları verilmedi)")
        + " bu sayıya uymuyor. Model ile veri seti sürümü uyuşmuyor; modeli ya da özellikleri güncelleyin.")


def predict(features, totals, rest_days, extra=None):
    """features: (n_games, 104) team stats, home then away. totals: O/U line per game or None.
    rest_days: (n_games, 2) days of rest for home and away.
    extra: optional (n_games, len(Game_Features.FEATURE_COLUMNS)) Elo/form columns; used only by models
    trained with them (ignored by older models).

    Returns (home_win_probs, over_probs); over_probs[i] is None when there is no line.
    """
    features = np.asarray(features, dtype=float)
    rest_days = np.asarray(rest_days, dtype=float)
    extra = None if extra is None else np.asarray(extra, dtype=float).reshape(len(features), -1)
    ml_probs = _load("ML")(_with_extra("ML", np.column_stack([features, rest_days]), extra))
    home_probs = [float(p[1]) for p in ml_probs]

    over_probs = [None] * len(features)
    rows = [i for i, total in enumerate(totals) if total is not None]
    if rows:
        # training column order: team stats, OU line, rest days, (Elo/form). Classes: 0 under, 1 over (old models: 2 push).
        uo_base = np.column_stack([features[rows], [float(totals[i]) for i in rows], rest_days[rows]])
        uo_features = _with_extra("UO", uo_base, None if extra is None else extra[rows])
        for i, p in zip(rows, _load("UO")(uo_features)):
            over_probs[i] = float(p[1] / (p[0] + p[1]))
    return home_probs, over_probs
