import re
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[2] / "Models"
ACCURACY_PATTERN = re.compile(r"Trained-Model-(?:ML|OU)-(\d+(?:\.\d+)?)")

_models = {}


def _select_best_model(prefix):
    candidates = [path for base in (MODEL_DIR, MODEL_DIR / "NN_Models") if base.exists()
                  for path in base.glob(f"{prefix}*") if path.is_dir() or path.suffix in {".keras", ".h5"}]
    if not candidates:
        raise FileNotFoundError(f"No model found for prefix {prefix} in {MODEL_DIR}")

    def score(path):
        match = ACCURACY_PATTERN.search(path.name)
        accuracy = float(match.group(1)) if match else 0.0
        return (accuracy if 0.0 <= accuracy <= 100.0 else 0.0, path.stat().st_mtime)

    return max(candidates, key=score)


def _load(prefix):
    if prefix not in _models:
        try:
            from keras.models import load_model
        except ImportError as e:
            raise RuntimeError(
                "Sinir ağı modeli için TensorFlow gerekli (pip install tensorflow). "
                "TensorFlow kurmadan -xgb seçeneğini kullanabilirsiniz.") from e
        _models[prefix] = load_model(str(_select_best_model(prefix)), compile=False)
    return _models[prefix]


def _normalize(data):
    # same as tf.keras.utils.normalize(data, axis=1), which the models were trained with
    data = np.asarray(data, dtype=float)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    return data / np.where(norms == 0, 1, norms)


def predict(features, totals, rest_days, extra=None):
    """Same contract as XGBoost_Runner.predict; `extra` (Elo/form columns) is accepted and ignored."""
    features = np.asarray(features, dtype=float)
    rest_days = np.asarray(rest_days, dtype=float)
    ml_preds = _load("Trained-Model-ML-").predict(_normalize(np.column_stack([features, rest_days])), verbose=0)
    home_probs = [float(p[1]) for p in ml_preds]

    over_probs = [None] * len(features)
    rows = [i for i, total in enumerate(totals) if total is not None]
    if rows:
        # NN training column order: team stats, rest days, OU line (differs from XGBoost)
        uo_features = np.column_stack([features[rows], rest_days[rows], [float(totals[i]) for i in rows]])
        preds = _load("Trained-Model-OU-").predict(_normalize(uo_features), verbose=0)
        for i, p in zip(rows, preds):
            over_probs[i] = float(p[1] / (p[0] + p[1]))
    return home_probs, over_probs
