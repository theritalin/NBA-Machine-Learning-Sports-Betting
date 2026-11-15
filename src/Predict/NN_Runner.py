import os

import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Models', 'NN_Models')

_models = {}


def _load(name):
    if name not in _models:
        try:
            from tensorflow.keras.models import load_model
        except ImportError as e:
            raise RuntimeError(
                "Sinir ağı modeli için TensorFlow gerekli (pip install 'tensorflow<2.16'). "
                "TensorFlow kurmadan -xgb seçeneğini kullanabilirsiniz.") from e
        _models[name] = load_model(os.path.join(MODEL_DIR, name))
    return _models[name]


def _normalize(data):
    import tensorflow as tf
    return tf.keras.utils.normalize(np.asarray(data, dtype=float), axis=1)


def predict(features, totals):
    """Same contract as XGBoost_Runner.predict."""
    features = np.asarray(features, dtype=float)
    ml_preds = _load('Trained-Model-ML').predict(_normalize(features), verbose=0)
    home_probs = [float(p[1]) for p in ml_preds]

    over_probs = [None] * len(features)
    rows = [i for i, total in enumerate(totals) if total is not None]
    if rows:
        uo_features = np.column_stack([features[rows], [float(totals[i]) for i in rows]])
        preds = _load('Trained-Model-OU').predict(_normalize(uo_features), verbose=0)
        for i, p in zip(rows, preds):
            over_probs[i] = float(p[1] / (p[0] + p[1]))
    return home_probs, over_probs
