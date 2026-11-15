"""Sigmoid (Platt) calibration of model probabilities, stored as plain JSON.

JSON keeps the calibration usable across scikit-learn/xgboost versions, unlike a pickle.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_sigmoid(probabilities, labels):
    """One-vs-rest sigmoid per class, fitted on held-out predictions.

    A 1-D input is the probability of the positive class of a binary model: a single Platt
    sigmoid on logit(p) is fitted (identity = slope 1, intercept 0).
    """
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)
    if probabilities.ndim == 1:
        if labels.min() == labels.max():
            return {"method": "sigmoid", "binary": True, "classes": [None]}
        fit = LogisticRegression(C=1e6).fit(_logit(probabilities).reshape(-1, 1), labels.astype(int))
        return {"method": "sigmoid", "binary": True,
                "classes": [{"slope": float(fit.coef_[0][0]), "intercept": float(fit.intercept_[0])}]}
    params = []
    for cls in range(probabilities.shape[1]):
        target = (labels == cls).astype(int)
        if target.min() == target.max():  # class absent from the calibration set
            params.append(None)
            continue
        fit = LogisticRegression(C=1e6).fit(probabilities[:, [cls]], target)
        params.append({"slope": float(fit.coef_[0][0]), "intercept": float(fit.intercept_[0])})
    return {"method": "sigmoid", "classes": params}


def _apply_binary(p, calibration):
    params = calibration["classes"][0]
    if params is None:
        return p
    return 1 / (1 + np.exp(-(params["slope"] * _logit(p) + params["intercept"])))


def apply(probabilities, calibration):
    """1-D in -> 1-D out for binary calibrations (a [1-p, p] 2-column input is also accepted)."""
    probabilities = np.asarray(probabilities, dtype=float)
    if calibration.get("binary"):
        if probabilities.ndim == 1:
            return _apply_binary(probabilities, calibration)
        p = _apply_binary(probabilities[:, 1], calibration)
        return np.column_stack([1 - p, p])
    if probabilities.ndim == 1:
        raise ValueError("1-D probabilities need a binary calibration.")
    calibrated = probabilities.copy()
    for cls, params in enumerate(calibration["classes"]):
        if params is not None:
            calibrated[:, cls] = 1 / (1 + np.exp(-(params["slope"] * probabilities[:, cls] + params["intercept"])))
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def save(calibration, path):
    with open(path, "w") as handle:
        json.dump(calibration, handle, indent=2)


def load(path):
    with open(path) as handle:
        return json.load(handle)
