"""Sigmoid (Platt) calibration of model probabilities, stored as plain JSON.

JSON keeps the calibration usable across scikit-learn/xgboost versions, unlike a pickle.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_sigmoid(probabilities, labels):
    """One-vs-rest sigmoid per class, fitted on held-out predictions."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)
    params = []
    for cls in range(probabilities.shape[1]):
        target = (labels == cls).astype(int)
        if target.min() == target.max():  # class absent from the calibration set
            params.append(None)
            continue
        fit = LogisticRegression(C=1e6).fit(probabilities[:, [cls]], target)
        params.append({"slope": float(fit.coef_[0][0]), "intercept": float(fit.intercept_[0])})
    return {"method": "sigmoid", "classes": params}


def apply(probabilities, calibration):
    probabilities = np.asarray(probabilities, dtype=float)
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
