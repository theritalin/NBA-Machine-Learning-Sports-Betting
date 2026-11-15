"""Shared training pipeline for the XGBoost moneyline (ML) and totals (UO) models.

Both targets are binary (`binary:logistic`): ML = home team won, UO = went over (pushes dropped).
Goal is well calibrated probabilities (log-loss), not accuracy.

Flow: random search with time-ordered CV (early stopping on the tail of each training fold,
scoring only on the untouched validation fold) -> held-out evaluation on the last 10% with
sigmoid calibration learned from the CV out-of-fold predictions -> optional refit on all rows.
"""
import argparse
import datetime
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.Utils import Calibration  # noqa: E402

DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"
MODEL_DIR = BASE_DIR / "Models" / "XGBoost_Models"

DEFAULT_DATASET = "dataset_2012-26"
DATE_COLUMN = "Date"
COMMON_DROP_COLUMNS = [
    "index", "Score", "Home-Team-Win", "TEAM_NAME", "Date",
    "index.1", "TEAM_NAME.1", "Date.1", "OU-Cover",
]
# ML must not see the line; UO uses it as a feature (column order: team stats, OU, rest days).
DROP_COLUMNS = {"ML": COMMON_DROP_COLUMNS + ["OU"], "UO": COMMON_DROP_COLUMNS}
MAX_BOOST_ROUNDS = 1500
EARLY_STOPPING_ROUNDS = 50
ES_FRACTION = 0.1
TEST_SIZE = 0.1


def load_dataset(dataset_name):
    with sqlite3.connect(DATASET_DB) as con:
        return pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)


def prepare_data(df, kind):
    """Returns X, y, dates sorted by date. UO: push rows (OU-Cover == 2) are dropped, over = 1."""
    data = df.copy()
    if kind == "UO":
        data = data[data["OU-Cover"] != 2]
    if DATE_COLUMN in data.columns:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
        data = data.sort_values(DATE_COLUMN, kind="stable")
    target = "Home-Team-Win" if kind == "ML" else "OU-Cover"
    y = data[target].astype(int).to_numpy()
    dates = data[DATE_COLUMN].to_numpy() if DATE_COLUMN in data.columns else np.array([None] * len(data))
    X = data.drop(columns=DROP_COLUMNS[kind], errors="ignore").astype(float).to_numpy()
    return X, y, dates


def split_train_test(n, test_size=TEST_SIZE):
    """Index where the (chronologically last) test block starts."""
    if n == 0:
        raise ValueError("Empty dataset.")
    return int(n * (1 - test_size))


def sample_params(rng, seed):
    def log_uniform(low, high):
        return float(10 ** rng.uniform(np.log10(low), np.log10(high)))

    return {
        "max_depth": int(rng.integers(2, 7)),
        "eta": log_uniform(0.01, 0.1),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "min_child_weight": log_uniform(1, 50),
        "gamma": float(rng.uniform(0.0, 5.0)),
        "lambda": log_uniform(0.1, 10.0),
        "alpha": log_uniform(0.01, 5.0),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "seed": int(seed),
        "tree_method": "hist",
    }


def cv_folds(n, n_splits, es_fraction=ES_FRACTION):
    """Yields (fit_idx, es_idx, val_idx). es_idx is the last part of each training fold; val_idx is
    the following block and is disjoint from both."""
    for train_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(np.arange(n)):
        es_start = len(train_idx) - max(1, int(len(train_idx) * es_fraction))
        yield train_idx[:es_start], train_idx[es_start:], val_idx


def cross_validate(X, y, params, n_splits):
    """Returns mean validation log-loss, per-fold best_iteration, and out-of-fold (idx, probs)."""
    losses, best_iterations, oof_idx, oof_probs = [], [], [], []
    for fit_idx, es_idx, val_idx in cv_folds(len(X), n_splits):
        dfit = xgb.DMatrix(X[fit_idx], label=y[fit_idx])
        des = xgb.DMatrix(X[es_idx], label=y[es_idx])
        model = xgb.train(
            params, dfit, num_boost_round=MAX_BOOST_ROUNDS, evals=[(des, "es")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose_eval=False,
        )
        best_iteration = int(model.best_iteration)
        probs = model.predict(xgb.DMatrix(X[val_idx]), iteration_range=(0, best_iteration + 1))
        losses.append(log_loss(y[val_idx], probs, labels=[0, 1]))
        best_iterations.append(best_iteration)
        oof_idx.append(val_idx)
        oof_probs.append(probs)
    return {
        "cv_log_loss": float(np.mean(losses)),
        "best_iterations": best_iterations,
        "oof_idx": np.concatenate(oof_idx),
        "oof_probs": np.concatenate(oof_probs),
    }


def random_search(X, y, trials, n_splits, seed, log=print):
    rng = np.random.default_rng(seed)
    best = None
    for trial in range(1, trials + 1):
        params = sample_params(rng, seed + trial)
        result = cross_validate(X, y, params, n_splits)
        rounds = int(np.median(result["best_iterations"])) + 1
        log(f"Trial {trial}/{trials}: cv log loss {result['cv_log_loss']:.4f} rounds {rounds}")
        if best is None or result["cv_log_loss"] < best["cv_log_loss"]:
            best = dict(result, params=params, num_boost_round=rounds)
    return best


def fit_fixed(X, y, params, num_boost_round):
    return xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=num_boost_round, verbose_eval=False)


def evaluate_heldout(X_train, y_train, X_test, y_test, best, y_oof, calibration_method="sigmoid"):
    """Trains on X_train with the fixed round count and scores the held-out block. Calibration is
    learned only from the CV out-of-fold predictions (y_oof = labels of those rows); X_test/y_test
    are used for scoring only."""
    model = fit_fixed(X_train, y_train, best["params"], best["num_boost_round"])
    calibration = None
    if calibration_method != "none":
        calibration = Calibration.fit_sigmoid(best["oof_probs"], y_oof)
    raw = model.predict(xgb.DMatrix(X_test))
    probs = Calibration.apply(raw, calibration) if calibration is not None else raw
    metrics = {
        "n": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, probs >= 0.5)),
        "log_loss": float(log_loss(y_test, probs, labels=[0, 1])),
        "brier": float(brier_score_loss(y_test, probs)),
        "accuracy_raw": float(accuracy_score(y_test, raw >= 0.5)),
        "log_loss_raw": float(log_loss(y_test, raw, labels=[0, 1])),
    }
    return model, calibration, metrics


def build_meta(best, model_rows, refit, heldout, n_features, trials):
    return {
        "params": best["params"],
        "num_boost_round": int(best["num_boost_round"]),
        "feature_count": int(n_features),
        "trained_rows": int(model_rows),
        "refit_on_all": bool(refit),
        "heldout": heldout,
        "cv_log_loss": float(best["cv_log_loss"]),
        "trials": int(trials),
    }


def train_pipeline(X, y, dates, trials, n_splits, seed, calibration_method="sigmoid", refit=True, log=print):
    """Runs the whole flow and returns (model, calibration, meta) without touching the disk."""
    test_start = split_train_test(len(X))
    X_tv, y_tv = X[:test_start], y[:test_start]
    X_test, y_test = X[test_start:], y[test_start:]

    best = random_search(X_tv, y_tv, trials, n_splits, seed, log)
    model, calibration, metrics = evaluate_heldout(
        X_tv, y_tv, X_test, y_test, best, y_tv[best["oof_idx"]], calibration_method
    )
    heldout = {"n": metrics["n"], "start_date": str(pd.Timestamp(dates[test_start]).date())}
    heldout.update({key: metrics[key] for key in
                    ["accuracy", "log_loss", "brier", "accuracy_raw", "log_loss_raw"]})
    log(f"Best cv log loss: {best['cv_log_loss']:.4f}  rounds: {best['num_boost_round']}  "
        f"fold best_iterations: {best['best_iterations']}")
    log("Held-out (n={n}, from {start_date}): acc {accuracy:.4f} logloss {log_loss:.4f} brier {brier:.4f} "
        "| raw acc {accuracy_raw:.4f} raw logloss {log_loss_raw:.4f}".format(**heldout))

    if refit:  # the held-out numbers above come from the 90% model, not from this one
        model = fit_fixed(X, y, best["params"], best["num_boost_round"])
    meta = build_meta(best, len(X) if refit else len(X_tv), refit, heldout, X.shape[1], trials)
    return model, calibration, meta


def save_artifacts(model, calibration, meta, kind, model_dir=None, today=None):
    model_dir = Path(model_dir or MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    today = today or datetime.date.today()
    path = model_dir / f"XGBoost_{meta['heldout']['accuracy'] * 100:.1f}%_{kind}_{today:%Y%m%d}.json"
    model.save_model(str(path))
    if calibration is not None:
        Calibration.save(calibration, path.with_name(f"{path.stem}_calibration.json"))
    with open(path.with_name(f"{path.stem}_meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    return path


def main(kind):
    label = "moneyline" if kind == "ML" else "totals"
    parser = argparse.ArgumentParser(description=f"Train XGBoost {label} model with walk-forward CV.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset table name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--trials", type=int, default=100, help="Number of parameter trials.")
    parser.add_argument("--splits", type=int, default=5, help="Walk-forward CV splits.")
    parser.add_argument("--calibration", default="sigmoid", choices=["sigmoid", "none"])
    parser.add_argument("--no-refit", action="store_true",
                        help="Save the model trained on the first 90%% instead of refitting on all rows.")
    args = parser.parse_args()

    df = load_dataset(args.dataset)
    if df.empty:
        print(f"No rows found for dataset {args.dataset}.")
        return
    X, y, dates = prepare_data(df, kind)
    model, calibration, meta = train_pipeline(
        X, y, dates, args.trials, args.splits, args.seed, args.calibration, refit=not args.no_refit
    )
    path = save_artifacts(model, calibration, meta, kind)
    print(f"Saved model: {path}")
