import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src" / "Train-Models"))

import xgb_training as T  # noqa: E402

META_KEYS = {"params", "num_boost_round", "feature_count", "trained_rows", "refit_on_all",
             "heldout", "cv_log_loss", "trials"}
HELDOUT_KEYS = {"n", "start_date", "accuracy", "log_loss", "brier", "accuracy_raw", "log_loss_raw"}


def synthetic_frame(n=900, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.random((n, 6)) * 10, columns=[f"f{i}" for i in range(6)])
    df.insert(0, "index", np.arange(n))
    df["Score"] = 200
    df["TEAM_NAME"] = "A"
    df["Date"] = pd.date_range("2020-01-01", periods=n).astype(str)
    df["OU"] = 220.0
    latent = df["f0"] + rng.normal(0, 3, n)
    df["Home-Team-Win"] = (latent > 5).astype(int)
    df["OU-Cover"] = (latent > 5).astype(float)
    df.loc[df.index % 25 == 0, "OU-Cover"] = 2.0  # pushes
    return df.sample(frac=1, random_state=1)  # unsorted on purpose


def small_run(kind="ML", n=900, refit=True):
    X, y, dates = T.prepare_data(synthetic_frame(n), kind)
    return X, y, dates, T.train_pipeline(X, y, dates, trials=2, n_splits=3, seed=1, refit=refit, log=lambda m: None)


class PrepareDataTest(unittest.TestCase):
    def test_push_rows_dropped_for_uo_only(self):
        df = synthetic_frame()
        pushes = int((df["OU-Cover"] == 2).sum())
        self.assertGreater(pushes, 0)
        X_uo, y_uo, _ = T.prepare_data(df, "UO")
        X_ml, y_ml, _ = T.prepare_data(df, "ML")
        self.assertEqual(len(y_uo), len(df) - pushes)
        self.assertEqual(set(y_uo), {0, 1})
        self.assertEqual(len(y_ml), len(df))

    def test_sorted_by_date_and_columns(self):
        df = synthetic_frame()
        X_ml, _, dates = T.prepare_data(df, "ML")
        X_uo, _, _ = T.prepare_data(df, "UO")
        self.assertTrue(np.all(np.diff(dates.astype("datetime64[ns]").astype(np.int64)) >= 0))
        self.assertEqual(X_uo.shape[1], X_ml.shape[1] + 1)  # UO keeps the OU line


class CvTest(unittest.TestCase):
    def test_validation_fold_disjoint_from_fit_and_early_stopping(self):
        folds = list(T.cv_folds(1000, 5))
        self.assertEqual(len(folds), 5)
        for fit_idx, es_idx, val_idx in folds:
            self.assertEqual(len(set(fit_idx) & set(es_idx)), 0)
            self.assertEqual(len(set(es_idx) & set(val_idx)), 0)
            self.assertEqual(len(set(fit_idx) & set(val_idx)), 0)
            self.assertLess(fit_idx.max(), es_idx.min())
            self.assertLess(es_idx.max(), val_idx.min())  # time order
            self.assertAlmostEqual(len(es_idx) / (len(es_idx) + len(fit_idx)), 0.1, delta=0.01)


class PipelineTest(unittest.TestCase):
    def test_meta_schema(self):
        X, y, dates, (model, calibration, meta) = small_run()
        self.assertEqual(set(meta), META_KEYS)
        self.assertEqual(set(meta["heldout"]), HELDOUT_KEYS)
        self.assertEqual(meta["feature_count"], X.shape[1])
        self.assertEqual(meta["trials"], 2)
        self.assertIsInstance(meta["num_boost_round"], int)
        self.assertEqual(meta["params"]["objective"], "binary:logistic")
        self.assertEqual(meta["params"]["eval_metric"], "logloss")
        self.assertNotIn("num_class", meta["params"])
        json.dumps(meta)  # serialisable
        self.assertEqual(meta["heldout"]["n"], len(X) - T.split_train_test(len(X)))

    def test_refit_uses_all_rows_but_heldout_is_from_90_percent_model(self):
        X, y, dates, (model, _, meta) = small_run(refit=True)
        self.assertTrue(meta["refit_on_all"])
        self.assertEqual(meta["trained_rows"], len(X))
        X, y, dates, (model, _, meta) = small_run(refit=False)
        self.assertFalse(meta["refit_on_all"])
        self.assertEqual(meta["trained_rows"], T.split_train_test(len(X)))

    def test_heldout_metrics_independent_of_refit(self):
        _, _, _, (_, _, meta_a) = small_run(refit=True)
        _, _, _, (_, _, meta_b) = small_run(refit=False)
        self.assertEqual(meta_a["heldout"], meta_b["heldout"])

    def test_oof_calibration_never_touches_test_rows(self):
        X, y, dates = T.prepare_data(synthetic_frame(), "ML")
        test_start = T.split_train_test(len(X))
        X_tv, y_tv, X_test, y_test = X[:test_start], y[:test_start], X[test_start:], y[test_start:]
        best = T.random_search(X_tv, y_tv, 1, 3, 1, log=lambda m: None)
        self.assertLess(best["oof_idx"].max(), test_start)
        self.assertEqual(len(best["oof_idx"]), len(best["oof_probs"]))
        _, cal_a, _ = T.evaluate_heldout(X_tv, y_tv, X_test, y_test, best, y_tv[best["oof_idx"]])
        _, cal_b, _ = T.evaluate_heldout(X_tv, y_tv, X_test, 1 - y_test, best, y_tv[best["oof_idx"]])
        self.assertEqual(cal_a, cal_b)  # flipping the test labels cannot change the calibration

    def test_save_artifacts_names_and_files(self):
        import datetime
        _, _, _, (model, calibration, meta) = small_run()
        with tempfile.TemporaryDirectory() as tmp:
            path = T.save_artifacts(model, calibration, meta, "UO", tmp, today=datetime.date(2026, 1, 2))
            acc = f"{meta['heldout']['accuracy'] * 100:.1f}"
            self.assertEqual(path.name, f"XGBoost_{acc}%_UO_20260102.json")
            self.assertTrue(path.exists())
            self.assertTrue(path.with_name(f"{path.stem}_calibration.json").exists())
            loaded = json.loads(path.with_name(f"{path.stem}_meta.json").read_text())
            self.assertEqual(set(loaded), META_KEYS)


if __name__ == "__main__":
    unittest.main()
