import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.Predict import XGBoost_Runner as Runner  # noqa: E402
from src.Utils import Calibration  # noqa: E402

N_STATS = 104


def train_tiny(n_features, signal_column, seed=0):
    """Binary model whose probability rises with one chosen column."""
    rng = np.random.default_rng(seed)
    X = rng.random((400, n_features)) * 50
    y = (X[:, signal_column] + rng.normal(0, 8, 400) > 25).astype(int)
    params = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.3, "seed": 1}
    return xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=20)


class RunnerTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        patches = [mock.patch.object(Runner, "MODEL_DIR", self.dir), mock.patch.dict(Runner._models, clear=True)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.tmp.cleanup)

    def write_model(self, kind, name, n_features, signal_column, calibration=True, meta=True):
        model = train_tiny(n_features, signal_column)
        path = self.dir / name
        model.save_model(str(path))
        if calibration:
            Calibration.save({"method": "sigmoid", "binary": True,
                              "classes": [{"slope": 1.0, "intercept": 0.0}]},
                             path.with_name(f"{path.stem}_calibration.json"))
        if meta:
            (path.with_name(f"{path.stem}_meta.json")).write_text(json.dumps({"feature_count": n_features}))
        return path, model


class SelectionTest(RunnerTestCase):
    def test_auxiliary_files_are_not_models(self):
        model_path, _ = self.write_model("ML", "XGBoost_60.0%_ML_20260101.json", N_STATS + 2, 0)
        # newer than the model, so a broken filter would pick them
        for suffix in ("_calibration", "_meta"):
            (self.dir / f"XGBoost_99.0%_ML_20260101{suffix}.json").write_text("{}")
        self.assertEqual(Runner._select_model_path("ML"), model_path)

    def test_short_names_match_kind(self):
        ml, _ = self.write_model("ML", "XGBoost_60.0%_ML_20260101.json", N_STATS + 2, 0)
        uo, _ = self.write_model("UO", "XGBoost_50.0%_UO_20260101.json", N_STATS + 3, 0)
        self.assertEqual(Runner._select_model_path("ML"), ml)
        self.assertEqual(Runner._select_model_path("UO"), uo)

    def test_missing_model_raises(self):
        with self.assertRaises(FileNotFoundError):
            Runner._select_model_path("ML")


class PredictTest(RunnerTestCase):
    def test_binary_output_becomes_two_columns_and_predict_contract(self):
        _, model = self.write_model("ML", "XGBoost_60.0%_ML_20260101.json", N_STATS + 2, 0)
        self.write_model("UO", "XGBoost_50.0%_UO_20260101.json", N_STATS + 3, 0)
        features = np.random.default_rng(3).random((3, N_STATS)) * 50
        rest = [[1, 2], [2, 2], [3, 1]]
        home, over = Runner.predict(features, [220.5, 230.0, 210.0], rest)
        expected = model.predict(xgb.DMatrix(np.column_stack([features, np.array(rest, float)])))
        np.testing.assert_allclose(home, expected, rtol=1e-5)  # identity calibration
        self.assertTrue(all(isinstance(p, float) and 0 <= p <= 1 for p in home + over))
        probs = Runner._load("ML")(np.column_stack([features, rest]))
        self.assertEqual(probs.shape, (3, 2))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-6)

    def test_missing_total_gives_none(self):
        self.write_model("ML", "XGBoost_60.0%_ML_20260101.json", N_STATS + 2, 0)
        self.write_model("UO", "XGBoost_50.0%_UO_20260101.json", N_STATS + 3, 0)
        features = np.random.default_rng(4).random((2, N_STATS)) * 50
        home, over = Runner.predict(features, [220.5, None], [[1, 2], [2, 2]])
        self.assertEqual(len(home), 2)
        self.assertIsNotNone(over[0])
        self.assertIsNone(over[1])

    def test_uo_feature_order_is_stats_then_line_then_rest(self):
        # The UO model only reacts to column N_STATS (the line): probability must follow the total.
        self.write_model("ML", "XGBoost_60.0%_ML_20260101.json", N_STATS + 2, 0)
        _, uo_model = self.write_model("UO", "XGBoost_50.0%_UO_20260101.json", N_STATS + 3, N_STATS)
        features = np.full((2, N_STATS), 25.0)
        rest = [[1, 7], [1, 7]]
        _, over = Runner.predict(features, [5.0, 45.0], rest)
        expected = uo_model.predict(xgb.DMatrix(np.array([
            [*([25.0] * N_STATS), 5.0, 1, 7], [*([25.0] * N_STATS), 45.0, 1, 7]])))
        np.testing.assert_allclose(over, expected, rtol=1e-5)
        self.assertGreater(over[1], over[0] + 0.3)

    def test_missing_calibration_warns_and_returns_raw(self):
        _, model = self.write_model("ML", "XGBoost_60.0%_ML_20260101.json", N_STATS + 2, 0, calibration=False)
        features = np.random.default_rng(5).random((2, N_STATS)) * 50
        rest = [[1, 2], [2, 2]]
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            probs = Runner._load("ML")(np.column_stack([features, rest]))
        self.assertIn("kalibrasyon", err.getvalue())
        np.testing.assert_allclose(probs[:, 1], model.predict(xgb.DMatrix(np.column_stack([features, rest]))),
                                   rtol=1e-5)

    def test_corrupt_calibration_warns_and_returns_raw(self):
        path, _ = self.write_model("ML", "XGBoost_60.0%_ML_20260101.json", N_STATS + 2, 0)
        path.with_name(f"{path.stem}_calibration.json").write_text("{not json")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            probs = Runner._load("ML")(np.random.default_rng(6).random((2, N_STATS + 2)) * 50)
        self.assertIn("kalibrasyon", err.getvalue())
        self.assertEqual(probs.shape, (2, 2))

    def test_calibration_changes_probabilities(self):
        path, model = self.write_model("ML", "XGBoost_60.0%_ML_20260101.json", N_STATS + 2, 0)
        Calibration.save({"method": "sigmoid", "binary": True, "classes": [{"slope": 0.5, "intercept": 0.0}]},
                         path.with_name(f"{path.stem}_calibration.json"))
        data = np.random.default_rng(7).random((20, N_STATS + 2)) * 50
        raw = model.predict(xgb.DMatrix(data))
        calibrated = Runner._load("ML")(data)[:, 1]
        # slope 0.5 on the logit shrinks every probability toward 0.5
        self.assertTrue(np.all(np.abs(calibrated - 0.5) <= np.abs(raw - 0.5) + 1e-9))
        self.assertFalse(np.allclose(calibrated, raw))

    def test_old_softprob_model_still_works(self):
        rng = np.random.default_rng(8)
        X = rng.random((300, N_STATS + 2)) * 50
        y = (X[:, 0] > 25).astype(int)
        params = {"objective": "multi:softprob", "num_class": 2, "max_depth": 3, "seed": 1}
        model = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=10)
        path = self.dir / "XGBoost_65.8%_ML_md9_eta0p1_nb10.json"
        model.save_model(str(path))
        Calibration.save({"method": "sigmoid", "classes": [{"slope": 1.0, "intercept": 0.0},
                                                          {"slope": 1.0, "intercept": 0.0}]},
                         path.with_name(f"{path.stem}_calibration.json"))
        probs = Runner._load("ML")(X[:5])
        self.assertEqual(probs.shape, (5, 2))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-6)


class CalibrationTest(unittest.TestCase):
    def test_binary_roundtrip_and_two_column_unchanged(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.2, 0.8, 2000)
        y = (rng.random(2000) < np.clip(p * 0.5 + 0.25, 0, 1)).astype(int)  # overconfident model
        cal = Calibration.fit_sigmoid(p, y)
        self.assertTrue(cal["binary"])
        out = Calibration.apply(p, cal)
        self.assertEqual(out.shape, p.shape)
        self.assertLess(np.abs(out - 0.5).mean(), np.abs(p - 0.5).mean())
        two = Calibration.apply(np.column_stack([1 - p, p]), cal)
        np.testing.assert_allclose(two[:, 1], out)
        # 2-D multi-class behaviour untouched
        multi = Calibration.fit_sigmoid(np.column_stack([1 - p, p]), y)
        self.assertNotIn("binary", multi)
        self.assertEqual(Calibration.apply(np.column_stack([1 - p, p]), multi).shape, (2000, 2))


if __name__ == "__main__":
    unittest.main()
