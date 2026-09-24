from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.estimators import BalancedXGBClassifier
from src.models.train_classifier import evaluate, group_split


def test_group_split_keeps_companies_on_one_side():
    rng = np.random.default_rng(0)
    groups = np.array([f"company_{i}" for i in rng.integers(0, 60, size=600)])
    y = rng.choice([0, 1, 2], size=600, p=[0.3, 0.65, 0.05])
    train_idx, test_idx = group_split(y, groups)
    assert not set(groups[train_idx]) & set(groups[test_idx])
    assert 0.1 < len(test_idx) / len(y) < 0.3
    assert set(y[test_idx]) == {0, 1, 2}  # every class reaches the test set


def test_balanced_xgb_upweights_rare_class():
    X = np.vstack([np.zeros((95, 1)), np.ones((5, 1))])
    y = np.array([0] * 95 + [1] * 5)
    model = BalancedXGBClassifier(n_estimators=20, max_depth=2, random_state=0).fit(
        X, y
    )
    assert model.predict(np.array([[1.0]]))[0] == 1


def test_evaluate_macro_f1_and_confusion_matrix():
    result = evaluate(np.array([0, 1, 2, 2]), np.array([0, 1, 2, 1]))
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    assert 0 < result["macro_f1"] < 1


@pytest.mark.parametrize("name", ["seniority", "seniority_notitle"])
def test_saved_model_loads_and_predicts(name):
    path = Path(__file__).resolve().parents[1] / "models" / f"{name}.joblib"
    if not path.exists():
        pytest.skip("model not trained")
    bundle = joblib.load(path)
    row = pd.DataFrame(
        {
            "title": ["Data Engineer"],
            "description": ["5+ years of experience with Spark"],
            "company_name": [None],
        }
    )
    proba = bundle["pipeline"].predict_proba(row)
    assert proba.shape == (1, len(bundle["classes"]))
    assert abs(proba.sum() - 1) < 1e-5
