"""Train the seniority classifier (Entry / Mid / Senior).

Evaluation design:
- Test split: ~20% of postings, grouped by company (StratifiedGroupKFold), so
  no company appears in both train and test. This avoids scoring well on
  near-identical template postings, and answers "does it work for a company
  it has never seen?".
- Model selection: 5-fold StratifiedGroupKFold on the training set only. The
  TF-IDF vocabulary is fitted inside each fold (it is part of the Pipeline).
- Main metric: macro F1. Every class counts equally, so the tiny Senior class
  (2.7%) cannot be ignored.
- The whole procedure runs twice: with the job title, and without it.

Usage:
    python -m src.models.train_classifier
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.features.pipeline import (
    load_features,
    make_full_pipeline,
    make_preprocessor,
    skill_columns,
)
from src.models.estimators import BalancedXGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOBS_PATH = PROJECT_ROOT / "data" / "processed" / "jobs.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "seniority_test_predictions.parquet"
)

CLASSES = ["Entry", "Mid", "Senior"]
SEED = 42
N_FOLDS = 5


def candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            C=3.0, class_weight="balanced", max_iter=3000
        ),
        "Linear SVM": LinearSVC(C=0.3, class_weight="balanced", max_iter=5000),
        "XGBoost": BalancedXGBClassifier(
            n_estimators=200,
            learning_rate=0.15,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.5,
            tree_method="hist",
            n_jobs=-1,
            random_state=SEED,
        ),
    }


PARAM_GRIDS = {
    "Logistic Regression": {
        "clf__C": [0.3, 1, 3, 10],
        "preprocess__tfidf__max_features": [10_000, 20_000],
    },
    "Linear SVM": {
        "clf__C": [0.03, 0.1, 0.3, 1],
        "preprocess__tfidf__max_features": [10_000, 20_000],
    },
    "XGBoost": {
        # Small grid: one XGBoost fit takes ~2-3 minutes on a 4-core laptop.
        "clf__max_depth": [4, 6],
    },
}


def load_data(include_title: bool):
    """Labeled rows: features X, integer labels y, company groups, and job info."""
    jobs = pd.read_parquet(JOBS_PATH)
    features = load_features(include_title)
    labeled = jobs["seniority"].notna().to_numpy()
    jobs, X = jobs.loc[labeled].reset_index(drop=True), features.loc[
        labeled
    ].reset_index(drop=True)
    y = jobs["seniority"].astype(str).map(CLASSES.index).to_numpy()
    # Postings without a company name each get their own group.
    groups = (
        jobs["company_name"]
        .fillna("job_" + jobs["job_id"].astype(str))
        .str.lower()
        .to_numpy()
    )
    return X, y, groups, jobs


def group_split(y, groups):
    """One fold of a 5-fold StratifiedGroupKFold = ~20% test, no shared companies."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y, groups))
    return train_idx, test_idx


# Trees don't need 20k sparse word features, and XGBoost on all of them took
# ~5 min per fit. It gets the top 3k TF-IDF terms (plus all hand-crafted features).
TFIDF_FEATURES = {"XGBoost": 3_000}


def make_model(name, clf, skill_cols) -> Pipeline:
    preprocess = make_preprocessor(
        skill_cols, max_features=TFIDF_FEATURES.get(name, 20_000)
    )
    return Pipeline([("preprocess", preprocess), ("clf", clf)])


def cv_splitter():
    return StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def compare_models(X, y, groups, skill_cols) -> pd.DataFrame:
    rows = []
    for name, clf in candidate_models().items():
        start = time.time()
        scores = cross_validate(
            make_model(name, clf, skill_cols),
            X,
            y,
            groups=groups,
            cv=cv_splitter(),
            scoring=["f1_macro", "balanced_accuracy"],
            n_jobs=1,
        )
        rows.append(
            {
                "model": name,
                "cv_macro_f1_mean": scores["test_f1_macro"].mean(),
                "cv_macro_f1_std": scores["test_f1_macro"].std(),
                "cv_balanced_acc_mean": scores["test_balanced_accuracy"].mean(),
                "seconds": time.time() - start,
            }
        )
        print(
            f"  {name}: macro F1 {rows[-1]['cv_macro_f1_mean']:.3f} ± {rows[-1]['cv_macro_f1_std']:.3f}"
        )
    return pd.DataFrame(rows).sort_values(
        "cv_macro_f1_mean", ascending=False, ignore_index=True
    )


def tune(name, X, y, groups, skill_cols) -> GridSearchCV:
    search = GridSearchCV(
        make_model(name, candidate_models()[name], skill_cols),
        PARAM_GRIDS[name],
        scoring="f1_macro",
        cv=cv_splitter(),
        n_jobs=1,
        refit=True,
    )
    search.fit(X, y, groups=groups)
    return search


def evaluate(y_true, y_pred) -> dict:
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=range(3),
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(3)).tolist(),
    }


def run(
    include_title: bool, model_name: str | None = None, params: dict | None = None
) -> dict:
    """Full procedure for one variant. With model_name/params set, skip selection and tuning."""
    variant = "with_title" if include_title else "without_title"
    print(f"\n=== {variant} ===")
    X, y, groups, jobs = load_data(include_title)
    skill_cols = skill_columns(X)
    train_idx, test_idx = group_split(y, groups)
    X_train, y_train, g_train = X.iloc[train_idx], y[train_idx], groups[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    assert not set(g_train) & set(
        groups[test_idx]
    ), "a company is in both train and test"

    result = {
        "split": {
            "train": len(train_idx),
            "test": len(test_idx),
            "train_companies": len(set(g_train)),
            "test_companies": len(set(groups[test_idx])),
            "test_class_counts": dict(
                zip(CLASSES, np.bincount(y_test, minlength=3).tolist())
            ),
        }
    }

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    result["baseline_test"] = evaluate(y_test, baseline.predict(X_test))
    print(f"  baseline macro F1 (always '{CLASSES[baseline.classes_[np.argmax(baseline.class_prior_)]]}'): "
          f"{result['baseline_test']['macro_f1']:.3f}")  # fmt: skip

    if model_name is None:
        comparison = compare_models(X_train, y_train, g_train, skill_cols)
        result["cv_comparison"] = comparison.to_dict(orient="records")
        model_name = comparison.loc[0, "model"]
        print(f"  tuning {model_name} ...")
        search = tune(model_name, X_train, y_train, g_train, skill_cols)
        params = search.best_params_
        result["tuning"] = {
            "model": model_name,
            "best_params": params,
            "best_cv_macro_f1": search.best_score_,
            "grid": [
                {"params": p, "mean": m, "std": s}
                for p, m, s in zip(
                    search.cv_results_["params"],
                    search.cv_results_["mean_test_score"],
                    search.cv_results_["std_test_score"],
                )
            ],
        }
        model = search.best_estimator_
    else:
        model = make_model(
            model_name, candidate_models()[model_name], skill_cols
        ).set_params(**params)
        scores = cross_validate(clone(model), X_train, y_train, groups=g_train, cv=cv_splitter(),
                                scoring="f1_macro")  # fmt: skip
        result["cv_macro_f1"] = {
            "mean": scores["test_score"].mean(),
            "std": scores["test_score"].std(),
        }
        model.fit(X_train, y_train)

    result["model"] = model_name
    result["params"] = params
    y_pred = model.predict(X_test)
    result["test"] = evaluate(y_test, y_pred)
    print(f"  TEST macro F1: {result['test']['macro_f1']:.3f}")

    predictions = jobs.iloc[test_idx][
        ["job_id", "title", "company_name", "description_clean"]
    ].copy()
    predictions["true"] = [CLASSES[i] for i in y_test]
    predictions["pred"] = [CLASSES[i] for i in y_pred]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        for i, c in enumerate(CLASSES):
            predictions[f"p_{c}"] = proba[:, i]
    predictions["variant"] = variant
    return {"result": result, "model": model, "predictions": predictions}


def main() -> None:
    with_title = run(include_title=True)
    # Same model and hyperparameters without the title, so only the input changes.
    without_title = run(
        include_title=False,
        model_name=with_title["result"]["model"],
        params=with_title["result"]["params"],
    )

    MODELS_DIR.mkdir(exist_ok=True)
    for name, out, include_title in [
        ("seniority", with_title, True),
        ("seniority_notitle", without_title, False),
    ]:
        full = make_full_pipeline(out["model"], include_title=include_title)
        joblib.dump({"pipeline": full, "classes": CLASSES, "model": out["result"]["model"]},
                    MODELS_DIR / f"{name}.joblib", compress=3)  # fmt: skip

    pd.concat([with_title["predictions"], without_title["predictions"]]).to_parquet(
        PREDICTIONS_PATH, index=False
    )

    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    metrics["seniority"] = {
        "classes": CLASSES,
        "split": "StratifiedGroupKFold by company (1 of 5 folds as test)",
        "with_title": with_title["result"],
        "without_title": without_title["result"],
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nSaved models to {MODELS_DIR.relative_to(PROJECT_ROOT)}/ and metrics to "
          f"{METRICS_PATH.relative_to(PROJECT_ROOT)}")  # fmt: skip


if __name__ == "__main__":
    main()
