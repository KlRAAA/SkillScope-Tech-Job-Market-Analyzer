"""Feature pipeline: raw postings -> model-ready matrix.

Two stages, so that expensive but stateless work runs once:

1. build_features(df): stateless. Builds the model text (company name removed)
   and the hand-crafted columns (skills, years, keyword flags, length). No
   fitting, so computing it once on the full dataset cannot leak test data.
   It is cached with `python -m src.features.pipeline`.
2. make_preprocessor(): a ColumnTransformer with the fitted parts (TF-IDF
   vocabulary, imputer, scaler). It is fitted on training data only, inside
   the model Pipeline.

make_full_pipeline() chains both, so a saved model accepts raw
title/description/company rows at prediction time.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.features.skills import skill_matrix
from src.features.text import (
    KEYWORD_FLAGS,
    cluster_text,
    extract_years,
    keyword_flags,
    make_tfidf,
    model_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"

NUMERIC_COLUMNS = ["log_desc_words", "years_required", "skill_count"]
FLAG_COLUMNS = list(KEYWORD_FLAGS)


def build_features(
    df: pd.DataFrame, include_title: bool = True, n_jobs: int = -1
) -> pd.DataFrame:
    """Stateless features from columns title, description (+ optional company_name).

    Keyword flags and skills are computed on the same text the model sees, so
    with include_title=False no information from the title gets in.
    """
    description = df["description"] if "description" in df else df["description_clean"]
    company = (
        df["company_name"] if "company_name" in df else pd.Series(None, index=df.index)
    )
    text = pd.Series(
        [
            model_text(t, d, c, include_title)
            for t, d, c in zip(df["title"], description, company)
        ],
        index=df.index,
    )
    desc_only = pd.Series(
        [
            model_text(None, d, c, include_title=False)
            for d, c in zip(description, company)
        ],
        index=df.index,
    )
    skills = skill_matrix(text, n_jobs=n_jobs)

    features = pd.DataFrame(
        {
            "text": text,
            "log_desc_words": np.log1p(desc_only.str.split().str.len()).astype(
                np.float32
            ),
            "years_required": desc_only.map(extract_years).astype(np.float32),
        },
        index=df.index,
    )
    return pd.concat([features, keyword_flags(text), skills], axis=1)


def cluster_texts(df: pd.DataFrame) -> list[str]:
    """Clustering text for each row (title, description[_clean], optional company_name)."""
    description = df["description"] if "description" in df else df["description_clean"]
    company = df["company_name"] if "company_name" in df else [None] * len(df)
    return [cluster_text(t, d, c) for t, d, c in zip(df["title"], description, company)]


def skill_columns(features: pd.DataFrame) -> list[str]:
    return [
        c for c in features.columns if c.startswith("skill_") and c != "skill_count"
    ]


def make_preprocessor(
    skill_cols: list[str], max_features: int = 20_000, min_df: int = 5
) -> ColumnTransformer:
    """TF-IDF on text + scaled numeric features + 0/1 flags and skills."""
    numeric = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
    )
    return ColumnTransformer(
        [
            ("tfidf", make_tfidf(max_features, min_df), "text"),
            ("numeric", numeric, NUMERIC_COLUMNS),
            ("binary", "passthrough", FLAG_COLUMNS + skill_cols),
        ],
        sparse_threshold=1.0,  # keep the output sparse
    )


def make_full_pipeline(
    model_pipeline: Pipeline, include_title: bool = True
) -> Pipeline:
    """Prepend the stateless feature step to a fitted preprocessor + model pipeline."""
    features = FunctionTransformer(
        build_features, kw_args={"include_title": include_title, "n_jobs": 1}
    )
    return Pipeline([("features", features), *model_pipeline.steps])


def main() -> None:
    """Cache features for all postings, with and without the title."""
    jobs = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "jobs.parquet")
    with_title = build_features(jobs, include_title=True)
    without_title = build_features(jobs, include_title=False)
    # Only the columns that depend on the title need a second version.
    title_dependent = ["text", *FLAG_COLUMNS, "skill_count", *skill_columns(with_title)]
    out = pd.concat(
        [
            pd.DataFrame({"job_id": jobs["job_id"].to_numpy()}, index=jobs.index),
            with_title,
            without_title[title_dependent].add_prefix("notitle__"),
        ],
        axis=1,
    )
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(FEATURES_PATH, index=False)
    print(
        f"Saved {out.shape[0]:,} rows x {out.shape[1]} columns to {FEATURES_PATH.relative_to(PROJECT_ROOT)}"
    )


def load_features(include_title: bool = True) -> pd.DataFrame:
    """Load cached features (one version) aligned to jobs.parquet rows."""
    cached = pd.read_parquet(FEATURES_PATH)
    prefix = "notitle__"
    base = [c for c in cached.columns if not c.startswith(prefix)]
    if include_title:
        return cached[base]
    swapped = {
        c[len(prefix) :]: cached[c] for c in cached.columns if c.startswith(prefix)
    }
    return cached[base].assign(**swapped)


if __name__ == "__main__":
    main()
