"""Analyze a single job posting: seniority, role cluster, skills, and related skills.

Used by the Streamlit app, and importable anywhere:

    from src.models.predict import analyze, load_models
    result = analyze("Data Engineer", "5+ years with Spark and Airflow ...", load_models(), skills_df)
"""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.skills import extract_skills

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

MIN_WORDS = 15  # below this, predictions are not meaningful
LOW_CONFIDENCE_WORDS = 60  # below this, show a reliability warning


@dataclass
class Analysis:
    words: int
    seniority: str | None = None
    seniority_proba: dict[str, float] = field(default_factory=dict)
    cluster: str | None = None
    skills: list[str] = field(default_factory=list)
    related: list[tuple[str, float]] = field(default_factory=list)
    warning: str | None = None


def load_models(models_dir: Path = MODELS_DIR) -> dict:
    return {
        "seniority": joblib.load(models_dir / "seniority.joblib"),
        "clusters": joblib.load(models_dir / "clusters.joblib"),
    }


def related_skills(
    found: list[str],
    skills: pd.DataFrame,
    mask: np.ndarray | None = None,
    top: int = 6,
    min_share: float = 0.10,
) -> list[tuple[str, float]]:
    """Skills that similar postings commonly list, which the input doesn't mention.

    "Similar" means postings in the same role cluster (mask) that share at least
    one of the found skills. Postings that share more skills count more. Returns
    (skill, share of those postings that mention it).
    """
    pool = skills if mask is None else skills.loc[mask]
    found = [s for s in found if s in pool.columns]
    if found:
        overlap = pool[found].sum(axis=1)
        pool, weights = pool.loc[overlap > 0], overlap[overlap > 0]
    else:
        weights = pd.Series(1.0, index=pool.index)
    if len(pool) < 20:  # too few similar postings to say anything
        return []
    share = pool.mul(weights, axis=0).sum() / weights.sum()
    share = share.drop(labels=found).sort_values(ascending=False)
    share = share[share >= min_share].head(top)
    return [(name, float(value)) for name, value in share.items()]


def analyze(
    title: str,
    description: str,
    models: dict,
    skills: pd.DataFrame | None = None,
    clusters: pd.Series | None = None,
) -> Analysis:
    """Run both models on one posting.

    skills: 0/1 skill matrix of the reference postings (columns = skill names).
    clusters: cluster name per reference posting (same index), used to restrict
    the related-skills search to the predicted role type.
    """
    title, description = (title or "").strip(), (description or "").strip()
    words = len(f"{title} {description}".split())
    result = Analysis(words=words)
    if len(description.split()) < MIN_WORDS:
        result.warning = (
            f"Please paste a longer job description (at least {MIN_WORDS} words). "
            "The models need the responsibilities and requirements to work."
        )
        return result
    if words < LOW_CONFIDENCE_WORDS:
        result.warning = (
            "Short description: predictions are less reliable than for a full posting."
        )

    row = pd.DataFrame(
        {"title": [title], "description": [description], "company_name": [None]}
    )

    sen = models["seniority"]
    proba = sen["pipeline"].predict_proba(row)[0]
    result.seniority_proba = {c: float(p) for c, p in zip(sen["classes"], proba)}
    result.seniority = sen["classes"][int(np.argmax(proba))]

    clu = models["clusters"]
    result.cluster = clu["names"][int(clu["pipeline"].predict(row)[0])]

    result.skills = extract_skills(f"{title} {description}")
    if skills is not None:
        mask = None if clusters is None else (clusters == result.cluster).to_numpy()
        result.related = related_skills(result.skills, skills, mask)
    return result
