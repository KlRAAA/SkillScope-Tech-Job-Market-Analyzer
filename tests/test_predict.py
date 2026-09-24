from pathlib import Path

import pandas as pd
import pytest

from src.models.predict import MIN_WORDS, analyze, load_models, related_skills

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _skills():
    # 50 postings: Python+SQL postings also list Spark; Excel postings list Tableau.
    rows = [{"Python": 1, "SQL": 1, "Spark": 1, "Excel": 0, "Tableau": 0}] * 25
    rows += [{"Python": 0, "SQL": 0, "Spark": 0, "Excel": 1, "Tableau": 1}] * 25
    return pd.DataFrame(rows)


def test_related_skills_suggests_co_listed_skills_not_already_found():
    related = related_skills(["Python"], _skills())
    names = [name for name, _ in related]
    assert names[0] in {"SQL", "Spark"}
    assert "Python" not in names
    assert "Tableau" not in names  # never listed alongside Python
    assert all(0 < share <= 1 for _, share in related)


def test_related_skills_needs_enough_similar_postings():
    assert related_skills(["Python"], _skills().head(10)) == []


def test_related_skills_respects_cluster_mask():
    mask = [False] * 25 + [True] * 25
    names = [
        name
        for name, _ in related_skills([], _skills(), mask=pd.Series(mask).to_numpy())
    ]
    assert set(names) == {"Excel", "Tableau"}


@pytest.mark.parametrize("description", ["", "   ", "Python developer needed"])
def test_analyze_rejects_short_input_without_running_models(description):
    result = analyze("Engineer", description, models={})
    assert result.seniority is None and result.cluster is None
    assert str(MIN_WORDS) in result.warning


def test_analyze_end_to_end():
    if (
        not (MODELS_DIR / "seniority.joblib").exists()
        or not (MODELS_DIR / "clusters.joblib").exists()
    ):
        pytest.skip("models not trained")
    description = (
        "Troubleshoot hardware, Windows and printer issues for end users, reset passwords in "
        "Active Directory, and manage tickets in ServiceNow. Entry level role, training provided."
    )
    result = analyze("Help Desk Technician", description, load_models(MODELS_DIR))
    assert result.seniority in {"Entry", "Mid", "Senior"}
    assert abs(sum(result.seniority_proba.values()) - 1) < 1e-5
    assert result.cluster == "IT Support & Help Desk"
    assert {"Active Directory", "ServiceNow"} <= set(result.skills)
    assert result.warning is not None  # short description -> reliability note
