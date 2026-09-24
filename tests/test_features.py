import math

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.pipeline import (
    build_features,
    make_full_pipeline,
    make_preprocessor,
    skill_columns,
)
from src.features.skills import SKILL_NAMES, extract_skills, skill_column, skill_matrix
from src.features.text import (
    extract_years,
    keyword_flags,
    model_text,
    remove_company,
)

# --- Skills -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Strong C++ and C# skills", {"C++", "C#"}),
        (".NET Core and ASP.NET MVC", {".NET", "ASP.NET"}),
        ("React.js, Node.js and TypeScript", {"React", "Node.js", "TypeScript"}),
        ("C/C++ on embedded systems", {"C", "C++", "Embedded Systems"}),
        ("AWS, Docker, Kubernetes (k8s)", {"AWS", "Docker", "Kubernetes"}),
    ],
)
def test_extract_skills_handles_symbols(text, expected):
    assert expected <= set(extract_skills(text))


@pytest.mark.parametrize(
    "text, not_expected",
    [
        ("Java and Spring Boot", "JavaScript"),
        ("JavaScript developer", "Java"),
        ("Build React Native apps", "React"),
        ("React quickly to production incidents", "React"),
        ("You will excel in a fast-paced team", "Excel"),
        ("Let's go to market together", "Go"),
        ("Guard rails for data quality", "Ruby on Rails"),
    ],
)
def test_extract_skills_avoids_false_positives(text, not_expected):
    assert not_expected not in extract_skills(text)


def test_extract_skills_empty_input():
    assert extract_skills("") == []
    assert extract_skills(None) == []


def test_skill_column_is_unique_and_safe():
    columns = [skill_column(n) for n in SKILL_NAMES]
    assert len(set(columns)) == len(columns)
    assert skill_column("C++") == "skill_c_plus_plus"
    assert skill_column("C#") == "skill_c_sharp"


def test_skill_matrix_shape_and_counts():
    texts = pd.Series(["Python and SQL", "", None], index=[10, 11, 12])
    m = skill_matrix(texts, n_jobs=1)
    assert list(m.index) == [10, 11, 12]
    assert m.shape[1] == len(SKILL_NAMES) + 1
    assert m.loc[10, "skill_python"] == 1 and m.loc[10, "skill_sql"] == 1
    assert m["skill_count"].tolist() == [2, 0, 0]


# --- Years of experience ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("3+ years of experience", 3),
        ("5-7 years of relevant experience", 5),
        ("two (2) years of professional experience", 2),
        ("At least 2 yrs. experience", 2),
        ("3 years experience required; 5+ years preferred experience", 3),
    ],
)
def test_extract_years(text, expected):
    assert extract_years(text) == expected


@pytest.mark.parametrize(
    "text",
    ["our 25 years of experience in the industry", "a 10 years old company", "", None],
)
def test_extract_years_returns_nan_when_absent_or_implausible(text):
    assert math.isnan(extract_years(text))


def test_keyword_flags():
    flags = keyword_flags(pd.Series(["You will mentor interns", "Principal architect"]))
    assert flags.loc[0, ["kw_mentor", "kw_intern"]].tolist() == [1, 1]
    assert flags.loc[1, ["kw_principal", "kw_architect", "kw_lead"]].tolist() == [
        1,
        1,
        0,
    ]


# --- Feature pipeline ----------------------------------------------------------------


def test_remove_company_is_case_insensitive_and_whole_phrase():
    text = "Join TEKsystems today! teksystems rocks. TEKsystemsX stays."
    assert (
        remove_company(text, "TEKsystems")
        == "Join   today!   rocks. TEKsystemsX stays."
    )
    assert remove_company(text, None) == text


def test_model_text_can_exclude_title():
    text = model_text(
        "Senior Engineer", "Build <b>APIs</b> at Acme", "Acme", include_title=False
    )
    assert "Senior" not in text
    assert "Acme" not in text
    assert "Build APIs" in text


def _toy_postings():
    return pd.DataFrame(
        {
            "title": [
                "Senior Python Engineer",
                "Junior Web Developer",
                "Director of AI",
                "Intern",
            ]
            * 3,
            "description": [
                "Lead and mentor a team. 7+ years of experience with Python and AWS.",
                "Entry level role. 1 year of experience with JavaScript and React.",
                "Set strategy for machine learning. 10 years of experience required.",
                "Summer internship for students learning SQL.",
            ]
            * 3,
            "company_name": ["Acme", "Globex", "Initech", "Acme"] * 3,
        }
    )


def test_build_features_without_title_has_no_title_signal():
    df = _toy_postings()
    with_title = build_features(df, include_title=True, n_jobs=1)
    without_title = build_features(df, include_title=False, n_jobs=1)
    assert (
        with_title.loc[3, "kw_intern"] == 1
    )  # from the title "Intern" (and description)
    assert "Senior" in with_title.loc[0, "text"]
    assert "Senior" not in without_title.loc[0, "text"]
    assert without_title.loc[0, "years_required"] == 7
    assert without_title.loc[0, "skill_python"] == 1


def test_full_pipeline_predicts_from_raw_rows():
    df = _toy_postings()
    y = ["Senior", "Entry", "Senior", "Entry"] * 3
    features = build_features(df, n_jobs=1)
    model = Pipeline(
        [
            ("preprocess", make_preprocessor(skill_columns(features), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    ).fit(features, y)
    full = make_full_pipeline(model)
    new = pd.DataFrame(
        {
            "title": ["Intern"],
            "description": ["Internship for students"],
            "company_name": [None],
        }
    )
    assert full.predict(new)[0] in {"Entry", "Senior"}
    assert full.predict_proba(new).shape == (1, 2)
