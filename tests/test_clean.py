import numpy as np
import pandas as pd
import pytest

from src.data.clean import (
    add_yearly_salary,
    clean_text,
    is_tech_title,
    map_seniority,
    parse_locations,
    to_yearly,
)

# --- Salary normalization ----------------------------------------------------


@pytest.mark.parametrize(
    "amount, period, expected",
    [
        (100_000, "YEARLY", 100_000),
        (50, "HOURLY", 104_000),  # 50 x 2080
        (5_000, "MONTHLY", 60_000),  # 5000 x 12
        (2_000, "WEEKLY", 104_000),
        (4_000, "BIWEEKLY", 104_000),
        (50, "hourly", 104_000),  # case-insensitive
    ],
)
def test_to_yearly_converts_each_pay_period(amount, period, expected):
    result = to_yearly(pd.Series([amount]), pd.Series([period]))
    assert result.iloc[0] == pytest.approx(expected)


def test_to_yearly_unknown_or_missing_period_is_nan():
    result = to_yearly(pd.Series([100.0, 100.0]), pd.Series(["DAILY", None]))
    assert result.isna().all()


def _salary_frame(rows):
    cols = ["min_salary", "med_salary", "max_salary", "pay_period", "currency"]
    return pd.DataFrame(rows, columns=cols)


def test_salary_prefers_median_then_midpoint():
    df = _salary_frame(
        [
            [None, 90_000, None, "YEARLY", "USD"],
            [80_000, None, 120_000, "YEARLY", "USD"],
            [40, None, 60, "HOURLY", "USD"],
        ]
        # Filler rows so the IQR bounds are wide enough to keep the rows above.
        + [[None, s, None, "YEARLY", "USD"] for s in range(70_000, 131_000, 15_000)]
    )
    out, _ = add_yearly_salary(df)
    assert out["salary_yearly"].iloc[:3].tolist() == pytest.approx(
        [90_000, 100_000, 104_000]
    )
    assert out["salary_min"].iloc[2] == pytest.approx(83_200)


def test_salary_drops_non_usd_and_iqr_outliers():
    typical = [[None, s, None, "YEARLY", "USD"] for s in range(80_000, 121_000, 5_000)]
    df = _salary_frame(
        typical
        + [
            [None, 90_000, None, "YEARLY", "EUR"],  # non-USD
            [None, 50_000_000, None, "YEARLY", "USD"],  # absurd outlier
            [None, 1, None, "YEARLY", "USD"],  # absurd outlier
        ]
    )
    out, (low, high) = add_yearly_salary(df)
    assert out["salary_yearly"].iloc[: len(typical)].notna().all()
    assert out["salary_yearly"].iloc[len(typical) :].isna().all()
    assert low < 80_000 and high > 120_000


# --- Seniority label mapping ---------------------------------------------------


def test_map_seniority_default_mapping():
    levels = pd.Series(
        [
            "Internship",
            "Entry level",
            "Associate",
            "Mid-Senior level",
            "Director",
            "Executive",
            None,
            "Unknown",
        ]
    )
    expected = ["Entry", "Entry", "Mid", "Mid", "Senior", "Senior", np.nan, np.nan]
    assert map_seniority(levels).tolist() == pytest.approx(expected, nan_ok=True)


def test_map_seniority_custom_mapping():
    levels = pd.Series(["Associate", "Director"])
    result = map_seniority(levels, {"Associate": "Entry", "Director": "Senior"})
    assert result.tolist() == ["Entry", "Senior"]


# --- Other helpers --------------------------------------------------------------


def test_clean_text_removes_html_urls_and_extra_whitespace():
    raw = "<p>Build&nbsp;APIs</p>\n\n Apply at https://example.com/job?id=1 or www.x.io  now"
    assert clean_text(raw) == "Build APIs Apply at or now"
    assert clean_text(None) == ""


def test_is_tech_title():
    titles = pd.Series(
        [
            "Senior Software Engineer",
            "Data Analyst - Power BI",
            "Java Architect",
            "Mechanical Engineer",
            "Sales Engineer",
            "Registered Nurse",
            None,
        ]
    )
    assert is_tech_title(titles).tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_parse_locations():
    locs = pd.Series(
        [
            "Austin, TX",
            "Los Angeles, California, United States",
            "United States",
            "Texas",
        ]
    )
    out = parse_locations(locs)
    assert out["state"].tolist()[:2] == ["TX", "CA"]
    assert out["location_level"].tolist() == ["city", "city", "country", "state"]
    assert out["state"].iloc[3] == "TX"
