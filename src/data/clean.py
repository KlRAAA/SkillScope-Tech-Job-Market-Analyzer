"""Turn the raw Kaggle CSVs into one clean table of tech postings.

Steps: load + join -> tech-title filter -> text cleaning -> dedupe ->
yearly salary + IQR -> location + work type -> seniority label.

Usage:
    python -m src.data.clean
"""

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "jobs.parquet"

POSTING_COLUMNS = [
    "job_id",
    "title",
    "description",
    "company_id",
    "company_name",
    "location",
    "formatted_experience_level",
    "formatted_work_type",
    "remote_allowed",
    "min_salary",
    "med_salary",
    "max_salary",
    "pay_period",
    "currency",
    "listed_time",
    "original_listed_time",
    "views",
    "applies",
]

# ---------------------------------------------------------------------------
# Tech-title filter
# ---------------------------------------------------------------------------

# A title is "tech" if it matches an include pattern and no exclude pattern.
TECH_INCLUDE = [
    r"software",
    r"developer",
    r"programmer",
    r"devops",
    r"\bdev ?sec ?ops\b",
    r"\bsre\b",
    r"site reliability",
    r"full[\s-]?stack",
    r"front[\s-]?end",
    r"back[\s-]?end",
    r"web (engineer|application)",
    r"\bdata (scientist|science|engineer|engineering|analyst|analytics|architect|modeler)",
    r"analytics engineer",
    r"machine learning",
    r"\bml\b",
    r"\bai\b",
    r"artificial intelligence",
    r"deep learning",
    r"\bnlp\b",
    r"computer vision",
    r"business intelligence",
    r"\bbi (developer|analyst|engineer)",
    r"\bcloud\b",
    r"\baws\b",
    r"\bazure\b",
    r"platform engineer",
    r"infrastructure engineer",
    r"\bsystems? (engineer|administrator|analyst)",
    r"sysadmin",
    r"\bnetwork (engineer|administrator|architect|analyst)",
    r"(solutions?|software|cloud|enterprise|data|security|technical|it|java|\.net|salesforce) architect",
    r"cyber ?security",
    r"information security",
    r"(application|network|cloud|it) security",
    r"security (engineer|analyst|architect)",
    r"penetration test",
    r"\bsoc analyst",
    r"\bqa (engineer|analyst|automation|tester|lead)",
    r"quality assurance (engineer|analyst|tester)",
    r"test automation",
    r"automation (test|qa)",
    r"\bsdet\b",
    r"software tester",
    r"\b(ios|android|mobile) (engineer|developer)",
    r"embedded (software|systems|engineer)",
    r"firmware",
    r"\bdatabase\b",
    r"\bdba\b",
    r"\bsql\b",
    r"\bit (support|specialist|analyst|manager|administrator|technician|director|engineer)",
    r"help ?desk",
    r"desktop support",
    r"technical support (engineer|specialist|analyst)",
    r"computer (scientist|engineer|programmer)",
]

TECH_EXCLUDE = [
    r"\bsales\b",
    r"recruit",
    r"talent acquisition",
    r"marketing",
    r"business develop",
    r"account (executive|manager)",
    r"real estate",
    r"mechanical",
    r"\bcivil\b",
    r"structural",
    r"electrical",
    r"chemical",
    r"manufacturing",
    r"construction",
    r"\bhvac\b",
    r"nurse",
    r"clinical",
    r"physician",
    r"data entry",
    r"\bai (trainer|tutor|writing|writer|content)",
]


def _compile_any(patterns: list[str]) -> re.Pattern:
    # Non-capturing groups only: pandas warns about capture groups in str.contains.
    joined = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(re.sub(r"\((?!\?)", "(?:", joined), re.IGNORECASE)


_INCLUDE_RE = _compile_any(TECH_INCLUDE)
_EXCLUDE_RE = _compile_any(TECH_EXCLUDE)


def is_tech_title(titles: pd.Series) -> pd.Series:
    titles = titles.fillna("")
    return titles.str.contains(_INCLUDE_RE) & ~titles.str.contains(_EXCLUDE_RE)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Remove HTML tags/entities and URLs, and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Salary
# ---------------------------------------------------------------------------

# Hours in a full-time year: 40 h x 52 weeks.
PAY_PERIOD_TO_YEARLY = {
    "YEARLY": 1,
    "MONTHLY": 12,
    "BIWEEKLY": 26,
    "WEEKLY": 52,
    "HOURLY": 2080,
}


def to_yearly(amount: pd.Series, pay_period: pd.Series) -> pd.Series:
    """Convert salary amounts to yearly. Unknown pay periods become NaN."""
    factor = pay_period.str.upper().map(PAY_PERIOD_TO_YEARLY)
    return amount.astype("float64") * factor.astype("float64")


def iqr_bounds(values: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def add_yearly_salary(df: pd.DataFrame, k: float = 1.5) -> tuple[pd.DataFrame, tuple]:
    """Add salary_min/max/yearly columns (USD per year) and blank out IQR outliers.

    salary_yearly is the median salary when given, otherwise the midpoint of
    min and max. Outliers are set to NaN rather than dropped, because the
    posting text is still useful for the NLP models.
    """
    df = df.copy()
    usd = df["currency"].fillna("USD").eq("USD")
    for col in ["min_salary", "med_salary", "max_salary"]:
        df[col] = df[col].where(usd)

    df["salary_min"] = to_yearly(df["min_salary"], df["pay_period"])
    df["salary_max"] = to_yearly(df["max_salary"], df["pay_period"])
    med = to_yearly(df["med_salary"], df["pay_period"])
    df["salary_yearly"] = med.fillna((df["salary_min"] + df["salary_max"]) / 2)

    low, high = iqr_bounds(df["salary_yearly"].dropna(), k)
    outlier = ~df["salary_yearly"].between(low, high)
    df.loc[outlier, ["salary_min", "salary_max", "salary_yearly"]] = np.nan
    return df, (low, high)


# ---------------------------------------------------------------------------
# Location and work type
# ---------------------------------------------------------------------------

US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}  # fmt: skip
STATE_NAME_TO_ABBR = {name.lower(): abbr for abbr, name in US_STATES.items()}
_AREA_SUFFIX_RE = re.compile(r"\s+(metropolitan area|area)$", re.IGNORECASE)


def _state_from_name(name: str) -> str | None:
    name = _AREA_SUFFIX_RE.sub("", name.strip()).lower()
    return STATE_NAME_TO_ABBR.get(name)


def parse_locations(locations: pd.Series) -> pd.DataFrame:
    """Split raw LinkedIn locations into city, state and a location level.

    Handles "City, ST", "City, State, United States", "State, United States",
    metro areas like "Greater Seattle Area" (matched to a known city), and
    the bare "United States".
    """
    out = pd.DataFrame(
        index=locations.index, columns=["city", "state", "location_level"]
    )
    unresolved = []

    for idx, loc in locations.fillna("").items():
        parts = [p.strip() for p in loc.split(",")]
        city = state = level = None
        if loc in ("", "United States"):
            level = "country"
        elif len(parts) == 2 and parts[1].upper() in US_STATES:
            city, state, level = parts[0], parts[1].upper(), "city"
        elif len(parts) == 2 and parts[1] == "United States":
            state, level = _state_from_name(parts[0]), "state"
        elif len(parts) == 2 and _state_from_name(parts[1]):
            state, level = _state_from_name(parts[1]), "state"
        elif len(parts) == 3 and parts[2] == "United States":
            city, state, level = parts[0], _state_from_name(parts[1]), "city"
        elif len(parts) == 1 and _state_from_name(parts[0]):
            state, level = _state_from_name(parts[0]), "state"
        else:
            level = "other"
            unresolved.append(idx)
        out.loc[idx] = [city, state, level]

    # Metro areas: find the most common state for any known city in the name,
    # e.g. "Greater Seattle Area" -> Seattle -> WA. The first city mentioned wins
    # ("Dallas-Fort Worth" -> Dallas). Anything unmatched (e.g. non-US) stays "other".
    known = out.dropna(subset=["city", "state"])
    city_state = known.groupby("city")["state"].agg(lambda s: s.mode().iat[0])
    city_counts = known["city"].value_counts()
    cities = city_counts[city_counts >= 5].index
    for idx in unresolved:
        loc = locations[idx]
        matches = [
            (m.start(), -len(c), c)
            for c in cities
            if (m := re.search(rf"\b{re.escape(c)}\b", loc))
        ]
        if matches:
            match = min(matches)[2]
            out.loc[idx] = [match, city_state[match], "metro"]
    return out


_HYBRID_RE = re.compile(
    r"\bhybrid\b(?!\s+(?:cloud|infrastructure|it\b|app|mobile|architecture|solution|environment))",
    re.IGNORECASE,
)
_REMOTE_TITLE_RE = re.compile(r"\bremote\b", re.IGNORECASE)


def work_type(
    remote_allowed: pd.Series, title: pd.Series, description: pd.Series
) -> pd.Series:
    """Classify each posting as remote / hybrid / on-site.

    remote: the remote_allowed flag is set, or the title says remote.
    hybrid: the title or description mentions hybrid work (not "hybrid cloud").
    on-site: everything else. The dataset has no explicit on-site flag, so
    this also includes postings that simply don't say.
    """
    text = title.fillna("") + " " + description.fillna("")
    remote = remote_allowed.eq(1) | title.fillna("").str.contains(_REMOTE_TITLE_RE)
    hybrid = text.str.contains(_HYBRID_RE)
    return pd.Series(
        np.select([remote & ~hybrid, hybrid], ["remote", "hybrid"], default="on-site"),
        index=title.index,
    )


# ---------------------------------------------------------------------------
# Seniority label
# ---------------------------------------------------------------------------

SENIORITY_MAP = {
    "Internship": "Entry",
    "Entry level": "Entry",
    "Associate": "Mid",
    "Mid-Senior level": "Mid",
    "Director": "Senior",
    "Executive": "Senior",
}


def map_seniority(levels: pd.Series, mapping: dict | None = None) -> pd.Series:
    """Map LinkedIn experience levels to Entry / Mid / Senior (NaN if unknown)."""
    return levels.map(mapping or SENIORITY_MAP)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _join_names(
    links: pd.DataFrame, names: pd.DataFrame, key: str, name: str
) -> pd.Series:
    merged = links.merge(names, on=key, how="left").dropna(subset=[name])
    return merged.groupby("job_id")[name].agg(lambda s: "|".join(sorted(set(s))))


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load postings and attach skill categories and industries as '|'-joined strings."""
    postings = pd.read_csv(
        raw_dir / "postings.csv", usecols=POSTING_COLUMNS, low_memory=False
    )
    skills = _join_names(
        pd.read_csv(raw_dir / "jobs" / "job_skills.csv"),
        pd.read_csv(raw_dir / "mappings" / "skills.csv"),
        "skill_abr",
        "skill_name",
    )
    industries = _join_names(
        pd.read_csv(raw_dir / "jobs" / "job_industries.csv"),
        pd.read_csv(raw_dir / "mappings" / "industries.csv"),
        "industry_id",
        "industry_name",
    )
    postings["skill_categories"] = postings["job_id"].map(skills)
    postings["industries"] = postings["job_id"].map(industries)
    return postings


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop postings with the same title, company and cleaned description."""
    key = pd.DataFrame(
        {
            "title": df["title"].str.lower().str.strip(),
            "company": df["company_name"].fillna("").str.lower().str.strip(),
            "description": df["description_clean"].str.lower(),
        }
    )
    return df.loc[~key.duplicated()]


def build_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run all cleaning steps on the joined raw table. Returns (df, stats)."""
    stats = {"raw_rows": len(raw)}

    df = raw.loc[is_tech_title(raw["title"])].copy()
    stats["tech_rows"] = len(df)

    df["title"] = df["title"].map(clean_text)
    df["description_clean"] = df["description"].map(clean_text)
    df = df.loc[df["description_clean"].str.len() > 0]
    df = deduplicate(df)
    stats["deduped_rows"] = len(df)

    df, stats["salary_bounds"] = add_yearly_salary(df)
    stats["salary_rows"] = int(df["salary_yearly"].notna().sum())

    df[["city", "state", "location_level"]] = parse_locations(df["location"])
    df["work_type"] = work_type(
        df["remote_allowed"], df["title"], df["description_clean"]
    )
    df["seniority"] = map_seniority(df["formatted_experience_level"])
    df["listed_date"] = pd.to_datetime(df["listed_time"], unit="ms")
    df["original_listed_date"] = pd.to_datetime(df["original_listed_time"], unit="ms")

    df = df.rename(
        columns={
            "description": "description_raw",
            "formatted_experience_level": "experience_level",
            "formatted_work_type": "employment_type",
        }
    )
    columns = [
        "job_id", "title", "description_raw", "description_clean",
        "company_id", "company_name", "industries", "skill_categories",
        "experience_level", "seniority", "employment_type", "work_type",
        "location", "city", "state", "location_level",
        "pay_period", "salary_min", "salary_max", "salary_yearly",
        "listed_date", "original_listed_date", "views", "applies",
    ]  # fmt: skip
    df = df[columns].reset_index(drop=True)

    for col in ["experience_level", "seniority", "employment_type", "work_type", "state",
                "location_level", "pay_period"]:  # fmt: skip
        df[col] = df[col].astype("category")
    for col in ["salary_min", "salary_max", "salary_yearly", "views", "applies"]:
        df[col] = df[col].astype("float32")
    df["company_id"] = df["company_id"].astype("Int64")
    return df, stats


def main() -> None:
    df, stats = build_dataset(load_raw())
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"Saved {len(df):,} rows to {PROCESSED_PATH.relative_to(PROJECT_ROOT)}")
    print(df["seniority"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
