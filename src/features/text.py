"""Text features: cleaning for models, TF-IDF, years of experience, keyword flags."""

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.clean import clean_text

# Tokens keep '+', '#' and inner dots so "c++", "c#" and "node.js" survive.
TOKEN_PATTERN = r"(?u)\b[a-z][a-z0-9+#]*(?:\.[a-z]+)?"


def remove_company(text: str, company: str | None) -> str:
    """Remove the posting's own company name from its text.

    Some companies label almost all their postings one way (e.g. a staffing
    agency posting mostly Entry roles), so a model could learn the company
    instead of the seniority. Case-insensitive, whole-phrase match.
    """
    if not isinstance(text, str):
        return ""
    if not isinstance(company, str) or len(company.strip()) < 2:
        return text
    return re.sub(
        rf"(?<!\w){re.escape(company.strip())}(?!\w)", " ", text, flags=re.IGNORECASE
    )


def model_text(title: str | None, description: str | None, company: str | None = None,
               include_title: bool = True) -> str:  # fmt: skip
    """The text the models see: [title +] cleaned description, company name removed.

    Case is kept because some skill patterns are case-sensitive ("Go", "React").
    TF-IDF lowercases on its own.
    """
    parts = [title, description] if include_title else [description]
    text = " ".join(clean_text(p) for p in parts if isinstance(p, str))
    return remove_company(text, company)


# HR boilerplate and generic business words. Without these, clustering groups
# postings by their equal-opportunity/benefits templates instead of by role.
BOILERPLATE_STOP_WORDS = """
status employment employer employers employee employees benefits benefit disability disabilities
gender identity expression equal opportunity opportunities protected veteran veterans race religion
sex sexual orientation national origin color age marital pregnancy genetic information applicants
applicant accommodation accommodations eeo affirmative action discrimination law laws regard
regardless qualified consideration pay salary compensation range bonus insurance dental vision
medical 401k pto paid time off holidays wellness perks company companies culture inclusive diversity
inclusion apply application hiring hire recruiting recruiter candidate candidates position positions
job jobs role roles join team teams w2 c2c contract duration month months location onsite remote
hybrid description title requirements required preferred qualifications responsibilities including
ability skills experience years work working strong excellent knowledge understanding environment
ensure provide new based world people business customer customers client clients management project
projects program technology technologies process processes services service solutions solution
product products organization support related level industry looking
""".split()  # noqa: SIM905 (a 140-word list literal is harder to read)


def cluster_text(
    title: str | None, description: str | None, company: str | None = None
) -> str:
    """Text for role clustering: the title repeated 3x (it names the role) + description."""
    title = clean_text(title)
    return f"{title} {title} {title} " + model_text(
        None, description, company, include_title=False
    )


def make_tfidf(max_features: int = 20_000, min_df: int = 5) -> TfidfVectorizer:
    """TF-IDF over 1-2 word phrases.

    sublinear_tf dampens repeated words in long descriptions, and max_df drops
    boilerplate that appears in most postings (EEO statements, benefits).
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.8,
        max_features=max_features,
        stop_words="english",
        sublinear_tf=True,
        token_pattern=TOKEN_PATTERN,
        dtype=np.float32,
    )


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15,
}  # fmt: skip
_NUM = r"(\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")"

# "3+ years", "5-7 years", "3 to 5 yrs", "two (2) years", optionally
# followed within a few words by "experience". Also "minimum of 3 years".
_YEARS_RE = re.compile(
    rf"{_NUM}\s*(?:\(\d{{1,2}}\)\s*)?\+?\s*(?:(?:-|–|to)\s*{_NUM}\s*\+?\s*)?"
    rf"(?:years?|yrs?\.?)(?![a-z])(?=[^.;\n]{{0,60}}experience)",
    re.IGNORECASE,
)
MAX_REASONABLE_YEARS = 20


def _to_int(token: str) -> int:
    token = token.lower()
    return _NUMBER_WORDS[token] if token in _NUMBER_WORDS else int(token)


def extract_years(text: str | None) -> float:
    """Minimum years of experience asked for, or NaN if none is stated.

    Takes the lower bound of each "N years ... experience" mention and returns
    the smallest one, which is usually the hard requirement ("3+ years
    required, 5+ preferred" -> 3). Values above 20 are ignored as noise
    (e.g. "our 25 years of experience in the industry").
    """
    if not isinstance(text, str):
        return np.nan
    values = [_to_int(m.group(1)) for m in _YEARS_RE.finditer(text)]
    values = [v for v in values if 0 < v <= MAX_REASONABLE_YEARS]
    return float(min(values)) if values else np.nan


KEYWORD_FLAGS = {
    "kw_lead": r"\blead(?:s|ing)?\b",
    "kw_mentor": r"\bmentor(?:s|ing|ship)?\b",
    "kw_architect": r"\barchitect(?:s|ure|ing)?\b",
    "kw_intern": r"\bintern(?:s|ship)?\b",
    "kw_graduate": r"\b(?:new )?grad(?:uate)?s?\b",
    "kw_principal": r"\bprincipal\b",
}


def keyword_flags(texts: pd.Series) -> pd.DataFrame:
    """0/1 column per seniority keyword."""
    return pd.DataFrame(
        {
            name: texts.fillna("").str.contains(pattern, case=False).astype(np.uint8)
            for name, pattern in KEYWORD_FLAGS.items()
        },
        index=texts.index,
    )
