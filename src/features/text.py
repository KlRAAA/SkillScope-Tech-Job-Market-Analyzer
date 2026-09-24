"""Text features: years of experience required and seniority keyword flags.

TF-IDF and the sklearn pipeline are added in Phase 4.
"""

import re

import numpy as np
import pandas as pd

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
