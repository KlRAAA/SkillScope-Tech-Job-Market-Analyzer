"""Custom estimators, kept in their own module so saved models can be loaded anywhere.

(A class defined in a script run with `python -m` is pickled as `__main__.X`,
which nothing else can import.)
"""

from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


class BalancedXGBClassifier(XGBClassifier):
    """XGBoost with class-balanced sample weights (it has no class_weight option)."""

    def fit(self, X, y, **kwargs):
        kwargs.setdefault("sample_weight", compute_sample_weight("balanced", y))
        return super().fit(X, y, **kwargs)
