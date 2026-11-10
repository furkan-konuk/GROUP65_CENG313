"""Baseline models for the pairwise task "which of the two texts is real?".

Every model follows the same tiny interface:
    model.fit(df)              # df has text_1, text_2, real_text_id
    model.predict(df) -> array of 1 / 2 (the predicted real_text_id)

Key idea for the ML baselines: we build features as   f(text_1) - f(text_2).
If we swap the two texts, the features flip sign and the label flips too.
We use this symmetry to double the (very small) training set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import get_target
from .utils import features_frame, latin_char_ratio


def _to_real_id(prob_first_is_real: np.ndarray) -> np.ndarray:
    return np.where(prob_first_is_real >= 0.5, 1, 2)


# ----------------------------------------------------------------------------
# 1) Trivial baselines
# ----------------------------------------------------------------------------
class RandomBaseline:
    """Coin flip. Any real model must beat ~50%."""

    name = "Random guess"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, df):
        return self

    def predict(self, df):
        rng = np.random.default_rng(self.seed)
        return rng.integers(1, 3, size=len(df))


class MajorityBaseline:
    """Always predict the most frequent label seen in training."""

    name = "Majority class"

    def fit(self, df):
        self.label_ = int(df["real_text_id"].mode()[0])
        return self

    def predict(self, df):
        return np.full(len(df), self.label_)


# ----------------------------------------------------------------------------
# 2) Rule-based baseline (idea from the organizers' public notebook)
# ----------------------------------------------------------------------------
class LatinRatioBaseline:
    """The text with the higher share of Latin letters is predicted as real."""

    name = "Rule: higher Latin-char ratio"

    def fit(self, df):
        return self

    def predict(self, df):
        r1 = df["text_1"].map(latin_char_ratio).values
        r2 = df["text_2"].map(latin_char_ratio).values
        return np.where(r1 >= r2, 1, 2)


# ----------------------------------------------------------------------------
# 3) Hand-crafted features + Logistic Regression
# ----------------------------------------------------------------------------
class FeatureDiffLR:
    """Logistic Regression on differences of simple text statistics."""

    name = "LogReg on handcrafted feature differences"

    def __init__(self, C: float = 1.0):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))

    @staticmethod
    def _diff(df) -> np.ndarray:
        f1 = features_frame(df["text_1"])
        f2 = features_frame(df["text_2"])
        return (f1 - f2).values

    def fit(self, df):
        X = self._diff(df)
        y = get_target(df)
        # symmetry augmentation: swapped pair -> negated features, flipped label
        self.model.fit(np.vstack([X, -X]), np.concatenate([y, 1 - y]))
        return self

    def predict_proba_first(self, df):
        return self.model.predict_proba(self._diff(df))[:, 1]

    def predict(self, df):
        return _to_real_id(self.predict_proba_first(df))

    def feature_importance(self) -> pd.Series:
        cols = features_frame(["x"]).columns
        coefs = self.model[-1].coef_[0]
        return pd.Series(coefs, index=cols).sort_values(key=np.abs, ascending=False)


# ----------------------------------------------------------------------------
# 4) TF-IDF (character n-grams) + Logistic Regression
# ----------------------------------------------------------------------------
class TfidfDiffLR:
    """TF-IDF vectors of both texts, the model sees vec(text_1) - vec(text_2)."""

    name = "LogReg on TF-IDF char n-gram differences"

    def __init__(self, C: float = 5.0, analyzer: str = "char_wb", ngram_range=(2, 4)):
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer, ngram_range=ngram_range, min_df=2, sublinear_tf=True
        )
        self.clf = LogisticRegression(C=C, max_iter=5000)

    def _diff(self, df):
        return self.vectorizer.transform(df["text_1"]) - self.vectorizer.transform(df["text_2"])

    def fit(self, df):
        corpus = pd.concat([df["text_1"], df["text_2"]])
        self.vectorizer.fit(corpus)
        X = self._diff(df)
        y = get_target(df)
        from scipy.sparse import vstack

        self.clf.fit(vstack([X, -X]), np.concatenate([y, 1 - y]))
        return self

    def predict_proba_first(self, df):
        return self.clf.predict_proba(self._diff(df))[:, 1]

    def predict(self, df):
        return _to_real_id(self.predict_proba_first(df))


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def cross_validate(model_factory, df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> dict:
    """Stratified K-fold accuracy (the competition metric is plain accuracy)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    oof = np.zeros(len(df), dtype=int)
    for tr_idx, va_idx in skf.split(df, df["real_text_id"]):
        model = model_factory().fit(df.iloc[tr_idx])
        pred = model.predict(df.iloc[va_idx])
        oof[va_idx] = pred
        scores.append(np.mean(pred == df["real_text_id"].values[va_idx]))
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "folds": scores, "oof": oof}


ALL_BASELINES = {
    "random": RandomBaseline,
    "majority": MajorityBaseline,
    "latin_rule": LatinRatioBaseline,
    "feature_lr": FeatureDiffLR,
    "tfidf_lr": TfidfDiffLR,
}
