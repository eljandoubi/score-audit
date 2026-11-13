"""Scorers that learn from rated essays.

A text scorer is fitted on essays and their scores and predicts a score for new essays. A feature
scorer does the same from a matrix of features, for example pooled hidden states of a language
model. Every `fit` takes optional sample weights, which the bias mitigation uses.
"""

import re
from collections import Counter
from collections.abc import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler

WORD = re.compile(r"[A-Za-z']+")
SENTENCE = re.compile(r"(?<=[.!?])\s+")

SURFACE_NAMES = [
    "words", "sentences", "paragraphs", "words_per_sentence", "sentence_length_sd",
    "mean_word_length", "long_word_share", "type_token_ratio", "root_ttr", "comma_rate",
    "question_rate", "capital_start_share", "lowercase_i", "unknown_word_share",
]  # fmt: skip


class MeanScorer:
    """Gives every essay the mean training score. Its agreement with the raters is the floor."""

    name = "mean"

    def fit(self, texts: Sequence[str], scores, weights=None):
        self.mean_ = float(np.average(scores, weights=weights))
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return np.full(len(texts), self.mean_)


class TfidfRidge:
    """Word and character n-grams into a ridge regression, the classical baseline."""

    name = "tfidf-ridge"

    def fit(self, texts: Sequence[str], scores, weights=None):
        self.vectorizer = make_union(
            TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=60_000),
            TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 5), min_df=5, sublinear_tf=True,
                max_features=100_000,
            ),
        )  # fmt: skip
        matrix = self.vectorizer.fit_transform(texts)
        self.model = RidgeCV(alphas=np.logspace(-1, 3, 9)).fit(
            matrix, scores, sample_weight=weights
        )
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.predict(self.vectorizer.transform(texts))


def surface_features(texts: Sequence[str], known: set[str]) -> np.ndarray:
    """Counts and rates a rater would not need to read the essay to see, one row per essay.

    `known` is the vocabulary of the training essays: a word outside it is often a misspelling.
    """
    rows = []
    for text in texts:
        words = WORD.findall(text)
        low = [w.lower() for w in words]
        sentences = [s for s in SENTENCE.split(text.strip()) if s]
        lengths = np.array([len(WORD.findall(s)) for s in sentences] or [0], dtype=float)
        n = max(len(words), 1)
        types = len(set(low))
        rows.append(
            [
                len(words),
                len(sentences),
                len([p for p in re.split(r"\n+", text) if p.strip()]),
                lengths.mean(),
                lengths.std(),
                np.mean([len(w) for w in words]) if words else 0.0,
                sum(len(w) >= 7 for w in words) / n,
                types / n,
                types / np.sqrt(n),
                text.count(",") / n,
                text.count("?") / n,
                np.mean([s[0].isupper() for s in sentences]) if sentences else 0.0,
                sum(w == "i" for w in words) / n,
                sum(w not in known for w in low) / n,
            ]
        )
    return np.array(rows, dtype=float)


class SurfaceGBM:
    """Gradient boosting on length, vocabulary and punctuation features."""

    name = "surface-gbm"

    def fit(self, texts: Sequence[str], scores, weights=None):
        counts = Counter(w.lower() for t in texts for w in WORD.findall(t))
        self.known = {w for w, c in counts.items() if c >= 3}
        self.model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0
        ).fit(surface_features(texts, self.known), scores, sample_weight=weights)
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.predict(surface_features(texts, self.known))


class FeatureRidge:
    """Ridge regression on a matrix of features, with the penalty chosen by cross-validation."""

    def __init__(self, name: str = "feature-ridge"):
        self.name = name

    def fit(self, features: np.ndarray, scores, weights=None):
        self.model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 11)))
        self.model.fit(features, scores, ridgecv__sample_weight=weights)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)
