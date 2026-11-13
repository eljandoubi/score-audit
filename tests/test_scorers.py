import numpy as np
import pytest

from scoreaudit.scorers import (
    SURFACE_NAMES,
    FeatureRidge,
    MeanScorer,
    SurfaceGBM,
    TfidfRidge,
    surface_features,
)

GOOD = "clear precise argument evidence reasoning thoughtful structure fluent".split()
WEAK = "bad stuff thing very much like is good nice maybe".split()


def make_essays(n=400, seed=0):
    """Essays drawn from two word lists. The score is the share of words from the first one."""
    rng = np.random.default_rng(seed)
    texts, scores = [], []
    for _ in range(n):
        share = rng.uniform(0, 1)
        picks = [rng.choice(GOOD) if rng.random() < share else rng.choice(WEAK) for _ in range(40)]
        texts.append(" ".join(picks) + ".")
        scores.append(1 + 4 * share)
    return texts, np.array(scores)


def split(texts, scores, k=300):
    return texts[:k], scores[:k], texts[k:], scores[k:]


def test_the_mean_scorer_gives_the_training_mean_and_respects_weights():
    scorer = MeanScorer().fit(["a", "b", "c"], [1.0, 2.0, 6.0])
    assert scorer.predict(["x", "y"]).tolist() == [3.0, 3.0]
    weighted = MeanScorer().fit(["a", "b", "c"], [1.0, 2.0, 6.0], weights=[1, 1, 0])
    assert weighted.predict(["x"])[0] == 1.5


def test_tfidf_ridge_learns_which_words_go_with_a_high_score():
    texts, scores = make_essays()
    tr_t, tr_s, te_t, te_s = split(texts, scores)
    pred = TfidfRidge().fit(tr_t, tr_s).predict(te_t)
    assert np.corrcoef(pred, te_s)[0, 1] > 0.9


def test_surface_gbm_picks_up_length():
    rng = np.random.default_rng(1)
    texts, scores = [], []
    for _ in range(400):
        n = int(rng.integers(10, 200))
        texts.append(("The students like the school. " * (n // 5 + 1)).strip())
        scores.append(1 + 4 * min(n, 200) / 200)
    pred = SurfaceGBM().fit(texts[:300], scores[:300]).predict(texts[300:])
    assert np.corrcoef(pred, scores[300:])[0, 1] > 0.9


def test_surface_features_have_one_column_per_name_and_one_row_per_essay():
    feats = surface_features(["One two. Three four five!", "i am here"], known={"one", "two"})
    assert feats.shape == (2, len(SURFACE_NAMES))
    assert np.isfinite(feats).all()


def test_unknown_words_raise_the_unknown_share_and_a_lowercase_i_is_counted():
    col = SURFACE_NAMES.index("unknown_word_share")
    known = {"the", "cat", "sat"}
    feats = surface_features(["the cat sat", "the dog stood"], known)
    assert feats[0, col] == 0.0
    assert feats[1, col] == pytest.approx(2 / 3)
    assert surface_features(["i think i can"], known)[0, SURFACE_NAMES.index("lowercase_i")] == 0.5


def test_an_empty_essay_does_not_break_the_features():
    assert np.isfinite(surface_features([""], known=set())).all()


def test_feature_ridge_fits_a_linear_signal_and_takes_weights():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(500, 20))
    y = x @ rng.normal(size=20) + rng.normal(0, 0.1, 500)
    scorer = FeatureRidge().fit(x[:400], y[:400])
    assert np.corrcoef(scorer.predict(x[400:]), y[400:])[0, 1] > 0.97
    weights = np.r_[np.ones(200), np.zeros(200)]
    only_first = FeatureRidge().fit(x[:200], y[:200])
    weighted = FeatureRidge().fit(x[:400], y[:400], weights=weights)
    assert np.corrcoef(weighted.predict(x[400:]), only_first.predict(x[400:]))[0, 1] > 0.99


def test_every_text_scorer_returns_one_score_per_text():
    texts, scores = make_essays(n=60)
    for scorer in (MeanScorer(), TfidfRidge(), SurfaceGBM()):
        assert scorer.fit(texts, scores).predict(texts[:7]).shape == (7,)
