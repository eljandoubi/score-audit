import numpy as np
import pytest

from scoreaudit import irt


def simulate(n=5000, a=(0.8, 1.2, 1.6, 2.0, 1.0), seed=0, n_categories=5):
    """Answers of n writers to items with the given discriminations, and the true parameters."""
    rng = np.random.default_rng(seed)
    a = np.array(a)
    base = np.linspace(-1.5, 1.5, n_categories - 1)
    b = base[None, :] + rng.normal(0, 0.2, (len(a), n_categories - 1))
    b = np.sort(b, axis=1)
    theta = rng.normal(size=n)
    reach = 1 / (1 + np.exp(-a[None, :, None] * (theta[:, None, None] - b[None, :, :])))
    draws = rng.uniform(size=(n, len(a), 1))
    responses = (draws < reach).sum(axis=2)  # nested thresholds, so this is the graded model
    return responses, theta, a, b


def test_recode_merges_the_sparse_ends():
    scores = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0]
    assert irt.recode(scores).tolist() == [0, 0, 0, 1, 2, 4, 5, 5]


def test_fit_recovers_discrimination_and_thresholds():
    responses, _, a, b = simulate()
    model = irt.fit(responses)
    assert np.corrcoef(model.a, a)[0, 1] > 0.97
    assert model.a == pytest.approx(a, rel=0.15)
    assert model.b == pytest.approx(b, abs=0.2)


def test_fitted_thresholds_are_increasing():
    responses, *_ = simulate(n=1500)
    model = irt.fit(responses)
    assert (np.diff(model.b, axis=1) > 0).all()
    assert (model.a > 0).all()


def test_proficiency_estimates_follow_the_true_proficiency():
    responses, theta, *_ = simulate()
    model = irt.fit(responses)
    estimate, sd = irt.proficiency(model, responses)
    corr = np.corrcoef(estimate, theta)[0, 1]
    assert corr > 0.83
    assert corr**2 == pytest.approx(irt.reliability(estimate, sd), abs=0.03)
    assert (sd > 0).all() and sd.max() < 1
    assert estimate.mean() == pytest.approx(0, abs=0.05)


def test_more_items_make_the_estimates_more_reliable():
    few, *_ = simulate(a=(1.0,) * 3, n=3000)
    many, *_ = simulate(a=(1.0,) * 12, n=3000)
    rel = []
    for table in (few, many):
        model = irt.fit(table)
        rel.append(irt.reliability(*irt.proficiency(model, table)))
    assert 0 < rel[0] < rel[1] < 1


def test_a_better_model_has_a_higher_likelihood_than_a_flat_one():
    responses, *_ = simulate(n=2000)
    model = irt.fit(responses)
    flat = -2000 * 5 * np.log(5)
    assert model.loglik > flat


def test_cronbach_alpha_of_identical_items_is_one_and_of_noise_is_low():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 1))
    assert irt.cronbach_alpha(np.hstack([x, x, x])) == pytest.approx(1.0)
    assert irt.cronbach_alpha(rng.normal(size=(500, 6))) < 0.2
