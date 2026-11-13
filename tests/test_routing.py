import numpy as np
import pytest

from scoreaudit.metrics import qwk
from scoreaudit.routing import coverage_curve, ensemble_spread


def pool(n=600, seed=0):
    """A machine that is right on the easy essays and often wrong on the doubtful ones."""
    rng = np.random.default_rng(seed)
    truth = rng.choice(np.arange(1.0, 5.01, 0.5), n)
    doubt = rng.uniform(0, 1, n)
    wrong = rng.random(n) < doubt
    machine = np.clip(truth + wrong * rng.choice([-1.5, 1.5], n), 1, 5)
    reviewer = np.clip(truth + rng.choice([-0.5, 0, 0.5], n, p=[0.1, 0.8, 0.1]), 1, 5)
    return machine, reviewer, truth, doubt


def test_the_ends_of_the_curve_are_the_machine_alone_and_the_reviewer_alone():
    machine, reviewer, truth, doubt = pool()
    curve = coverage_curve(machine, reviewer, truth, doubt)
    assert curve["qwk_routed"].iloc[0] == pytest.approx(qwk(machine, truth))
    assert curve["qwk_routed"].iloc[-1] == pytest.approx(qwk(reviewer, truth))
    assert curve["qwk_random"].iloc[0] == curve["qwk_routed"].iloc[0]
    assert curve["n_routed"].tolist()[-1] == len(machine)


def test_sending_the_doubtful_essays_first_beats_sending_random_ones():
    machine, reviewer, truth, doubt = pool()
    curve = coverage_curve(machine, reviewer, truth, doubt, fractions=(0.2, 0.3))
    assert (curve["qwk_routed"] > curve["qwk_random"] + 0.03).all()


def test_an_uninformative_screen_is_no_better_than_random():
    machine, reviewer, truth, _ = pool()
    noise = np.random.default_rng(9).uniform(size=len(machine))
    curve = coverage_curve(machine, reviewer, truth, noise, fractions=(0.3,))
    assert curve["qwk_routed"].iloc[0] == pytest.approx(curve["qwk_random"].iloc[0], abs=0.05)


def test_agreement_rises_as_the_doubtful_essays_go_to_the_reviewer():
    machine, reviewer, truth, doubt = pool()
    curve = coverage_curve(machine, reviewer, truth, doubt, fractions=(0.0, 0.1, 0.2, 0.3, 0.5))
    assert curve["qwk_routed"].is_monotonic_increasing


def test_the_most_uncertain_essay_goes_first():
    machine = np.array([1.0, 1.0, 1.0])
    reviewer = np.array([5.0, 5.0, 5.0])
    reference = np.array([5.0, 3.0, 1.0])
    curve = coverage_curve(machine, reviewer, reference, [0.1, 0.9, 0.5], fractions=(1 / 3,))
    mixed = np.array([1.0, 5.0, 1.0])  # the essay with doubt 0.9 is the one that changed
    assert curve["qwk_routed"].iloc[0] == pytest.approx(qwk(mixed, reference))


def test_ensemble_spread_is_larger_where_the_training_data_say_less():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 5))
    y = x[:, 0] + rng.normal(0, 0.3, 300)
    near = rng.normal(size=(50, 5))
    far = 8 * rng.normal(size=(50, 5))
    spread = ensemble_spread(x, y, np.vstack([near, far]), alpha=1.0, n_models=30)
    assert spread[50:].mean() > 3 * spread[:50].mean()


def test_ensemble_spread_is_reproducible():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 4))
    y = x[:, 0]
    a = ensemble_spread(x, y, x[:10], alpha=1.0, n_models=5, seed=3)
    assert np.array_equal(a, ensemble_spread(x, y, x[:10], alpha=1.0, n_models=5, seed=3))
