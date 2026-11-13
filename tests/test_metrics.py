import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from scoreaudit.metrics import (
    bland_altman,
    bootstrap,
    exact,
    icc,
    levels,
    mean_error,
    paired_difference,
    qwk,
    smd,
    within,
)

# Table 2 of Shrout and Fleiss (1979): 6 essays rated by 4 raters
SHROUT_FLEISS = np.array(
    [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]
)


def noisy_pair(n=400, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 9, n)
    b = np.clip(a + rng.integers(-2, 3, n), 0, 8)
    return 1 + a / 2, 1 + b / 2


def test_levels_count_half_points_from_one():
    assert levels([1.0, 1.5, 3.0, 5.0]).tolist() == [0, 1, 4, 8]


def test_qwk_matches_scikit_learn():
    a, b = noisy_pair()
    reference = cohen_kappa_score(levels(a), levels(b), weights="quadratic", labels=list(range(9)))
    assert qwk(a, b) == pytest.approx(reference)


def test_qwk_is_one_for_identical_scores_and_symmetric():
    a, b = noisy_pair()
    assert qwk(a, a) == 1.0
    assert qwk(a, b) == pytest.approx(qwk(b, a))


def test_qwk_of_a_constant_scorer_is_zero():
    a, _ = noisy_pair()
    assert abs(qwk(a, np.full_like(a, 3.0))) < 1e-9


def test_qwk_is_undefined_when_both_are_constant():
    assert np.isnan(qwk([3.0, 3.0], [3.0, 3.0]))


def test_qwk_punishes_a_far_miss_more_than_a_near_miss():
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 3.0])
    near = truth + np.array([0.5, -0.5, 0.5, -0.5, 0.0, 0.0])
    far = truth.copy()
    far[0] = 5.0
    assert qwk(truth, near) > qwk(truth, far)


def test_qwk_reads_a_continuous_score_as_the_nearest_half_point():
    assert qwk([1.0, 3.0, 5.0, 2.0], [1.2, 3.1, 4.9, 2.2]) == 1.0


def test_qwk_refuses_scores_off_the_scale():
    with pytest.raises(ValueError, match="outside the scale"):
        qwk([1.0, 5.5], [1.0, 3.0])
    with pytest.raises(ValueError, match="outside the scale"):
        qwk([0.5, 3.0], [1.0, 3.0])


def test_within_and_exact():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, 2.5, 4.0, 4.0])
    assert exact(a, b) == 0.5
    assert within(a, b, 0.5) == 0.75
    assert within(a, b, 1.0) == 1.0


def test_icc_reproduces_the_shrout_fleiss_example():
    expected = {"1,1": 0.17, "2,1": 0.29, "3,1": 0.71, "1,k": 0.44, "2,k": 0.62, "3,k": 0.91}
    for form, value in expected.items():
        assert icc(SHROUT_FLEISS, form) == pytest.approx(value, abs=0.006)


def test_icc_of_identical_raters_is_one():
    x = np.random.default_rng(0).normal(size=(30, 1))
    both = np.hstack([x, x])
    assert all(icc(both, f) == pytest.approx(1.0) for f in ("1,1", "2,1", "3,1"))


def test_a_constant_shift_hurts_absolute_agreement_but_not_consistency():
    x = np.random.default_rng(1).normal(size=(50, 1))
    shifted = np.hstack([x, x + 2])
    assert icc(shifted, "3,1") == pytest.approx(1.0)
    assert icc(shifted, "2,1") < 0.5


def test_unknown_icc_form():
    with pytest.raises(ValueError, match="unknown form"):
        icc(SHROUT_FLEISS, "4,1")


def test_bland_altman_finds_a_constant_bias_without_a_slope():
    rng = np.random.default_rng(0)
    truth = rng.uniform(1, 5, 500)
    result = bland_altman(truth + 0.3 + rng.normal(0, 0.1, 500), truth)
    assert result.bias == pytest.approx(0.3, abs=0.02)
    assert result.lower < 0.3 < result.upper
    assert abs(result.slope) < 0.05
    assert result.slope_p > 0.01


def test_bland_altman_finds_a_proportional_bias():
    truth = np.linspace(1, 5, 200)
    result = bland_altman(1.2 * truth, truth)
    assert result.slope == pytest.approx(0.2 / 1.1, rel=1e-6)
    assert result.slope_p < 1e-6


def test_mean_error_has_the_sign_of_generosity():
    assert mean_error([4.0, 4.0], [3.0, 3.0]) == 1.0


def test_smd_of_a_shift_of_one_standard_deviation():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 5000), rng.normal(0, 1, 5000)
    assert smd(a, b) == pytest.approx(1.0, abs=0.05)
    assert smd(a, a) == 0.0


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    a, b = noisy_pair()
    first = bootstrap(qwk, a, b, n=200, seed=3)
    assert first == bootstrap(qwk, a, b, n=200, seed=3)
    estimate, low, high = first
    assert estimate == qwk(a, b)
    assert low < estimate < high


def test_bootstrap_interval_narrows_with_more_essays():
    small = bootstrap(qwk, *noisy_pair(100), n=300)
    large = bootstrap(qwk, *noisy_pair(1600), n=300)
    assert (large[2] - large[1]) < (small[2] - small[1])


def test_paired_difference_of_a_scorer_with_itself_is_exactly_zero():
    a, b = noisy_pair()
    assert paired_difference(qwk, a, b, b, n=50) == (0.0, 0.0, 0.0)


def test_paired_difference_interval_is_narrower_than_the_two_separate_ones():
    truth, good = noisy_pair(300)
    rng = np.random.default_rng(5)
    other = np.clip(good + rng.choice([-0.5, 0, 0.5], len(good), p=[0.1, 0.8, 0.1]), 1, 5)
    _, low, high = paired_difference(qwk, truth, good, other, n=300)
    _, low_a, high_a = bootstrap(qwk, truth, good, n=300)
    assert (high - low) < (high_a - low_a)
