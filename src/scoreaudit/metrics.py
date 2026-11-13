"""Agreement between two sets of scores, and the intervals around it."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

LOW, STEP, N_LEVELS = 1.0, 0.5, 9  # scores run from 1 to 5 in steps of half a point


def levels(scores, low: float = LOW, step: float = STEP) -> np.ndarray:
    """Whole-number level of each score, 0 for the lowest."""
    return np.rint((np.asarray(scores, dtype=float) - low) / step).astype(int)


def qwk(a, b, n_levels: int = N_LEVELS) -> float:
    """Quadratic weighted kappa between two ratings on the half-point scale.

    The scores are rounded to the nearest level first, so a continuous score is first read as the
    level a rater would have given. nan when both ratings are constant.
    """
    x, y = levels(a), levels(b)
    if x.min() < 0 or y.min() < 0 or max(x.max(), y.max()) >= n_levels:
        raise ValueError("scores outside the scale")
    observed = np.bincount(x * n_levels + y, minlength=n_levels**2).reshape(n_levels, n_levels)
    observed = observed / observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    grid = np.arange(n_levels)
    weights = (grid[:, None] - grid[None, :]) ** 2
    denominator = (weights * expected).sum()
    if denominator == 0:
        return float("nan")
    return float(1 - (weights * observed).sum() / denominator)


def within(a, b, tolerance: float) -> float:
    """Share of pairs that differ by at most `tolerance` points."""
    return float(
        (np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) <= tolerance + 1e-9).mean()
    )


def exact(a, b) -> float:
    """Share of pairs that fall on the same level."""
    return float((levels(a) == levels(b)).mean())


def pearson(a, b) -> float:
    return float(stats.pearsonr(a, b).statistic)


def spearman(a, b) -> float:
    return float(stats.spearmanr(a, b).statistic)


def mae(a, b) -> float:
    return float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).mean())


def rmse(a, b) -> float:
    return float(np.sqrt(((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2).mean()))


def mean_error(pred, truth) -> float:
    """Average of pred minus truth: positive when the scorer is more generous than the raters."""
    return float((np.asarray(pred, dtype=float) - np.asarray(truth, dtype=float)).mean())


def icc(ratings, form: str = "2,1") -> float:
    """Intraclass correlation of an (essays x raters) table, in the notation of Shrout and Fleiss.

    "1,1" and "1,k" treat the raters as interchangeable, "2,1" and "2,k" ask for the same scores
    (absolute agreement) from raters drawn at random, "3,1" and "3,k" ask only that the scores
    move together (consistency) among fixed raters. The "k" forms are for the mean of the raters.
    """
    x = np.asarray(ratings, dtype=float)
    n, k = x.shape
    grand = x.mean()
    ss_rows = k * ((x.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((x.mean(axis=0) - grand) ** 2).sum()
    ss_total = ((x - grand) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_error / ((n - 1) * (k - 1))
    msw = (ss_cols + ss_error) / (n * (k - 1))
    formulas = {
        "1,1": (msr - msw) / (msr + (k - 1) * msw),
        "2,1": (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n),
        "3,1": (msr - mse) / (msr + (k - 1) * mse),
        "1,k": (msr - msw) / msr,
        "2,k": (msr - mse) / (msr + (msc - mse) / n),
        "3,k": (msr - mse) / msr,
    }
    if form not in formulas:
        raise ValueError(f"unknown form {form!r}")
    return float(formulas[form])


@dataclass
class BlandAltman:
    bias: float  # mean of a minus b
    lower: float  # limits of agreement, bias plus or minus 1.96 standard deviations
    upper: float
    slope: float  # change in the difference per point of the mean of the two: proportional bias
    slope_p: float


def bland_altman(a, b) -> BlandAltman:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff, mean = x - y, (x + y) / 2
    sd = diff.std(ddof=1)
    fit = stats.linregress(mean, diff)
    return BlandAltman(
        bias=float(diff.mean()),
        lower=float(diff.mean() - 1.96 * sd),
        upper=float(diff.mean() + 1.96 * sd),
        slope=float(fit.slope),
        slope_p=float(fit.pvalue),
    )


def smd(a, b) -> float:
    """Difference of the means of `a` and `b` over the pooled standard deviation."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = np.sqrt(
        ((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2)
    )
    return float((x.mean() - y.mean()) / pooled)


Statistic = Callable[..., float]


def bootstrap(
    statistic: Statistic, *arrays, n: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float]:
    """The statistic on the data, and the ends of its percentile interval over resampled essays."""
    arrays = [np.asarray(a) for a in arrays]
    size = len(arrays[0])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        pick = rng.integers(0, size, size)
        draws.append(statistic(*(a[pick] for a in arrays)))
    tail = (1 - level) / 2 * 100
    low, high = np.nanpercentile(draws, [tail, 100 - tail])
    return float(statistic(*arrays)), float(low), float(high)


def paired_difference(
    statistic: Statistic, truth, first, second, n: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Statistic of `first` minus statistic of `second` against the same truth, with its interval.

    Both scorers are scored on the same resampled essays, so the interval is that of the gap and
    is much narrower than the two separate intervals would suggest.
    """

    def gap(t, f, s):
        return statistic(t, f) - statistic(t, s)

    return bootstrap(gap, truth, first, second, n=n, seed=seed)
