"""Ways to narrow a gap between groups of writers, so that the cost in accuracy can be measured."""

import numpy as np
import pandas as pd

from scoreaudit.fairness import MIN_N


def reweigh(groups, scores, max_weight: float = 5.0) -> np.ndarray:
    """Training weights that make group and score level independent (Kamiran and Calders, 2012).

    A cell with fewer essays than independence would give is weighted up, and one with more is
    weighted down. The weights are capped, because the smallest cells would otherwise carry
    the whole fit, and rescaled to a mean of one.
    """
    frame = pd.DataFrame({"g": np.asarray(groups), "y": np.rint(np.asarray(scores, dtype=float))})
    p_group = frame["g"].map(frame["g"].value_counts(normalize=True))
    p_score = frame["y"].map(frame["y"].value_counts(normalize=True))
    p_cell = frame.groupby(["g", "y"])["g"].transform("size") / len(frame)
    weights = np.minimum((p_group * p_score / p_cell).to_numpy(), max_weight)
    return weights / weights.mean()


def _line(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(pred, truth, 1)
    return float(slope), float(intercept)


def crossfit_calibration(
    frame: pd.DataFrame,
    by: str | None = None,
    folds: int = 5,
    seed: int = 0,
    min_n: int = MIN_N,
) -> pd.Series:
    """The machine score mapped onto the scale of the raters by a line, essay by essay.

    Each essay is mapped by a line fitted on the essays of the other folds, so the score of an
    essay never helps to calibrate itself. With `by`, one line is fitted for each group of that
    column that has at least `min_n` essays in the fitting folds, and the others use the
    common line. Using the group of the writer at scoring time is a choice with its own legal and
    ethical cost: compare it with the common line, which does not use it.
    """
    pred = frame["pred"].clip(1, 5).to_numpy()
    truth = frame["truth"].to_numpy()
    order = np.random.default_rng(seed).permutation(len(frame))
    fold = np.empty(len(frame), dtype=int)
    fold[order] = np.arange(len(frame)) % folds
    out = np.empty(len(frame))
    for k in range(folds):
        fit, hold = fold != k, fold == k
        slope, intercept = _line(pred[fit], truth[fit])
        out[hold] = slope * pred[hold] + intercept
        if by is None:
            continue
        groups = frame[by].to_numpy()
        for group in set(groups[hold]):
            members = fit & (groups == group)
            if members.sum() >= min_n:
                s, i = _line(pred[members], truth[members])
                target = hold & (groups == group)
                out[target] = s * pred[target] + i
    return pd.Series(np.clip(out, 1, 5), index=frame.index)
