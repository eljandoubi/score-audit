"""Hybrid scoring: the essays the machine is least sure of go to a person."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from scoreaudit.metrics import qwk


def ensemble_spread(
    train_features: np.ndarray,
    train_scores: np.ndarray,
    features: np.ndarray,
    alpha: float,
    n_models: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """Standard deviation of the predictions of ridge models fitted on resampled training essays.

    Essays on which the models disagree are essays whose score depends on which training essays the
    scorer happened to see. That is a measure of how much to trust the score.
    """
    rng = np.random.default_rng(seed)
    predictions = []
    for _ in range(n_models):
        counts = np.bincount(rng.integers(0, len(train_scores), len(train_scores)),
                             minlength=len(train_scores))  # fmt: skip
        model = Ridge(alpha=alpha).fit(train_features, train_scores, sample_weight=counts)
        predictions.append(model.predict(features))
    return np.std(predictions, axis=0, ddof=1)


def coverage_curve(
    machine,
    reviewer,
    reference,
    uncertainty,
    fractions=(0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
    n_random: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    """Agreement with the reference when a share of the essays is scored by the reviewer instead.

    The reviewer is one rater and the reference is another, so the curve ends at the agreement of
    two raters and not at a perfect score. `qwk_routed` sends the most uncertain essays first and
    `qwk_random` sends the same number of essays at random, which is what a screen for uncertainty
    has to beat.
    """
    machine = np.clip(np.asarray(machine, dtype=float), 1, 5)
    reviewer = np.asarray(reviewer, dtype=float)
    reference = np.asarray(reference, dtype=float)
    by_doubt = np.argsort(-np.asarray(uncertainty, dtype=float), kind="stable")
    rng = np.random.default_rng(seed)
    n = len(machine)
    rows = []
    for fraction in fractions:
        k = int(round(fraction * n))
        routed = machine.copy()
        routed[by_doubt[:k]] = reviewer[by_doubt[:k]]
        draws = []
        for _ in range(n_random if 0 < k < n else 1):
            picks = rng.permutation(n)[:k]
            mixed = machine.copy()
            mixed[picks] = reviewer[picks]
            draws.append(qwk(mixed, reference))
        rows.append(
            {
                "fraction": fraction,
                "n_routed": k,
                "qwk_routed": qwk(routed, reference),
                "qwk_random": float(np.mean(draws)),
            }
        )
    return pd.DataFrame(rows)
