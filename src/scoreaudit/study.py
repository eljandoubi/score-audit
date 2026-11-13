"""The tables of the audit, built from the predictions of each scorer."""

import numpy as np
import pandas as pd

from scoreaudit.fairness import smd_machine_human
from scoreaudit.metrics import (
    bland_altman,
    bootstrap,
    exact,
    icc,
    mae,
    pearson,
    qwk,
    spearman,
    within,
)
from scoreaudit.raters import wide

# Williamson, Xi and Breyer (2012): what a scorer has to reach to be used next to a person.
QWK_FLOOR = 0.70
MAX_DEGRADATION = 0.10
MAX_SMD = 0.15


def _expert_pairs(ratings: pd.DataFrame, pred: pd.Series, score: str = "overall"):
    """Scores of the scorer, of each of the two raters and their mean, for the essays all have."""
    pair = wide(ratings, score).join(pred.rename("pred"), how="inner").dropna()
    return (
        pair["pred"].clip(1, 5).to_numpy(),
        pair["first"].to_numpy(),
        pair["second"].to_numpy(),
        pair.index,
    )


def _single(p, a, b):
    return (qwk(p, a) + qwk(p, b)) / 2


def _loss(p, a, b):
    return qwk(a, b) - _single(p, a, b)


def agreement_table(
    predictions: dict[str, pd.Series], ratings: pd.DataFrame, n_boot: int = 500, seed: int = 0
) -> pd.DataFrame:
    """One row per scorer: agreement with the mean of the raters and with each rater alone.

    `qwk_humans` is the agreement of the two raters on the same essays: it is what the scorer
    is measured against, and `loss` is what is lost by swapping one rater for the scorer. The last
    three columns say whether the scorer meets the three criteria of Williamson, Xi and Breyer.
    """
    rows = []
    for name, pred in predictions.items():
        p, a, b, index = _expert_pairs(ratings, pred)
        mean = (a + b) / 2
        vs_mean = bootstrap(qwk, p, mean, n=n_boot, seed=seed)
        vs_single = bootstrap(_single, p, a, b, n=n_boot, seed=seed)
        humans = bootstrap(qwk, a, b, n=n_boot, seed=seed)
        loss = bootstrap(_loss, p, a, b, n=n_boot, seed=seed)
        smd = smd_machine_human(p, mean)
        rows.append(
            {
                "scorer": name,
                "n": len(index),
                "qwk_mean": vs_mean[0],
                "qwk_mean_low": vs_mean[1],
                "qwk_mean_high": vs_mean[2],
                "qwk_single": vs_single[0],
                "qwk_single_low": vs_single[1],
                "qwk_single_high": vs_single[2],
                "qwk_humans": humans[0],
                "loss": loss[0],
                "loss_low": loss[1],
                "loss_high": loss[2],
                "exact": exact(p, mean),
                "within_half": within(p, mean, 0.5),
                "pearson": pearson(p, mean) if np.std(p) > 0 else float("nan"),
                "icc": icc(np.column_stack([p, mean]), "2,1"),
                "mae": mae(p, mean),
                "bias": float((p - mean).mean()),
                "smd": smd,
                "meets_qwk": vs_mean[0] >= QWK_FLOOR,
                "meets_loss": loss[0] <= MAX_DEGRADATION,
                "meets_smd": abs(smd) <= MAX_SMD,
            }
        )
    return pd.DataFrame(rows)


def bland_altman_table(predictions: dict[str, pd.Series], ratings: pd.DataFrame) -> pd.DataFrame:
    """Bias, limits of agreement and proportional bias of each scorer against the rater mean."""
    rows = []
    for name, pred in predictions.items():
        p, a, b, _ = _expert_pairs(ratings, pred)
        result = bland_altman(p, (a + b) / 2)
        rows.append({"scorer": name, **result.__dict__})
    return pd.DataFrame(rows)


def prompt_spread(
    scores: dict[str, pd.Series], truth: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How much the zero-shot score depends on the wording of the prompt.

    Returns one row per prompt (mean, spread and agreement with the raters) and the rank
    correlations between the prompts, which say whether two wordings order the essays alike.
    """
    common = truth.index
    for s in scores.values():
        common = common.intersection(s.index)
    table = pd.DataFrame(
        [
            {
                "prompt": name,
                "n": len(common),
                "mean": s.loc[common].mean(),
                "sd": s.loc[common].std(),
                "spearman": spearman(s.loc[common], truth.loc[common]),
                "qwk": qwk(s.loc[common].clip(1, 5), truth.loc[common]),
            }
            for name, s in scores.items()
        ]
    )
    frame = pd.DataFrame({name: s.loc[common] for name, s in scores.items()})
    return table, frame.corr(method="spearman")


def sampling_reliability(probs: np.ndarray, values=(1, 2, 3, 4, 5)) -> dict:
    """Reliability of one sampled answer, from the probabilities the model gives to each score.

    If the scorer is run with sampling, two runs on the same essay can differ. The share of
    the variance of a single run that comes from the essays and not from the sampling is the
    intraclass correlation of two runs, which has a closed form here.
    """
    v = np.asarray(values, dtype=float)
    mean = probs @ v
    within_variance = (probs @ v**2 - mean**2).mean()
    between_variance = mean.var(ddof=1)
    return {
        "icc_one_run": float(between_variance / (between_variance + within_variance)),
        "mean_sd_within_essay": float(np.sqrt(np.maximum(probs @ v**2 - mean**2, 0)).mean()),
    }
