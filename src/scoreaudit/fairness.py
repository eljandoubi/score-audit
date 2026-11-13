"""Does the scorer treat every group of writers the same way?

Every function takes a frame with the human score in `truth`, the machine score in `pred` and one
column per writer attribute (gender, race, ses).
"""

import numpy as np
import pandas as pd
from scipy import stats

from scoreaudit.metrics import LOW, STEP, bootstrap, levels, qwk

MIN_N = 30


def smd_machine_human(pred, truth) -> float:
    """Gap between the mean machine score and the mean human score, over the spread of the two.

    This is the statistic of Williamson, Xi and Breyer (2012), who ask for an absolute value
    below 0.15 in every group.
    """
    p, t = np.asarray(pred, dtype=float), np.asarray(truth, dtype=float)
    return float((p.mean() - t.mean()) / np.sqrt((p.var(ddof=1) + t.var(ddof=1)) / 2))


def group_report(
    frame: pd.DataFrame, by: str, min_n: int = MIN_N, n_boot: int = 500, seed: int = 0
) -> pd.DataFrame:
    """Agreement and bias of the scorer inside each group of the column `by`.

    `bias` is the mean of machine minus human: positive means the scorer is more generous to the
    group than the raters are. Groups smaller than `min_n` are left out, their intervals would say
    nothing.
    """
    rows = []
    for group, part in frame.groupby(by):
        if len(part) < min_n:
            continue
        pred = part["pred"].clip(1, 5).to_numpy()
        truth = part["truth"].to_numpy()
        bias = bootstrap(lambda p, t: float((p - t).mean()), pred, truth, n=n_boot, seed=seed)
        agree = bootstrap(qwk, pred, truth, n=n_boot, seed=seed)
        rows.append(
            {
                "attribute": by,
                "group": group,
                "n": len(part),
                "human_mean": truth.mean(),
                "machine_mean": pred.mean(),
                "bias": bias[0],
                "bias_low": bias[1],
                "bias_high": bias[2],
                "smd": smd_machine_human(pred, truth),
                "qwk": agree[0],
                "qwk_low": agree[1],
                "qwk_high": agree[2],
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def bias_gap(
    frame: pd.DataFrame, by: str, group: str, reference: str, n: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Bias of a group minus the bias of the reference group, with a bootstrap interval.

    The two groups are resampled on their own, since they are different essays.
    """
    residual = (frame["pred"].clip(1, 5) - frame["truth"]).to_numpy()
    a = residual[(frame[by] == group).to_numpy()]
    b = residual[(frame[by] == reference).to_numpy()]
    rng = np.random.default_rng(seed)
    draws = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(low), float(high)


def impact_ratios(frame: pd.DataFrame, by: str, cut: float, min_n: int = MIN_N) -> pd.DataFrame:
    """Share of each group scoring at least `cut`, and its ratio to the best-placed group.

    The scorer's score is rounded to the half point first, as the mean of two raters is, so that the
    two rates compare like with like: a cut on the raw score would count the essays that sit
    exactly on the cut for the raters and only half of them for the scorer.

    The ratio is the one of the four-fifths rule and of New York City Local Law 144: a value below
    0.8 is read as adverse impact. It is given for the raters and for the scorer, so one can see
    whether the scorer makes a gap that the raters did not make, or only repeats it.
    """
    rows = []
    for group, part in frame.groupby(by):
        if len(part) < min_n:
            continue
        on_the_scale = LOW + STEP * levels(part["pred"].clip(1, 5))
        rows.append(
            {
                "attribute": by,
                "group": group,
                "n": len(part),
                "rate_human": float((part["truth"] >= cut).mean()),
                "rate_machine": float((on_the_scale >= cut).mean()),
            }
        )
    table = pd.DataFrame(rows)
    table["ratio_human"] = table["rate_human"] / table["rate_human"].max()
    table["ratio_machine"] = table["rate_machine"] / table["rate_machine"].max()
    table["adverse_machine"] = table["ratio_machine"] < 0.8
    return table.sort_values("n", ascending=False).reset_index(drop=True)


def ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least squares with robust (HC3) standard errors: coefficients, errors, p values."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    inverse = np.linalg.inv(x.T @ x)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, inverse, x)
    scaled = x * (residual / (1 - leverage))[:, None]
    cov = inverse @ (scaled.T @ scaled) @ inverse
    se = np.sqrt(np.diag(cov))
    p = 2 * stats.t.sf(np.abs(beta / se), len(y) - x.shape[1])
    return beta, se, p


def group_regression(
    frame: pd.DataFrame, outcome: str, covariate: str, factors: list[str]
) -> pd.DataFrame:
    """Regress `outcome` on `covariate` and on writer groups, each against its largest group."""
    columns = [np.ones(len(frame)), (frame[covariate] - frame[covariate].mean()).to_numpy()]
    names = ["intercept", covariate]
    for factor in factors:
        reference = frame[factor].value_counts().idxmax()
        for group in sorted(set(frame[factor].dropna()) - {reference}):
            columns.append((frame[factor] == group).to_numpy(dtype=float))
            names.append(f"{factor}: {group} vs {reference}")
    beta, se, p = ols(np.column_stack(columns), frame[outcome].to_numpy())
    return pd.DataFrame(
        {
            "term": names,
            "coef": beta,
            "se": se,
            "low": beta - 1.96 * se,
            "high": beta + 1.96 * se,
            "p": p,
        }
    )


def error_model(frame: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """Machine minus human score, regressed on the human score and the writer groups.

    A coefficient for a group says how much more generous the scorer is to that group than to the
    largest one, among essays that the raters scored alike.
    """
    data = frame.assign(error=frame["pred"].clip(1, 5) - frame["truth"])
    return group_regression(data, "error", "truth", factors)


def cleary_model(frame: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """Human score regressed on the machine score and the writer groups (Cleary, 1968).

    A group coefficient away from zero means that the same machine score stands for a different
    human score depending on the group.
    """
    data = frame.assign(pred=frame["pred"].clip(1, 5))
    return group_regression(data, "truth", "pred", factors)
