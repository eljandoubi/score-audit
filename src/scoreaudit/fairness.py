"""Does the scorer treat every group of writers the same way?

Every function takes a frame with the human score in `truth`, the machine score in `pred` and one
column per writer attribute (gender, race, ses).
"""

import numpy as np
import pandas as pd

from scoreaudit.metrics import bootstrap, qwk

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
