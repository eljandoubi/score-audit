"""What the human raters agree on, and how the machine compares with them."""

import numpy as np
import pandas as pd

from scoreaudit.metrics import exact, icc, qwk, within


def wide(ratings: pd.DataFrame, score: str = "overall") -> pd.DataFrame:
    """One row per essay, with the rating of the first and of the second rater."""
    table = ratings.pivot(index="essay_id", columns="slot", values=score)
    table.columns = ["first", "second"]
    return table


def human_ceiling(ratings: pd.DataFrame, score: str = "overall") -> dict:
    """Agreement between the two raters of an essay: the level a scorer is asked to reach."""
    pair = wide(ratings, score)
    first, second = pair["first"], pair["second"]
    table = pair.to_numpy()
    return {
        "n": len(pair),
        "qwk": qwk(first, second),
        "exact": exact(first, second),
        "within_1": within(first, second, 1.0),
        "icc_single": icc(table, "2,1"),
        "icc_mean_of_two": icc(table, "2,k"),
    }


def leniency(ratings: pd.DataFrame, score: str = "overall") -> pd.DataFrame:
    """For each rater, how far above or below the other rater of the same essay they score.

    A rater is compared with whoever scored the same essays, so the figure is a difference
    between two raters and not an absolute severity.
    """
    pair = ratings.pivot(index="essay_id", columns="slot", values=["rater", score])
    rows = []
    for me, other in ((1, 2), (2, 1)):
        rows.append(
            pd.DataFrame(
                {
                    "rater": pair["rater"][me],
                    "score": pair[score][me],
                    "difference": pair[score][me] - pair[score][other],
                }
            )
        )
    both = pd.concat(rows)
    table = both.groupby("rater").agg(
        n=("score", "size"), mean_score=("score", "mean"), leniency=("difference", "mean")
    )
    return table.sort_values("leniency")


def leniency_spread(table: pd.DataFrame, min_n: int = 100) -> float:
    """Standard deviation of the leniency of the raters who scored at least `min_n` essays."""
    return float(table.loc[table["n"] >= min_n, "leniency"].std(ddof=1))


def against_raters(pred: pd.Series, ratings: pd.DataFrame, score: str = "overall") -> dict:
    """Agreement of a scorer with the mean of the raters and with each rater alone.

    `pred` is indexed by essay id. A scorer is judged against the raters one by one because that
    is what the raters do to each other: the gap between the two is the loss of agreement
    that comes from swapping a human for the machine.
    """
    pair = wide(ratings, score).join(pred.rename("pred"), how="inner").dropna()
    guess = pair["pred"].clip(1, 5)
    mean = (pair["first"] + pair["second"]) / 2
    single = (qwk(guess, pair["first"]) + qwk(guess, pair["second"])) / 2
    humans = qwk(pair["first"], pair["second"])
    return {
        "n": len(pair),
        "qwk_mean": qwk(guess, mean),
        "qwk_single": single,
        "qwk_humans": humans,
        "degradation": humans - single,
        "exact_mean": exact(guess, mean),
        "within_half": within(guess, mean, 0.5),
        "icc_mean": icc(np.column_stack([guess, mean]), "2,1"),
    }
