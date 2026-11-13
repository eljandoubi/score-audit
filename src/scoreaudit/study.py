"""The tables of the audit, built from the predictions of each scorer."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from scoreaudit import irt
from scoreaudit.data import TRAITS
from scoreaudit.fairness import (
    cleary_model,
    error_model,
    group_report,
    impact_ratios,
    smd_machine_human,
)
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
from scoreaudit.mitigate import crossfit_calibration, reweigh
from scoreaudit.raters import wide

# Williamson, Xi and Breyer (2012): what a scorer has to reach to be used next to a person.
QWK_FLOOR = 0.70
MAX_DEGRADATION = 0.10
MAX_SMD = 0.15
SAME_WITHIN = 0.10  # a change of the mean score smaller than this counts as no change
STABLE_QWK = 0.90  # and the scores must keep this agreement with the ones before the change


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
    three columns say whether the scorer meets the three criteria of Williamson, Xi and Breyer;
    the kappa criterion is applied to the agreement with one rater, the figure that two raters
    can be compared on, and not to the mean of the two, which holds less noise.
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
                "meets_qwk": vs_single[0] >= QWK_FLOOR,
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


def robustness_table(
    before: pd.Series,
    after: dict[str, pd.Series],
    truth: pd.Series,
    expected: dict[str, str],
    n_boot: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """What happens to the scores of one scorer when the essays are changed.

    `expected` says, for each change, whether the score should stay put ("same") or go down
    ("lower"). The change is read as a mean shift with an interval, as the share of essays that
    move by half a point or more, and as the agreement between the scores before and after.
    A change that should leave the score alone is only as expected when the mean stays put and
    the essays keep their order (agreement before and after of at least `STABLE_QWK`): a scorer
    that moves every essay a little in both directions has the same mean and is not stable.
    """
    rows = []
    for name, changed in after.items():
        common = before.index.intersection(changed.index).intersection(truth.index)
        b = before.loc[common].clip(1, 5).to_numpy()
        a = changed.loc[common].clip(1, 5).to_numpy()
        t = truth.loc[common].to_numpy()
        shift = bootstrap(lambda x, y: float((y - x).mean()), b, a, n=n_boot, seed=seed)
        stability = qwk(b, a)
        wanted = expected.get(name)
        if wanted == "same":
            holds = abs(shift[0]) <= SAME_WITHIN and bool(
                np.isnan(stability) or stability >= STABLE_QWK
            )
        elif wanted == "lower":
            holds = shift[0] <= -SAME_WITHIN
        else:
            holds = None
        rows.append(
            {
                "change": name,
                "expected": wanted,
                "n": len(common),
                "mean_shift": shift[0],
                "shift_low": shift[1],
                "shift_high": shift[2],
                "moved_half_point": float((np.abs(a - b) >= 0.5).mean()),
                "qwk_before_after": stability,
                "qwk_truth_before": qwk(b, t),
                "qwk_truth_after": qwk(a, t),
                "as_expected": holds,
            }
        )
    return pd.DataFrame(rows)


def pool_small(groups: pd.Series, min_n: int = 100, label: str = "Other") -> pd.Series:
    """Groups with fewer than `min_n` essays are merged: their results would not mean anything."""
    counts = groups.value_counts()
    small = counts[counts < min_n].index
    return groups.where(~groups.isin(small), label)


def scoring_frame(essays: pd.DataFrame, pred: pd.Series, attributes: list[str]) -> pd.DataFrame:
    """One row per essay: the corpus score as `truth`, the scorer's as `pred`, the writer groups."""
    base = essays.set_index("essay_id")[[*attributes, "overall"]].rename(
        columns={"overall": "truth"}
    )
    frame = base.join(pred.rename("pred"), how="inner")
    return frame.dropna(subset=[*attributes, "pred"])


def fairness_tables(
    frame: pd.DataFrame, attributes: list[str], cuts=(3.0, 3.5), n_boot: int = 300
) -> dict[str, pd.DataFrame]:
    """Subgroup agreement and bias, impact ratios and the two regressions, for each attribute."""
    return {
        "groups": pd.concat([group_report(frame, a, n_boot=n_boot) for a in attributes]),
        "impact": pd.concat(
            [impact_ratios(frame, a, cut).assign(cut=cut) for a in attributes for cut in cuts]
        ),
        "error_model": error_model(frame, attributes),
        "cleary_model": cleary_model(frame, attributes),
    }


def psychometrics(essays: pd.DataFrame, items=TRAITS):
    """A graded response model of the trait scores: item table, summary, proficiency per essay."""
    responses = np.column_stack([irt.recode(essays[t]) for t in items])
    model = irt.fit(responses)
    theta, sd = irt.proficiency(model, responses)
    table = pd.DataFrame({"item": list(items), "discrimination": model.a})
    for k in range(model.b.shape[1]):
        table[f"b{k + 1}"] = model.b[:, k]
    summary = {
        "n": len(essays),
        "items": len(items),
        "reliability": irt.reliability(theta, sd),
        "alpha": irt.cronbach_alpha(essays[list(items)].to_numpy()),
        "loglik": model.loglik,
    }
    return table, summary, pd.Series(theta, index=essays["essay_id"].to_numpy(), name="theta")


def error_by_proficiency(
    frame: pd.DataFrame, theta: pd.Series, bands: int = 5, n_boot: int = 300, seed: int = 0
) -> pd.DataFrame:
    """Bias and error of the scorer across the range of writer proficiency, in bands of equal size.

    A scorer that pulls every essay toward the middle is too generous at the low end and too
    harsh at the high end, which the overall bias hides.
    """
    data = frame.join(theta, how="inner")
    data["band"] = pd.qcut(data["theta"], bands, labels=False) + 1
    rows = []
    for band, part in data.groupby("band"):
        pred, truth = part["pred"].clip(1, 5).to_numpy(), part["truth"].to_numpy()
        bias = bootstrap(lambda p, t: float((p - t).mean()), pred, truth, n=n_boot, seed=seed)
        rows.append(
            {
                "band": band,
                "n": len(part),
                "theta_mean": part["theta"].mean(),
                "human_mean": truth.mean(),
                "machine_mean": pred.mean(),
                "bias": bias[0],
                "bias_low": bias[1],
                "bias_high": bias[2],
                "mae": mae(pred, truth),
            }
        )
    return pd.DataFrame(rows)


def _gaps(frame: pd.DataFrame, attribute: str, min_n: int = 30) -> dict:
    report = group_report(frame, attribute, min_n=min_n, n_boot=20)
    return {
        "bias_gap": float(report["bias"].max() - report["bias"].min()),
        "max_abs_smd": float(report["smd"].abs().max()),
        "worst_group_qwk": float(report["qwk"].min()),
    }


def mitigation_table(
    frame: pd.DataFrame,
    attribute: str,
    train_groups: np.ndarray,
    train_scores: np.ndarray,
    refit: Callable[[np.ndarray], pd.Series],
    n_boot: int = 300,
) -> pd.DataFrame:
    """What narrowing the gaps between groups costs in agreement with the raters.

    Four versions of the same scorer: as it is, refitted on reweighed training essays, mapped onto
    the raters' scale by one common line, and by one line for each group. `refit` takes training
    weights and returns the predictions for the essays of `frame`. The line for each group uses
    the group of the writer when an essay is scored.
    """
    variants = {
        "as fitted": frame["pred"],
        "reweighed training": refit(reweigh(train_groups, train_scores)),
        "common line": crossfit_calibration(frame),
        "line for each group": crossfit_calibration(frame, attribute),
    }
    rows = []
    for name, pred in variants.items():
        changed = frame.assign(pred=pred.reindex(frame.index))
        agree = bootstrap(
            qwk, changed["pred"].clip(1, 5).to_numpy(), changed["truth"].to_numpy(), n=n_boot
        )
        rows.append(
            {
                "version": name,
                "attribute": attribute,
                "qwk": agree[0],
                "qwk_low": agree[1],
                "qwk_high": agree[2],
                **_gaps(changed, attribute),
            }
        )
    return pd.DataFrame(rows)
