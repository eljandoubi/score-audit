import numpy as np
import pandas as pd
import pytest

from scoreaudit.metrics import icc
from scoreaudit.study import (
    MAX_SMD,
    agreement_table,
    bland_altman_table,
    prompt_spread,
    sampling_reliability,
)


def rated_essays(n=500, seed=0):
    """Latent quality, two noisy single ratings on the 1 to 5 scale, in the layout of the corpus."""
    rng = np.random.default_rng(seed)
    quality = rng.uniform(1.5, 4.5, n)
    first = np.clip(np.rint(quality + rng.normal(0, 0.6, n)), 1, 5)
    second = np.clip(np.rint(quality + rng.normal(0, 0.6, n)), 1, 5)
    ids = [f"e{i}" for i in range(n)]
    rows = []
    for slot, scores in ((1, first), (2, second)):
        rows.append(
            pd.DataFrame({"essay_id": ids, "slot": slot, "rater": f"r{slot}", "overall": scores})
        )
    return pd.concat(rows, ignore_index=True), pd.Series((first + second) / 2, index=ids), quality


def test_a_scorer_that_gives_the_rater_mean_agrees_better_with_each_rater_than_they_do_together():
    ratings, mean, _ = rated_essays()
    table = agreement_table({"oracle": mean}, ratings, n_boot=50).iloc[0]
    assert table["qwk_mean"] == 1.0
    assert table["qwk_single"] > table["qwk_humans"]
    assert table["loss"] < 0
    assert table["meets_qwk"] and table["meets_loss"] and table["meets_smd"]
    assert table["n"] == 500


def test_a_constant_scorer_has_no_agreement_and_fails_the_criteria():
    ratings, mean, _ = rated_essays()
    flat = pd.Series(3.0, index=mean.index)
    row = agreement_table({"flat": flat}, ratings, n_boot=50).iloc[0]
    assert abs(row["qwk_mean"]) < 1e-9
    assert np.isnan(row["pearson"])
    assert not row["meets_qwk"]
    assert not row["meets_loss"]


def test_a_scorer_that_is_too_generous_fails_the_smd_criterion():
    ratings, mean, _ = rated_essays()
    row = agreement_table({"generous": mean + 1.0}, ratings, n_boot=50).iloc[0]
    assert abs(row["smd"]) > MAX_SMD
    assert not row["meets_smd"]
    assert 0.8 < row["bias"] < 1.0  # clipped at 5, so a little under the full point


def test_essays_without_ratings_are_left_out_and_every_scorer_gets_a_row():
    ratings, mean, _ = rated_essays(n=100)
    extra = pd.concat([mean, pd.Series([3.0, 4.0], index=["unrated1", "unrated2"])])
    table = agreement_table({"a": extra, "b": mean}, ratings, n_boot=20)
    assert table["scorer"].tolist() == ["a", "b"]
    assert table["n"].tolist() == [100, 100]


def test_the_intervals_bracket_the_estimates():
    ratings, mean, quality = rated_essays()
    noisy = pd.Series(
        quality + np.random.default_rng(1).normal(0, 0.5, len(quality)), index=mean.index
    )
    row = agreement_table({"noisy": noisy}, ratings, n_boot=100).iloc[0]
    assert row["qwk_mean_low"] < row["qwk_mean"] < row["qwk_mean_high"]
    assert row["qwk_single_low"] < row["qwk_single"] < row["qwk_single_high"]
    assert row["loss_low"] < row["loss"] < row["loss_high"]


def test_bland_altman_table_sees_a_shift():
    ratings, mean, _ = rated_essays()
    row = bland_altman_table({"shifted": mean + 0.3}, ratings).iloc[0]
    assert row["bias"] == pytest.approx(0.3, abs=0.03)
    assert row["lower"] < row["bias"] < row["upper"]


def test_two_identical_prompts_put_the_essays_in_the_same_order():
    _, mean, quality = rated_essays()
    scores = {"a": mean, "b": mean + 0.5, "c": mean.iloc[::-1].set_axis(mean.index)}
    table, corr = prompt_spread(scores, mean)
    assert corr.loc["a", "b"] == pytest.approx(1.0)
    assert corr.loc["a", "c"] < 0.2
    assert table.set_index("prompt").loc["b", "mean"] == pytest.approx(mean.mean() + 0.5)
    assert table["n"].nunique() == 1


def test_prompts_are_compared_on_the_essays_they_all_scored():
    _, mean, _ = rated_essays(n=50)
    table, _ = prompt_spread({"a": mean, "b": mean.iloc[:30]}, mean)
    assert table["n"].tolist() == [30, 30]


def test_sampling_reliability_of_certain_and_of_undecided_models():
    certain = np.eye(5)[np.random.default_rng(0).integers(0, 5, 400)]
    assert sampling_reliability(certain)["icc_one_run"] == pytest.approx(1.0)
    uniform = np.full((400, 5), 0.2)
    result = sampling_reliability(uniform)
    assert result["icc_one_run"] == pytest.approx(0.0, abs=1e-9)
    assert result["mean_sd_within_essay"] == pytest.approx(np.sqrt(2.0))


def test_sampling_reliability_matches_a_simulation_of_two_runs():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(3000, 5)) + np.linspace(0, 2, 5) * rng.normal(size=(3000, 1))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    draws = np.stack(
        [(rng.random((3000, 1)) > probs.cumsum(axis=1)).sum(axis=1) + 1 for _ in range(2)], axis=1
    )
    assert sampling_reliability(probs)["icc_one_run"] == pytest.approx(icc(draws, "1,1"), abs=0.04)
