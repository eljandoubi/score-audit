import numpy as np
import pandas as pd
import pytest

from scoreaudit.data import TRAITS
from scoreaudit.metrics import icc
from scoreaudit.study import (
    MAX_SMD,
    agreement_table,
    bland_altman_table,
    error_by_proficiency,
    fairness_tables,
    mitigation_table,
    pool_small,
    prompt_spread,
    psychometrics,
    robustness_table,
    sampling_reliability,
    scoring_frame,
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


def test_robustness_reads_the_shift_and_whether_it_is_the_one_that_was_expected():
    _, mean, _ = rated_essays()
    after = {"typos": mean - 0.3, "injection": mean, "greedy": mean + 0.5, "other": mean - 0.2}
    expected = {"typos": "lower", "injection": "same", "greedy": "same"}
    table = robustness_table(mean, after, mean, expected, n_boot=50).set_index("change")
    assert table.loc["typos", "mean_shift"] == pytest.approx(-0.3, abs=0.02)
    assert table.loc["typos", "as_expected"]
    assert table.loc["injection", "as_expected"]
    assert table.loc["injection", "moved_half_point"] == 0.0
    assert table.loc["injection", "qwk_before_after"] == 1.0
    assert not table.loc["greedy", "as_expected"]
    assert table.loc["other", "as_expected"] is None
    assert table.loc["typos", "shift_high"] < 0


def test_a_scorer_that_does_not_go_down_when_it_should_fails_the_expectation():
    _, mean, _ = rated_essays()
    table = robustness_table(mean, {"shuffle": mean + 0.01}, mean, {"shuffle": "lower"}, n_boot=20)
    assert not table.loc[0, "as_expected"]


def test_small_groups_are_pooled():
    groups = pd.Series(["A"] * 200 + ["B"] * 150 + ["C"] * 30 + ["D"] * 10)
    pooled = pool_small(groups, min_n=100)
    assert pooled.value_counts().to_dict() == {"A": 200, "B": 150, "Other": 40}
    assert pool_small(groups, min_n=5).equals(groups)


def test_the_scoring_frame_keeps_the_essays_that_have_everything():
    essays = pd.DataFrame(
        {
            "essay_id": ["a", "b", "c", "d"],
            "gender": ["F", "M", None, "F"],
            "overall": [3.0, 4.0, 2.0, 5.0],
        }
    )
    frame = scoring_frame(essays, pd.Series({"a": 3.5, "b": 3.0, "c": 2.0}), ["gender"])
    assert frame.index.tolist() == ["a", "b"]
    assert frame["truth"].tolist() == [3.0, 4.0]
    assert frame["pred"].tolist() == [3.5, 3.0]


def group_frame(n=900, seed=0):
    rng = np.random.default_rng(seed)
    group = np.where(np.arange(n) % 3 == 0, "B", "A")
    truth = rng.choice(np.arange(2.0, 4.01, 0.5), n)
    pred = truth + np.where(np.arange(n) % 2 == 0, 0.2, -0.2) + np.where(group == "B", 0.4, 0.0)
    return pd.DataFrame({"truth": truth, "pred": pred, "group": group})


def test_fairness_tables_cover_each_attribute_and_cut():
    tables = fairness_tables(group_frame(), ["group"], cuts=(3.0, 3.5), n_boot=20)
    assert set(tables) == {"groups", "impact", "error_model", "cleary_model"}
    assert set(tables["groups"]["group"]) == {"A", "B"}
    assert sorted(tables["impact"]["cut"].unique()) == [3.0, 3.5]
    assert "group: B vs A" in set(tables["error_model"]["term"])


def trait_scores(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    ability = rng.normal(size=n)
    frame = pd.DataFrame({"essay_id": [f"e{i}" for i in range(n)]})
    for trait in TRAITS:
        frame[trait] = np.clip(np.rint((3 + 0.7 * ability + rng.normal(0, 0.35, n)) * 2) / 2, 1, 5)
    return frame, ability


def test_psychometrics_gives_one_row_per_trait_and_a_proficiency_for_each_essay():
    essays, ability = trait_scores()
    table, summary, theta = psychometrics(essays)
    assert table["item"].tolist() == list(TRAITS)
    assert (table["discrimination"] > 0).all()
    assert summary["reliability"] > 0.8
    assert summary["alpha"] > 0.8
    assert np.corrcoef(theta.to_numpy(), ability)[0, 1] > 0.9
    assert theta.index.tolist() == essays["essay_id"].tolist()


def test_a_scorer_that_pulls_toward_the_middle_is_too_generous_at_the_bottom_only():
    rng = np.random.default_rng(1)
    truth = rng.uniform(1, 5, 1000)
    frame = pd.DataFrame(
        {"truth": truth, "pred": 0.5 * truth + 1.5}, index=[f"e{i}" for i in range(1000)]
    )
    theta = pd.Series(truth + rng.normal(0, 0.3, 1000), index=frame.index, name="theta")
    table = error_by_proficiency(frame, theta, bands=5, n_boot=50)
    assert table["n"].tolist() == [200] * 5
    assert table["bias"].iloc[0] > 0.5
    assert table["bias"].iloc[-1] < -0.5
    assert table["bias"].is_monotonic_decreasing
    assert (table["bias_low"] < table["bias"]).all()


def test_mitigation_compares_four_versions_and_the_group_line_closes_the_gap():
    frame = group_frame()
    calls = []

    def refit(weights):
        calls.append(len(weights))
        return frame["pred"] - np.where(frame["group"] == "B", 0.2, 0.0)

    table = mitigation_table(
        frame, "group", frame["group"].to_numpy(), frame["truth"].to_numpy(), refit, n_boot=20
    ).set_index("version")
    assert calls == [len(frame)]
    assert table.index.tolist() == [
        "as fitted",
        "reweighed training",
        "common line",
        "line for each group",
    ]
    assert table.loc["line for each group", "bias_gap"] < 0.1
    assert table.loc["as fitted", "bias_gap"] > 0.35
    assert table.loc["reweighed training", "bias_gap"] < table.loc["as fitted", "bias_gap"]
    assert table.loc["line for each group", "qwk"] >= table.loc["as fitted", "qwk"] - 0.03
