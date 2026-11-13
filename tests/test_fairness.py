import numpy as np
import pandas as pd
import pytest

from scoreaudit.fairness import (
    bias_gap,
    cleary_model,
    error_model,
    group_regression,
    group_report,
    impact_ratios,
    ols,
    smd_machine_human,
)


def frame_with_bias(n=600, shift_b=0.4, seed=0):
    """Group A is scored 0.2 above or below the raters in turn. Group B gets `shift_b` on top."""
    rng = np.random.default_rng(seed)
    group = np.where(np.arange(n) % 3 == 0, "B", "A")
    truth = rng.choice([2.0, 2.5, 3.0, 3.5, 4.0], n)
    noise = np.where(np.arange(n) % 2 == 0, 0.2, -0.2)
    pred = truth + noise + np.where(group == "B", shift_b, 0.0)
    return pd.DataFrame({"truth": truth, "pred": pred, "group": group})


def test_smd_follows_the_ets_formula():
    pred = np.array([2.0, 4.0, 3.0, 5.0])
    truth = np.array([1.0, 3.0, 2.0, 4.0])
    spread = np.sqrt((pred.var(ddof=1) + truth.var(ddof=1)) / 2)
    assert smd_machine_human(pred, truth) == pytest.approx(1.0 / spread)


def test_smd_is_zero_for_a_scorer_with_the_right_mean():
    truth = np.array([1.0, 3.0, 2.0, 4.0])
    assert smd_machine_human(truth[::-1], truth) == 0.0


def test_group_report_finds_the_group_that_is_scored_too_high():
    report = group_report(frame_with_bias(), "group", n_boot=200).set_index("group")
    assert report.loc["B", "bias"] == pytest.approx(0.4)
    assert report.loc["A", "bias"] == pytest.approx(0.0, abs=1e-9)
    assert report.loc["B", "bias_low"] > 0.3
    assert report.loc["B", "smd"] > 0.15 > abs(report.loc["A", "smd"])
    assert report.loc["A", "n"] == 400
    assert report.loc["B", "n"] == 200


def test_small_groups_are_left_out_and_the_largest_comes_first():
    frame = frame_with_bias()
    frame.loc[frame.index[:5], "group"] = "C"
    report = group_report(frame, "group", n_boot=50)
    assert "C" not in set(report["group"])
    assert report["group"].tolist()[0] == "A"


def test_group_report_clips_predictions_to_the_scale():
    frame = frame_with_bias()
    frame["pred"] = frame["pred"] * 10
    report = group_report(frame, "group", n_boot=50)
    assert report["machine_mean"].max() <= 5.0


def test_bias_gap_is_the_difference_of_the_two_biases_and_excludes_zero():
    gap, low, high = bias_gap(frame_with_bias(), "group", "B", "A", n=300)
    assert gap == pytest.approx(0.4)
    assert 0.3 < low < gap < high


def test_bias_gap_interval_covers_zero_when_the_groups_are_treated_alike():
    a = frame_with_bias(shift_b=0.0).query("group == 'A'")
    frame = pd.concat([a, a.assign(group="B")], ignore_index=True)
    gap, low, high = bias_gap(frame, "group", "B", "A", n=300)
    assert gap == 0.0
    assert low < 0 < high


def two_groups():
    truth = [4.0] * 6 + [2.0] * 4 + [4.0] * 3 + [2.0] * 7
    pred = [4.0] * 6 + [2.0] * 4 + [4.0] * 2 + [2.0] * 8
    group = ["A"] * 10 + ["B"] * 10
    return pd.DataFrame({"truth": truth, "pred": pred, "group": group})


def test_impact_ratio_compares_each_group_with_the_best_placed():
    table = impact_ratios(two_groups(), "group", cut=3.5, min_n=5).set_index("group")
    assert table.loc["A", "rate_human"] == 0.6
    assert table.loc["B", "rate_human"] == 0.3
    assert table.loc["B", "ratio_human"] == pytest.approx(0.5)
    assert table.loc["B", "ratio_machine"] == pytest.approx(0.2 / 0.6)
    assert table.loc["A", "ratio_machine"] == 1.0
    assert table["adverse_machine"].to_dict() == {"A": False, "B": True}


def test_impact_ratio_reads_the_scorer_on_the_half_point_scale_of_the_raters():
    frame = pd.DataFrame(
        {"truth": [3.5] * 10 + [3.0] * 10, "pred": [3.3] * 10 + [3.2] * 10, "group": ["A"] * 20}
    )
    row = impact_ratios(frame, "group", cut=3.5, min_n=5).iloc[0]
    assert row["rate_human"] == 0.5
    assert row["rate_machine"] == 0.5


def test_impact_ratio_leaves_out_small_groups():
    frame = pd.concat([two_groups(), two_groups().head(3).assign(group="C")])
    assert set(impact_ratios(frame, "group", cut=3.5, min_n=5)["group"]) == {"A", "B"}


def test_ols_recovers_the_coefficients():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 2 + 3 * x + rng.normal(0, 0.1, 500)
    beta, se, p = ols(np.column_stack([np.ones(500), x]), y)
    assert beta == pytest.approx([2, 3], abs=0.02)
    assert (se > 0).all()
    assert (p < 1e-10).all()


def test_robust_errors_are_close_to_the_classical_ones_when_the_noise_is_even():
    rng = np.random.default_rng(1)
    x = np.column_stack([np.ones(2000), rng.normal(size=2000)])
    y = x @ [1.0, 0.5] + rng.normal(0, 1.0, 2000)
    _, se, _ = ols(x, y)
    classical = np.sqrt(np.diag(np.linalg.inv(x.T @ x)))
    assert se == pytest.approx(classical, rel=0.15)


def test_each_group_is_compared_with_the_largest_one():
    frame = frame_with_bias()
    frame["other"] = np.where(np.arange(len(frame)) % 10 == 0, "Z", "Y")
    table = group_regression(frame.assign(y=frame["pred"]), "y", "truth", ["group", "other"])
    assert table["term"].tolist() == [
        "intercept",
        "truth",
        "group: B vs A",
        "other: Z vs Y",
    ]


def test_error_model_finds_the_group_scored_too_high_among_alike_essays():
    table = error_model(frame_with_bias(), ["group"]).set_index("term")
    row = table.loc["group: B vs A"]
    assert row["coef"] == pytest.approx(0.4, abs=0.05)
    assert row["low"] > 0.3
    assert row["p"] < 1e-6


def test_error_model_finds_nothing_when_the_groups_are_treated_alike():
    table = error_model(frame_with_bias(shift_b=0.0), ["group"]).set_index("term")
    assert abs(table.loc["group: B vs A", "coef"]) < 0.05


def test_cleary_model_sees_a_shift_between_the_groups_at_the_same_machine_score():
    rng = np.random.default_rng(3)
    n = 1500
    group = np.where(np.arange(n) % 3 == 0, "B", "A")
    pred = rng.uniform(1.5, 4.5, n)
    truth = pred + rng.normal(0, 0.3, n) - np.where(group == "B", 0.5, 0.0)
    frame = pd.DataFrame({"truth": truth, "pred": pred, "group": group})
    row = cleary_model(frame, ["group"]).set_index("term").loc["group: B vs A"]
    assert row["coef"] == pytest.approx(-0.5, abs=0.05)
    fair = frame.assign(truth=pred + rng.normal(0, 0.3, n))
    assert cleary_model(fair, ["group"]).set_index("term").loc["group: B vs A", "p"] > 0.001
