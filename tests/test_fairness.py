import numpy as np
import pandas as pd
import pytest

from scoreaudit.fairness import bias_gap, group_report, smd_machine_human


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
