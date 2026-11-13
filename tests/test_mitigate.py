import numpy as np
import pandas as pd
import pytest

from scoreaudit.fairness import group_report
from scoreaudit.metrics import qwk
from scoreaudit.mitigate import crossfit_calibration, reweigh


def test_reweighting_makes_group_and_score_independent():
    rng = np.random.default_rng(0)
    group = rng.choice(["A", "B"], 2000, p=[0.7, 0.3])
    score = np.where(group == "A", rng.choice([2, 3, 4], 2000, p=[0.2, 0.4, 0.4]),
                     rng.choice([2, 3, 4], 2000, p=[0.5, 0.4, 0.1]))  # fmt: skip
    w = reweigh(group, score)
    table = pd.DataFrame({"g": group, "y": score, "w": w}).pivot_table(
        index="g", columns="y", values="w", aggfunc="sum"
    )
    share = table / table.to_numpy().sum()
    expected = np.outer(share.sum(axis=1), share.sum(axis=0))
    assert share.to_numpy() == pytest.approx(expected, abs=1e-9)
    assert w.mean() == pytest.approx(1.0)


def test_the_cell_that_independence_would_fill_more_gets_a_weight_above_one():
    group = np.array(["A"] * 8 + ["B"] * 2)
    score = np.array([4, 4, 4, 4, 4, 4, 2, 2] + [4, 2])
    w = reweigh(group, score)
    assert w[-2] > w[0]  # B with a high score is rare, A with a high score is common


def test_weights_are_capped_before_they_are_rescaled():
    group = np.array(["A"] * 200 + ["B"])
    score = np.array([4] * 100 + [2] * 100 + [4])
    uncapped = reweigh(group, score, max_weight=1e9)
    capped = reweigh(group, score, max_weight=1.0)
    assert capped.max() < uncapped.max()
    assert capped.mean() == pytest.approx(1.0)


def biased_scorer(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    group = rng.choice(["A", "B"], n)
    truth = rng.choice(np.arange(2.0, 4.01, 0.5), n)
    noise = rng.normal(0, 0.25, n)
    pred = truth + noise + np.where(group == "A", -0.4, 0.4)
    return pd.DataFrame({"truth": truth, "pred": pred, "group": group})


def test_a_line_for_each_group_removes_the_gap_between_them():
    frame = biased_scorer()
    before = group_report(frame, "group", n_boot=20).set_index("group")["bias"]
    after = frame.assign(pred=crossfit_calibration(frame, "group"))
    after_bias = group_report(after, "group", n_boot=20).set_index("group")["bias"]
    assert abs(before["B"] - before["A"]) > 0.7
    assert abs(after_bias["B"] - after_bias["A"]) < 0.1


def test_one_common_line_does_not_remove_the_gap_between_groups():
    frame = biased_scorer()
    common = frame.assign(pred=crossfit_calibration(frame))
    bias = group_report(common, "group", n_boot=20).set_index("group")["bias"]
    # the line is flatter than the identity, which shrinks the gap, but most of it stays
    assert abs(bias["B"] - bias["A"]) > 0.4


def test_calibration_keeps_the_agreement_with_the_raters():
    frame = biased_scorer()
    raw = qwk(frame["pred"].clip(1, 5), frame["truth"])
    calibrated = qwk(crossfit_calibration(frame, "group"), frame["truth"])
    assert calibrated >= raw - 0.02


def test_a_small_group_uses_the_common_line():
    frame = biased_scorer()
    frame.loc[frame.index[:10], "group"] = "C"
    out = crossfit_calibration(frame, "group", min_n=30)
    assert out.index.equals(frame.index)
    assert out.between(1, 5).all()


def test_calibration_is_reproducible_and_never_uses_the_essay_it_scores():
    frame = biased_scorer(n=300)
    first = crossfit_calibration(frame, "group", seed=1)
    assert first.equals(crossfit_calibration(frame, "group", seed=1))
    assert not first.equals(crossfit_calibration(frame, "group", seed=2))
    changed = frame.copy()
    changed.loc[0, "truth"] = 5.0  # a different answer for one essay
    other = crossfit_calibration(changed, "group", seed=1)
    assert first.iloc[0] == other.iloc[0]
