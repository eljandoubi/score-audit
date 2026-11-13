import numpy as np
import pandas as pd
import pytest

from scoreaudit.raters import against_raters, human_ceiling, leniency, leniency_spread, wide


def ratings_from(first, second, raters=("a", "b")):
    rows = []
    for i, (x, y) in enumerate(zip(first, second, strict=True)):
        rows.append({"essay_id": f"e{i}", "slot": 1, "rater": raters[0], "overall": x})
        rows.append({"essay_id": f"e{i}", "slot": 2, "rater": raters[1], "overall": y})
    return pd.DataFrame(rows)


def test_wide_has_one_row_per_essay():
    table = wide(ratings_from([1, 2, 3], [3, 2, 1]))
    assert table.columns.tolist() == ["first", "second"]
    assert table["first"].tolist() == [1, 2, 3]
    assert table["second"].tolist() == [3, 2, 1]


def test_raters_who_always_agree_reach_the_top_of_every_measure():
    ceiling = human_ceiling(ratings_from([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]))
    assert ceiling["qwk"] == 1.0
    assert ceiling["exact"] == 1.0
    assert ceiling["icc_single"] == pytest.approx(1.0)


def test_the_mean_of_two_raters_is_more_reliable_than_one():
    rng = np.random.default_rng(0)
    truth = rng.integers(1, 6, 400)
    first = np.clip(truth + rng.integers(-1, 2, 400), 1, 5)
    second = np.clip(truth + rng.integers(-1, 2, 400), 1, 5)
    ceiling = human_ceiling(ratings_from(first, second))
    assert ceiling["icc_mean_of_two"] > ceiling["icc_single"]
    assert ceiling["within_1"] > ceiling["exact"]


def test_leniency_is_the_gap_to_the_other_rater():
    table = leniency(ratings_from([4, 3, 5, 4], [3, 2, 4, 3]))
    assert table.loc["a", "leniency"] == 1.0
    assert table.loc["b", "leniency"] == -1.0
    assert table.loc["a", "n"] == 4


def test_leniency_counts_a_rater_in_both_slots():
    first = ratings_from([3, 3], [3, 3], raters=("a", "b"))
    second = ratings_from([3, 3], [3, 3], raters=("b", "a")).assign(
        essay_id=lambda d: d["essay_id"] + "x"
    )
    table = leniency(pd.concat([first, second]))
    assert table.loc["a", "n"] == 4
    assert table.loc["b", "n"] == 4


def test_leniency_spread_ignores_raters_with_few_essays():
    table = pd.DataFrame({"n": [500, 400, 5], "leniency": [-0.2, 0.2, 3.0]}, index=["a", "b", "c"])
    assert leniency_spread(table, min_n=100) == pytest.approx(np.std([-0.2, 0.2], ddof=1))


def test_a_scorer_that_gives_the_mean_matches_the_expert_average_but_not_each_expert():
    first, second = [1, 2, 3, 4, 5, 3, 2, 4], [2, 2, 4, 3, 5, 4, 1, 4]
    ratings = ratings_from(first, second)
    mean = pd.Series((np.array(first) + second) / 2, index=[f"e{i}" for i in range(8)])
    result = against_raters(mean, ratings)
    assert result["n"] == 8
    assert result["qwk_mean"] == 1.0
    assert result["qwk_single"] < 1.0
    assert result["degradation"] == pytest.approx(result["qwk_humans"] - result["qwk_single"])


def test_scores_outside_the_scale_are_clipped_and_missing_ones_skipped():
    ratings = ratings_from([1, 3, 5], [1, 3, 5])
    pred = pd.Series([0.2, 3.0, np.nan], index=["e0", "e1", "e2"])
    assert against_raters(pred, ratings)["n"] == 2
