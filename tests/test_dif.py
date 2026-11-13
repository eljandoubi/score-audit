import numpy as np
import pandas as pd
import pytest

from scoreaudit.data import TRAITS
from scoreaudit.dif import dif_table, smd_dif


def traits_with_dif(n=3000, shift=0.4, item="grammar", seed=0):
    rng = np.random.default_rng(seed)
    group = np.where(np.arange(n) % 3 == 0, "F", "M")
    ability = rng.normal(0, 1, n)
    frame = pd.DataFrame({"group": group})
    for trait in TRAITS:
        frame[trait] = 3 + 0.8 * ability + rng.normal(0, 0.4, n)
    frame[item] += np.where(group == "F", shift, 0.0)
    return frame


def test_dif_finds_a_trait_that_is_scored_higher_for_one_group_at_the_same_level():
    row = smd_dif(traits_with_dif(), "grammar", "group", "F", "M", n_boot=50)
    assert row["dif"] == pytest.approx(0.4, abs=0.08)
    assert row["dif_low"] > 0.2
    assert row["smd"] > 0.3
    assert row["n_focal"] == 1000
    assert row["n_reference"] == 2000


def test_dif_is_near_zero_for_a_trait_without_a_shift():
    clean = ["cohesion", "vocabulary", "phraseology", "conventions"]
    row = smd_dif(traits_with_dif(), "syntax", "group", "F", "M", rest=clean, n_boot=50)
    assert abs(row["dif"]) < 0.08


def test_matching_on_a_shifted_trait_makes_a_clean_one_look_biased():
    row = smd_dif(traits_with_dif(), "syntax", "group", "F", "M", n_boot=50)
    assert row["dif"] < -0.05


def test_the_same_gap_in_ability_alone_is_not_dif():
    rng = np.random.default_rng(1)
    n = 4000
    group = np.where(np.arange(n) % 2 == 0, "F", "M")
    ability = rng.normal(0, 1, n) + np.where(group == "F", 1.0, 0.0)
    frame = pd.DataFrame({"group": group})
    for trait in ("cohesion", "syntax", "grammar"):
        frame[trait] = 3 + 0.8 * ability + rng.normal(0, 0.3, n)
    raw_gap = (
        frame.loc[frame.group == "F", "grammar"].mean()
        - frame.loc[frame.group == "M", "grammar"].mean()
    )
    row = smd_dif(frame, "grammar", "group", "F", "M", rest=["cohesion", "syntax"], n_boot=50)
    assert raw_gap > 0.7
    assert abs(row["dif"]) < 0.15


def test_dif_table_has_one_row_per_trait():
    table = dif_table(traits_with_dif(n=900), "group", "F", "M", items=TRAITS[:4], n_boot=20)
    assert table["item"].tolist() == list(TRAITS[:4])


def test_a_focal_group_too_small_to_fill_any_cell_gives_no_estimate():
    frame = traits_with_dif(n=300)
    few = pd.concat([frame[frame.group == "F"].head(3), frame[frame.group == "M"]])
    row = smd_dif(few, "grammar", "group", "F", "M", n_boot=10)
    assert np.isnan(row["dif"])
