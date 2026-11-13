import pandas as pd

from scoreaudit.perturb import PERTURBATIONS


def names(folder):
    return {p.name for p in folder.glob("*.csv")}


def test_every_section_writes_its_tables(audit):
    out, messages = audit
    expected = {
        "human_ceiling.csv", "rater_leniency.csv", "agreement_test.csv", "agreement_subset.csv",
        "bland_altman.csv", "irt_items.csv", "irt_summary.csv", "layers_fake.csv",
        "prompts_fake.csv", "prompt_correlation_fake.csv", "robustness_learned.csv",
        "robustness_zero_shot.csv", "routing_fake_ridge.csv", "mitigation_fake_ridge.csv",
        "proficiency_tfidf_ridge.csv", "fairness_groups_fake_ridge.csv", "dif_traits.csv",
        "predictions_test.csv", "zero_shot_scores.csv",
    }  # fmt: skip
    assert expected <= names(out)
    assert messages[-1] == "done"


def test_the_agreement_table_has_every_scorer_on_every_test_essay(audit):
    out, _ = audit
    table = pd.read_csv(out / "agreement_test.csv")
    assert table["scorer"].tolist() == ["mean", "surface-gbm", "tfidf-ridge", "fake ridge"]
    assert table["n"].tolist() == [100, 100, 100, 100]


def test_the_subset_table_adds_the_prompts_and_their_linear_maps(audit):
    out, _ = audit
    table = pd.read_csv(out / "agreement_subset.csv")
    scorers = set(table["scorer"])
    assert {"fake zero-shot letters", "fake zero-shot letters, linear map"} <= scorers
    assert set(table["n"]) == {40}


def test_the_four_versions_of_the_scorer_are_compared_for_each_attribute(audit):
    out, _ = audit
    table = pd.read_csv(out / "mitigation_fake_ridge.csv")
    assert set(table["attribute"]) == {"gender", "race", "ses"}
    assert table["version"].nunique() == 4


def test_robustness_covers_every_change_for_the_learned_scorers(audit):
    out, _ = audit
    table = pd.read_csv(out / "robustness_learned.csv")
    assert set(table["change"]) == set(PERTURBATIONS)
    assert set(table["scorer"]) == {"surface-gbm", "tfidf-ridge", "fake ridge"}
    zero = pd.read_csv(out / "robustness_zero_shot.csv")
    assert zero["scorer"].tolist() == ["fake zero-shot letters"]
    assert zero["change"].tolist() == ["injection"]


def test_the_routing_curve_runs_from_the_machine_to_the_reviewer(audit):
    out, _ = audit
    curve = pd.read_csv(out / "routing_fake_ridge.csv")
    assert curve["fraction"].iloc[0] == 0.0
    assert curve["fraction"].iloc[-1] == 1.0
    assert curve["n_routed"].iloc[-1] > 0


def test_groups_too_small_for_a_dif_comparison_are_skipped(audit):
    out, _ = audit
    table = pd.read_csv(out / "dif_traits.csv")
    assert set(table["attribute"]) <= {"gender", "ses", "race"}
    assert "Black" not in set(table["focal"])
