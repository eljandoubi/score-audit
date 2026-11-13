import numpy as np
import pandas as pd
import pytest

from scoreaudit import pipeline
from scoreaudit.data import load_essays
from scoreaudit.features import extract, priority_order, shuffled
from scoreaudit.perturb import PERTURBATIONS
from tests.corpus import write_corpus, write_raw_zip
from tests.tiny import backbone


@pytest.fixture(scope="module")
def audit(tmp_path_factory):
    """A small fake corpus with the outputs a model run would leave, and the audit run on it."""
    root = tmp_path_factory.mktemp("corpus")
    cache = tmp_path_factory.mktemp("cache")
    out = tmp_path_factory.mktemp("results")
    write_raw_zip(root, write_corpus(root, n_train=150, n_test=100, seed=1))
    essays = load_essays(root)
    rng = np.random.default_rng(0)

    ids = priority_order(essays, n_eval=40)
    extract(
        backbone(), essays, ids, cache / "emb" / "fake", layers=(0, 1, 2, 3), log=lambda m: None
    )

    first = shuffled(essays, "test")[:40]
    for variant in ("letters", "rubric"):
        probs = rng.dirichlet(np.ones(5), 40)
        (cache / "zeroshot" / "fake").mkdir(parents=True, exist_ok=True)
        np.savez(cache / "zeroshot" / "fake" / f"{variant}.npz", ids=first, probs=probs)
    moved = cache / "perturbed" / "fake"
    moved.mkdir(parents=True)
    for name in PERTURBATIONS:
        np.savez(moved / f"emb_{name}.npz", ids=first[:30], feats=rng.normal(size=(30, 64)))
    np.savez(
        moved / "zs_letters_injection.npz", ids=first[:30], probs=rng.dirichlet(np.ones(5), 30)
    )

    messages = []
    pipeline.run(root, cache, out, models=("fake",), n_boot=10, say=messages.append)
    return out, messages


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
