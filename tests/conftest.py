import numpy as np
import pytest

from scoreaudit import pipeline
from scoreaudit.data import load_essays
from scoreaudit.features import extract, priority_order, shuffled
from scoreaudit.perturb import PERTURBATIONS
from tests.corpus import write_corpus, write_raw_zip
from tests.tiny import backbone


@pytest.fixture(scope="session")
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
