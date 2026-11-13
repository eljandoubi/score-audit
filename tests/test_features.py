import numpy as np
import pytest

from scoreaudit import features
from scoreaudit.data import load_essays
from scoreaudit.features import extract, load_features, priority_order, shuffled, zero_shot
from scoreaudit.prompts import VARIANTS
from tests.corpus import write_corpus
from tests.tiny import backbone


@pytest.fixture
def essays(tmp_path):
    write_corpus(tmp_path, n_train=8, n_test=6)
    frame = load_essays(tmp_path)
    # the toy tokenizer only knows w0 to w39
    frame["text"] = [" ".join(f"w{(i + j) % 40}" for j in range(5 + i)) for i in range(len(frame))]
    return frame


def test_shuffle_is_stable_and_stays_inside_the_split(essays):
    a = shuffled(essays, "test", seed=1)
    assert a.tolist() == shuffled(essays.sample(frac=1, random_state=3), "test", seed=1).tolist()
    assert set(a) == set(essays.loc[essays["split"] == "test", "essay_id"])
    assert a.tolist() != shuffled(essays, "test", seed=2).tolist()


def test_priority_order_puts_the_evaluation_essays_first(essays):
    order = priority_order(essays, n_eval=3)
    test = shuffled(essays, "test")
    assert order[:3].tolist() == test[:3].tolist()
    assert set(order[3:11]) == set(essays.loc[essays["split"] == "train", "essay_id"])
    assert order[11:].tolist() == test[3:].tolist()
    assert len(set(order)) == len(essays)


def test_extract_writes_chunks_and_load_reads_them_back(essays, tmp_path):
    bb = backbone()
    ids = priority_order(essays, n_eval=2)
    extract(bb, essays, ids, tmp_path / "f", layers=(3,), chunk=5, log=lambda m: None)
    assert len(list((tmp_path / "f").glob("chunk_*.npz"))) == 3
    got_ids, got = load_features(tmp_path / "f")
    assert got_ids.tolist() == ids.tolist()
    assert got.shape == (len(ids), 16)


def test_extract_skips_chunks_that_exist(essays, tmp_path, monkeypatch):
    bb = backbone()
    ids = priority_order(essays, n_eval=2)
    extract(bb, essays, ids, tmp_path / "f", layers=(3,), chunk=5, log=lambda m: None)
    calls = []
    monkeypatch.setattr(features, "embed", lambda *a, **k: calls.append(1))
    extract(bb, essays, ids, tmp_path / "f", layers=(3,), chunk=5, log=lambda m: None)
    assert calls == []


def test_features_of_a_stopped_job_are_still_readable(essays, tmp_path):
    bb = backbone()
    ids = priority_order(essays, n_eval=2)
    extract(bb, essays, ids[:5], tmp_path / "f", layers=(3,), chunk=5, log=lambda m: None)
    got_ids, _ = load_features(tmp_path / "f")
    assert got_ids.tolist() == ids[:5].tolist()


def test_nothing_to_load_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_features(tmp_path)


def test_zero_shot_gives_one_distribution_per_text(essays):
    bb = backbone()
    probs = zero_shot(bb, VARIANTS["letters"], essays["text"].head(4).tolist())
    assert probs.shape == (4, 5)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)
