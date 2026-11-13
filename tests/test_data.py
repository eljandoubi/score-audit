from pathlib import Path

import numpy as np
import pytest

from scoreaudit import data
from scoreaudit.data import SCORES, load_essays, load_ratings
from tests.corpus import write_corpus, write_raw_zip

REAL = Path("data/raw/ellipse")


@pytest.fixture
def corpus(tmp_path):
    raw = write_corpus(tmp_path)
    write_raw_zip(tmp_path, raw)
    return tmp_path, raw


def test_essays_come_with_both_splits(corpus):
    root, _ = corpus
    essays = load_essays(root)
    assert len(essays) == 10
    assert essays["split"].value_counts().to_dict() == {"train": 6, "test": 4}
    assert set(essays["race"]) <= {"Hispanic", "Asian/PI"}
    assert set(essays["ses"]) <= {"disadvantaged", "not disadvantaged"}
    assert set(SCORES) <= set(essays.columns)


def test_two_ratings_per_essay(corpus):
    root, _ = corpus
    essays = load_essays(root)
    ratings = load_ratings(root, essays)
    assert len(ratings) == 20
    assert (ratings.groupby("essay_id").size() == 2).all()
    assert ratings.groupby("essay_id")["slot"].apply(list).map(sorted).eq([[1, 2]] * 10).all()


def test_an_essay_with_a_zero_rating_is_dropped(corpus):
    root, raw = corpus
    raw.loc[3, "Grammar_2"] = 0
    write_raw_zip(root, raw)
    essays = load_essays(root)
    ratings = load_ratings(root, essays)
    assert raw.loc[3, "text_id_kaggle"] not in set(ratings["essay_id"])
    assert ratings["essay_id"].nunique() == 9


def test_ratings_of_unknown_essays_are_ignored(corpus):
    root, raw = corpus
    raw.loc[0, "text_id_kaggle"] = "NOT-IN-THE-CORPUS"
    write_raw_zip(root, raw)
    ratings = load_ratings(root, load_essays(root))
    assert "NOT-IN-THE-CORPUS" not in set(ratings["essay_id"])


def test_download_checks_the_hash(tmp_path, monkeypatch):
    def fake_urlretrieve(url, target):
        Path(target).write_bytes(b"not the corpus")

    monkeypatch.setattr(data.urllib.request, "urlretrieve", fake_urlretrieve)
    with pytest.raises(ValueError, match="pinned hash"):
        data.download(tmp_path)


def test_download_skips_a_file_that_is_already_there(tmp_path, monkeypatch):
    content = b"kept"
    monkeypatch.setattr(data, "FILES", {"a.csv": data.hashlib.sha256(content).hexdigest()})
    (tmp_path / "a.csv").write_bytes(content)

    def refuse(url, target):
        raise AssertionError("downloaded again")

    monkeypatch.setattr(data.urllib.request, "urlretrieve", refuse)
    data.download(tmp_path)


@pytest.mark.skipif(not REAL.exists(), reason="the corpus is not downloaded")
def test_real_corpus_sizes():
    essays = load_essays(REAL)
    ratings = load_ratings(REAL, essays)
    assert len(essays) == 6482
    assert essays["split"].value_counts().to_dict() == {"train": 3911, "test": 2571}
    assert (ratings.groupby("essay_id").size() == 2).all()
    both = ratings.groupby("essay_id")["overall"].mean()
    ref = essays.set_index("essay_id").loc[both.index, "overall"]
    assert np.isclose(both, ref).mean() > 0.99
