import re

import pytest

from scoreaudit.perturb import (
    ORDER,
    PERTURBATIONS,
    inject_end,
    lowercase,
    repeat,
    shuffle_sentences,
    typos,
    whitespace,
)

ESSAY = (
    "Distance learning gives students a chance to study from home. Some people think it is "
    "lonely, but others enjoy the freedom. I believe that schools should offer both choices. "
    "Teachers can record lessons and answer questions later. Students who are shy might speak "
    "more in writing than in a crowded classroom."
)


def words(text):
    return re.findall(r"[A-Za-z]+", text)


def test_no_typos_at_rate_zero():
    assert typos(ESSAY, 0.0) == ESSAY


def test_typos_are_reproducible_and_depend_on_the_seed():
    assert typos(ESSAY, 0.3, seed=1) == typos(ESSAY, 0.3, seed=1)
    assert typos(ESSAY, 0.3, seed=1) != typos(ESSAY, 0.3, seed=2)


def test_typos_leave_short_words_and_first_letters_alone():
    changed = typos(ESSAY, 1.0, seed=4)
    before, after = words(ESSAY), words(changed)
    assert len(before) == len(after)
    for old, new in zip(before, after, strict=True):
        if len(old) < 4:
            assert new == old
        else:
            assert new[0] == old[0]
            assert abs(len(new) - len(old)) <= 1


def test_the_share_of_damaged_words_follows_the_rate():
    text = " ".join(["probably"] * 4000)
    damaged = sum(w != "probably" for w in typos(text, 0.05).split())
    assert damaged == pytest.approx(0.05 * 4000, rel=0.2)


def test_punctuation_and_spacing_survive_typos():
    changed = typos(ESSAY, 1.0, seed=2)
    assert re.sub(r"[A-Za-z]+", "w", changed) == re.sub(r"[A-Za-z]+", "w", ESSAY)


def test_a_slip_keeps_the_capital_of_a_neighbouring_key():
    for seed in range(30):
        out = typos("Teachers", 1.0, seed=seed)
        assert out[0] == "T"


def test_shuffle_keeps_every_sentence_and_changes_their_order():
    out = shuffle_sentences(ESSAY)
    split = re.split(r"(?<=[.!?])\s+", ESSAY)
    assert sorted(re.split(r"(?<=[.!?])\s+", out)) == sorted(split)
    assert out != ESSAY
    assert shuffle_sentences(ESSAY) == out


def test_injection_comes_after_the_essay_and_the_essay_is_untouched():
    out = inject_end(ESSAY)
    assert out.startswith(ESSAY)
    assert out.endswith(ORDER)


def test_repeat_doubles_the_words_and_lowercase_keeps_them():
    assert len(words(repeat(ESSAY))) == 2 * len(words(ESSAY))
    assert words(lowercase(ESSAY)) == [w.lower() for w in words(ESSAY)]


def test_whitespace_changes_only_spacing():
    assert whitespace(ESSAY).split() == ESSAY.split()
    assert whitespace(ESSAY) != ESSAY


def test_every_registered_perturbation_changes_the_text_and_says_what_to_expect():
    long_essay = ESSAY * 20  # at a 2% rate a short text can come out untouched
    for name, p in PERTURBATIONS.items():
        assert p.name == name
        assert p.expected in {"same", "lower"}
        assert p.apply(long_essay) != long_essay, name
