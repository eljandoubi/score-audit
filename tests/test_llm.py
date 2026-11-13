import numpy as np
import pytest

from scoreaudit.llm import batches, embed, next_token_probs, option_ids
from scoreaudit.prompts import VARIANTS, cut, expected_score, render
from tests.tiny import backbone, tokenizer


def test_batches_cover_every_index_once_and_fit_the_budget():
    lengths = [5, 50, 7, 30, 31, 2, 49]
    groups = list(batches(lengths, budget=100))
    assert sorted(i for g in groups for i in g) == list(range(len(lengths)))
    for g in groups:
        assert max(lengths[i] for i in g) * len(g) <= 100 or len(g) == 1


def test_a_text_longer_than_the_budget_gets_its_own_batch():
    assert list(batches([500, 3, 4], budget=100)) == [[1, 2], [0]]


def test_embedding_does_not_depend_on_the_other_texts_in_the_batch():
    bb = backbone()
    texts = ["w1 w2 w3", "w4 w5 w6 w7 w8 w9 w10 w11", "w12 w13"]
    together = embed(bb, texts, layers=(1, 3))
    alone = np.vstack([embed(bb, [t], layers=(1, 3)) for t in texts])
    assert together.shape == (3, 2 * 16)
    assert np.allclose(together, alone, atol=1e-5)


def test_embedding_keeps_the_input_order_when_batches_are_reordered():
    bb = backbone()
    texts = ["w1 w2 w3 w4 w5 w6 w7 w8", "w1", "w2 w3 w4"]
    small_budget = embed(bb, texts, layers=(3,), budget=9)
    one_batch = embed(bb, texts, layers=(3,), budget=10_000)
    assert np.allclose(small_budget, one_batch, atol=1e-5)


def test_embedding_cuts_long_texts():
    bb = backbone()
    long = " ".join(f"w{i % 40}" for i in range(100))
    cut_text = " ".join(f"w{i % 40}" for i in range(20))
    assert np.allclose(
        embed(bb, [long], layers=(3,), max_tokens=20), embed(bb, [cut_text], layers=(3,)), atol=1e-5
    )


def test_embedding_rejects_a_layer_the_model_does_not_have():
    with pytest.raises(ValueError, match="between 0 and 3"):
        embed(backbone(), ["w1"], layers=(4,))


def test_options_must_be_single_tokens():
    tok = tokenizer()
    assert option_ids(tok, ["1", "5"]) == [
        tok.convert_tokens_to_ids("1"),
        tok.convert_tokens_to_ids("5"),
    ]
    with pytest.raises(ValueError, match="single token"):
        option_ids(tok, ["1 2"])


def test_option_probabilities_sum_to_one_and_ignore_padding():
    bb = backbone()
    prompts = ["w1 w2 answer", "w3 w4 w5 w6 w7 w8 answer", "w9 answer"]
    options = ("1", "2", "3", "4", "5")
    together = next_token_probs(bb, prompts, options)
    alone = np.vstack([next_token_probs(bb, [p], options) for p in prompts])
    assert np.allclose(together.sum(axis=1), 1, atol=1e-5)
    assert np.allclose(together, alone, atol=1e-4)


def test_expected_score_is_the_mean_under_the_probabilities():
    probs = np.array([[0, 0, 1, 0, 0], [0.5, 0, 0, 0, 0.5]])
    assert expected_score(probs).tolist() == [3.0, 3.0]


def test_every_variant_ends_with_the_generation_prompt_and_keeps_the_essay_tail():
    tok = tokenizer()
    for variant in VARIANTS.values():
        text = render(tok, variant, "w1 w2 w3")
        assert text.endswith("answer")
        assert "w1 w2 w3" in text
        assert len(variant.options) == 5


def test_long_essays_are_cut_before_they_go_in_the_prompt():
    tok = tokenizer()
    essay = " ".join(f"w{i % 40}" for i in range(60))
    assert cut(tok, essay, 10).split() == essay.split()[:10]
    assert cut(tok, "w1 w2", 10) == "w1 w2"
    prompt = render(tok, VARIANTS["rubric"], essay, max_essay_tokens=10)
    tail = prompt.split("Essay:")[-1].split()
    assert tail[:10] == [f"w{i}" for i in range(10)]
    assert tail[10:] == ["answer"]
