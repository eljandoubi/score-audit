"""Rating prompts for the zero-shot scorers.

Four wordings of the same task. They differ in where the rubric stands, in how the answer is
written and in how much the model is told, so the spread between them shows how much a score
depends on the prompt.
"""

from collections.abc import Callable
from dataclasses import dataclass

LEVELS = {
    5: "Very good control of English. Ideas are clear and well connected, the vocabulary is "
    "varied and precise, and grammar and punctuation are almost always correct.",
    4: "Good control of English. There are some errors or awkward phrases, but the meaning is "
    "always clear.",
    3: "Adequate control of English. Errors are frequent but rarely stop the reader from "
    "understanding. Vocabulary and sentences are plain.",
    2: "Limited control of English. Errors often make the meaning hard to follow. Vocabulary is "
    "narrow and sentences are simple or broken.",
    1: "Very limited control of English. The meaning is mostly unclear.",
}
LETTERS = {5: "A", 4: "B", 3: "C", 2: "D", 1: "E"}
VALUES = (1, 2, 3, 4, 5)

SYSTEM = "You are an experienced rater of writing by English language learners."


def _rubric() -> str:
    return "\n".join(f"{score}: {text}" for score, text in sorted(LEVELS.items(), reverse=True))


def _letter_rubric() -> str:
    return "\n".join(f"{LETTERS[s]}: {LEVELS[s]}" for s in sorted(LEVELS, reverse=True))


def _rubric_first(essay: str) -> list[dict]:
    task = (
        "Rate the overall English language proficiency shown in the essay below on a scale from "
        f"1 to 5.\n\n{_rubric()}\n\nAnswer with the number only.\n\nEssay:\n{essay}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]


def _essay_first(essay: str) -> list[dict]:
    task = (
        f"Essay:\n{essay}\n\nHow good is the English in this essay? Use this scale.\n\n"
        f"{_rubric()}\n\nAnswer with a single number from 1 to 5."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]


def _letters(essay: str) -> list[dict]:
    task = (
        "Give the essay below a grade for the overall English language proficiency of its "
        f"writer.\n\n{_letter_rubric()}\n\nAnswer with the letter only.\n\nEssay:\n{essay}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]


def _short(essay: str) -> list[dict]:
    task = (
        "Here is an essay by an English language learner. From 1 (poor) to 5 (excellent), how "
        f"good is its English? Answer with the number only.\n\n{essay}"
    )
    return [{"role": "user", "content": task}]


@dataclass(frozen=True)
class Variant:
    name: str
    options: tuple[str, ...]  # answer tokens, from the lowest score to the highest
    build: Callable[[str], list[dict]]


VARIANTS = {
    "rubric": Variant("rubric", ("1", "2", "3", "4", "5"), _rubric_first),
    "essay_first": Variant("essay_first", ("1", "2", "3", "4", "5"), _essay_first),
    "letters": Variant("letters", ("E", "D", "C", "B", "A"), _letters),
    "short": Variant("short", ("1", "2", "3", "4", "5"), _short),
}


def cut(tokenizer, essay: str, max_tokens: int) -> str:
    """Keep the start of an essay that is longer than `max_tokens`."""
    ids = tokenizer.encode(essay, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return essay
    return tokenizer.decode(ids[:max_tokens])


def render(tokenizer, variant: Variant, essay: str, max_essay_tokens: int = 700) -> str:
    messages = variant.build(cut(tokenizer, essay, max_essay_tokens))
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def expected_score(probs):
    """Mean score under the answer probabilities, a number between 1 and 5."""
    return probs @ VALUES
