"""Run the models over the corpus and keep the output on disk, so a long job can stop and resume."""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from scoreaudit.llm import Backbone, embed, next_token_probs
from scoreaudit.prompts import Variant, render


def shuffled(essays: pd.DataFrame, split: str, seed: int = 0) -> np.ndarray:
    """Essay ids of one split in a fixed random order, so that any prefix is a random subset."""
    ids = essays.loc[essays["split"] == split, "essay_id"].sort_values().to_numpy(dtype=str)
    return np.random.default_rng(seed).permutation(ids)


def priority_order(essays: pd.DataFrame, n_eval: int, seed: int = 0) -> np.ndarray:
    """Ids in the order they should be embedded: the evaluation essays first, then the training
    essays, then the rest of the test split."""
    test = shuffled(essays, "test", seed)
    train = shuffled(essays, "train", seed)
    return np.concatenate([test[:n_eval], train, test[n_eval:]])


def extract(
    backbone: Backbone,
    essays: pd.DataFrame,
    ids: Sequence[str],
    out_dir: Path,
    layers: Sequence[int],
    max_tokens: int = 640,
    chunk: int = 128,
    log: Callable[[str], None] = print,
) -> None:
    """Embed the essays `ids` by chunks of `chunk`. A chunk that is already on disk is skipped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    text = essays.set_index("essay_id")["text"]
    for start in range(0, len(ids), chunk):
        path = out_dir / f"chunk_{start:05d}.npz"
        if path.exists():
            continue
        part = np.asarray(ids[start : start + chunk], dtype=str)
        feats = embed(backbone, text.loc[part].tolist(), layers, max_tokens)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, ids=part, feats=feats)
        tmp.rename(path)
        log(f"{start + len(part)}/{len(ids)}")


def load_features(out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Ids and features of every chunk on disk, in the order they were written."""
    chunks = sorted(out_dir.glob("chunk_*.npz"))
    if not chunks:
        raise FileNotFoundError(f"no feature chunks in {out_dir}")
    loaded = [np.load(path) for path in chunks]
    return (
        np.concatenate([c["ids"] for c in loaded]),
        np.concatenate([c["feats"] for c in loaded]),
    )


def zero_shot(
    backbone: Backbone,
    variant: Variant,
    texts: Sequence[str],
    max_essay_tokens: int = 700,
    on_batch: Callable[[int], None] | None = None,
) -> np.ndarray:
    """Probability of each score under one prompt, one row per text."""
    prompts = [render(backbone.tokenizer, variant, t, max_essay_tokens) for t in texts]
    return next_token_probs(backbone, prompts, variant.options, on_batch=on_batch)
