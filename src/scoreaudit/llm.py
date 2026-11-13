"""Run a causal language model over essays.

Two uses: pooled hidden states as features for a regression head, and zero-shot scores read from
the probabilities of the next token after a rating prompt.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Backbone:
    tokenizer: object
    model: torch.nn.Module
    device: str


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_backbone(path: str, device: str | None = None, dtype=torch.bfloat16) -> Backbone:
    device = device or pick_device()
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
    return Backbone(tokenizer, model, device)


def batches(lengths: Sequence[int], budget: int):
    """Groups of indices with similar length, each group holding at most `budget` padded tokens."""
    order = np.argsort(lengths, kind="stable")
    group: list[int] = []
    for i in order:
        if group and lengths[i] * (len(group) + 1) > budget:
            yield group
            group = []
        group.append(int(i))
    if group:
        yield group


def embed(
    backbone: Backbone,
    texts: Sequence[str],
    layers: Sequence[int],
    max_tokens: int = 768,
    budget: int = 6144,
    on_batch: Callable[[int], None] | None = None,
) -> np.ndarray:
    """Mean of the hidden states over the tokens of each text, for each of the given layers.

    `layers` index the tuple of hidden states, so 0 is the embedding layer and the number of
    layers of the model is the last one. The layers are concatenated: one row per text.
    """
    tok, model = backbone.tokenizer, backbone.model
    n_layers = model.config.num_hidden_layers
    if any(not 0 <= layer <= n_layers for layer in layers):
        raise ValueError(f"layers must be between 0 and {n_layers}")
    tok.padding_side = "right"
    ids = tok(list(texts), truncation=True, max_length=max_tokens, add_special_tokens=False)
    ids = ids["input_ids"]
    out = np.zeros((len(texts), len(layers) * model.config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for group in batches([len(x) for x in ids], budget):
            width = max(len(ids[i]) for i in group)
            batch = torch.full((len(group), width), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(group), width), dtype=torch.long)
            for row, i in enumerate(group):
                batch[row, : len(ids[i])] = torch.tensor(ids[i])
                mask[row, : len(ids[i])] = 1
            batch, mask = batch.to(backbone.device), mask.to(backbone.device)
            states = model.base_model(
                input_ids=batch, attention_mask=mask, output_hidden_states=True
            )
            weights = mask.unsqueeze(-1).to(torch.float32)
            pooled = [
                (states.hidden_states[layer].to(torch.float32) * weights).sum(1) / weights.sum(1)
                for layer in layers
            ]
            out[group] = torch.cat(pooled, dim=1).cpu().numpy()
            if on_batch:
                on_batch(len(group))
    return out


def option_ids(tokenizer, options: Sequence[str]) -> list[int]:
    """Token id of each answer option. Every option has to be a single token."""
    found = []
    for option in options:
        pieces = tokenizer.encode(option, add_special_tokens=False)
        if len(pieces) != 1:
            raise ValueError(f"{option!r} is not a single token")
        found.append(pieces[0])
    return found


def next_token_probs(
    backbone: Backbone,
    prompts: Sequence[str],
    options: Sequence[str],
    budget: int = 6144,
    on_batch: Callable[[int], None] | None = None,
) -> np.ndarray:
    """Probability of each option as the next token after each prompt, over the options only."""
    tok, model = backbone.tokenizer, backbone.model
    wanted = option_ids(tok, options)
    tok.padding_side = "left"
    ids = tok(list(prompts), add_special_tokens=False)["input_ids"]
    out = np.zeros((len(prompts), len(options)), dtype=np.float32)
    with torch.inference_mode():
        for group in batches([len(x) for x in ids], budget):
            width = max(len(ids[i]) for i in group)
            batch = torch.full((len(group), width), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(group), width), dtype=torch.long)
            for row, i in enumerate(group):
                batch[row, width - len(ids[i]) :] = torch.tensor(ids[i])
                mask[row, width - len(ids[i]) :] = 1
            batch, mask = batch.to(backbone.device), mask.to(backbone.device)
            positions = (mask.cumsum(-1) - 1).clamp(min=0)
            logits = model(
                input_ids=batch, attention_mask=mask, position_ids=positions, logits_to_keep=1
            ).logits[:, -1]
            probs = torch.softmax(logits[:, wanted].to(torch.float32), dim=-1)
            out[group] = probs.cpu().numpy()
            if on_batch:
                on_batch(len(group))
    return out
