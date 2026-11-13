"""A tiny random model and a word-level tokenizer, so the tests need no download."""

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from scoreaudit.llm import Backbone

WORDS = ["<pad>", "<unk>", "1", "2", "3", "4", "5", "A", "B", "C", "D", "E"] + [
    f"w{i}" for i in range(40)
]
TEMPLATE = (
    "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    "{% if add_generation_prompt %}answer{% endif %}"
)


def tokenizer():
    base = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=base, pad_token="<pad>", unk_token="<unk>")
    fast.chat_template = TEMPLATE
    return fast


def backbone(layers=3, seed=0):
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=len(WORDS),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    model = Qwen2ForCausalLM(config).eval()
    return Backbone(tokenizer(), model, "cpu")
