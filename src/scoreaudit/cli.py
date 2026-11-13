"""Command line: fetch the corpus, run a model over it."""

import argparse
import time
from pathlib import Path

import numpy as np

from scoreaudit.data import download, load_essays
from scoreaudit.features import extract, priority_order, shuffled, zero_shot
from scoreaudit.llm import load_backbone
from scoreaudit.prompts import VARIANTS

ROOT = Path("data/raw/ellipse")


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def cmd_download(args):
    download(ROOT)
    log(f"corpus in {ROOT}")


def cmd_features(args):
    essays = load_essays(ROOT)
    ids = priority_order(essays, args.n_eval, args.seed)
    if args.limit:
        ids = ids[: args.limit]
    backbone = load_backbone(args.model)
    log(f"{backbone.model.config.num_hidden_layers} layers on {backbone.device}")
    extract(backbone, essays, ids, Path(args.out), args.layers, args.max_tokens, args.chunk, log)
    log("done")


def score_variant(backbone, name, ids, text, target):
    seen = 0

    def progress(n):
        nonlocal seen
        seen += n
        if seen % 64 < n:
            log(f"{name} {seen}/{len(ids)}")

    probs = zero_shot(backbone, VARIANTS[name], text, on_batch=progress)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, ids=ids, probs=probs)
    log(f"{name} saved")


def cmd_zeroshot(args):
    essays = load_essays(ROOT)
    ids = shuffled(essays, "test", args.seed)[: args.n]
    text = essays.set_index("essay_id")["text"].loc[ids].tolist()
    backbone = load_backbone(args.model)
    for name in args.variants:
        target = Path(args.out) / f"{name}.npz"
        if not target.exists():
            score_variant(backbone, name, ids, text, target)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="scoreaudit")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="fetch the ELLIPSE corpus").set_defaults(run=cmd_download)

    feats = sub.add_parser("features", help="embed the essays with a causal language model")
    feats.add_argument("--model", required=True, help="folder or hub id of the model")
    feats.add_argument("--out", required=True)
    feats.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18, 24])
    feats.add_argument("--max-tokens", type=int, default=640)
    feats.add_argument("--n-eval", type=int, default=800, help="test essays embedded first")
    feats.add_argument("--chunk", type=int, default=128)
    feats.add_argument("--limit", type=int, default=0, help="stop after this many essays")
    feats.add_argument("--seed", type=int, default=0)
    feats.set_defaults(run=cmd_features)

    zero = sub.add_parser("zeroshot", help="score test essays from a rating prompt")
    zero.add_argument("--model", required=True)
    zero.add_argument("--out", required=True, help="folder, one file per prompt variant")
    zero.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    zero.add_argument("--n", type=int, default=400)
    zero.add_argument("--seed", type=int, default=0)
    zero.set_defaults(run=cmd_zeroshot)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
