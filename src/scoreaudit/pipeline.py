"""Run the whole audit from the corpus and the outputs the models left on disk."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scoreaudit.data import SCORES, load_essays, load_ratings
from scoreaudit.dif import dif_table
from scoreaudit.features import load_features
from scoreaudit.metrics import qwk
from scoreaudit.mitigate import crossfit_calibration
from scoreaudit.perturb import PERTURBATIONS
from scoreaudit.prompts import expected_score
from scoreaudit.raters import human_ceiling, leniency, leniency_spread, wide
from scoreaudit.routing import coverage_curve, ensemble_spread
from scoreaudit.scorers import FeatureRidge, MeanScorer, SurfaceGBM, TfidfRidge
from scoreaudit.study import (
    agreement_table,
    bland_altman_table,
    error_by_proficiency,
    fairness_tables,
    mitigation_table,
    pool_small,
    prompt_spread,
    psychometrics,
    robustness_table,
    sampling_reliability,
    scoring_frame,
)

ATTRIBUTES = ["gender", "race", "ses"]
LAYERS = (6, 12, 18, 24)
DIF_PAIRS = [
    ("gender", "Female", "Male"),
    ("ses", "disadvantaged", "not disadvantaged"),
    ("race", "Asian/PI", "Hispanic"),
    ("race", "Black", "Hispanic"),
    ("race", "White", "Hispanic"),
]

Say = Callable[[str], None]


@dataclass
class Corpus:
    essays: pd.DataFrame
    ratings: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def truth(self) -> pd.Series:
        return self.test.set_index("essay_id")["overall"]


def load_corpus(root: Path) -> Corpus:
    essays = load_essays(root)
    ratings = load_ratings(root, essays)
    train = essays[essays["split"] == "train"].reset_index(drop=True)
    test = essays[essays["split"] == "test"].reset_index(drop=True)
    return Corpus(essays, ratings, train, test)


def series(ids, values) -> pd.Series:
    return pd.Series(np.asarray(values), index=pd.Index(np.asarray(ids)))


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def save(table: pd.DataFrame, out: Path, name: str, index: bool = False) -> None:
    table.to_csv(out / f"{name}.csv", index=index, float_format="%.4f")


def fit_text_scorers(corpus: Corpus, say: Say) -> tuple[dict, dict[str, pd.Series]]:
    fitted, predictions = {}, {}
    for scorer in (MeanScorer(), SurfaceGBM(), TfidfRidge()):
        scorer.fit(corpus.train["text"].tolist(), corpus.train["overall"].to_numpy())
        fitted[scorer.name] = scorer
        pred = scorer.predict(corpus.test["text"].tolist())
        predictions[scorer.name] = series(corpus.test["essay_id"], pred)
        say(f"fitted {scorer.name}")
    return fitted, predictions


@dataclass
class Embedded:
    """A ridge regression on the pooled hidden states of a language model."""

    scorer: FeatureRidge
    columns: slice
    features: np.ndarray
    position: pd.Series  # essay id to row of `features`
    train_ids: np.ndarray
    test_ids: np.ndarray
    train_scores: np.ndarray

    def x(self, ids) -> np.ndarray:
        return self.features[self.position.loc[ids].to_numpy()][:, self.columns]

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.scorer.predict(features)

    def refit(self, weights: np.ndarray) -> pd.Series:
        model = FeatureRidge(self.scorer.name).fit(
            self.x(self.train_ids), self.train_scores, weights
        )
        return series(self.test_ids, model.predict(self.x(self.test_ids)))


def select_layers(features: np.ndarray, scores: np.ndarray, layers=LAYERS, seed: int = 0):
    """The block of columns that predicts best on a held-out fifth of the training essays."""
    width = features.shape[1] // len(layers)
    options = {f"layer {n}": slice(i * width, (i + 1) * width) for i, n in enumerate(layers)}
    options["all layers"] = slice(0, features.shape[1])
    order = np.random.default_rng(seed).permutation(len(scores))
    hold, fit = order[: len(order) // 5], order[len(order) // 5 :]
    rows = []
    for name, columns in options.items():
        model = FeatureRidge().fit(features[fit][:, columns], scores[fit])
        pred = np.clip(model.predict(features[hold][:, columns]), 1, 5)
        rows.append({"features": name, "qwk": qwk(pred, scores[hold])})
    table = pd.DataFrame(rows)
    return options[table.loc[table["qwk"].idxmax(), "features"]], table


def fit_embedded(key: str, folder: Path, corpus: Corpus, out: Path, say: Say) -> Embedded:
    ids, features = load_features(folder)
    position = pd.Series(np.arange(len(ids)), index=pd.Index(ids))
    train = corpus.train[corpus.train["essay_id"].isin(position.index)]
    test = corpus.test[corpus.test["essay_id"].isin(position.index)]
    train_ids, test_ids = train["essay_id"].to_numpy(), test["essay_id"].to_numpy()
    scores = train["overall"].to_numpy()
    columns, chosen = select_layers(features[position.loc[train_ids].to_numpy()], scores)
    save(chosen, out, f"layers_{key}")
    scorer = FeatureRidge(f"{key} ridge")
    scorer.fit(features[position.loc[train_ids].to_numpy()][:, columns], scores)
    say(f"{scorer.name}: {len(train_ids)} training and {len(test_ids)} test essays embedded")
    return Embedded(scorer, columns, features, position, train_ids, test_ids, scores)


def people_section(corpus: Corpus, out: Path, say: Say) -> None:
    ceiling = [{"score": s, **human_ceiling(corpus.ratings, s)} for s in SCORES]
    save(pd.DataFrame(ceiling), out, "human_ceiling")
    table = leniency(corpus.ratings)
    save(table, out, "rater_leniency", index=True)
    say(f"two raters agree at QWK {ceiling[0]['qwk']:.3f}")
    say(f"leniency spread of the raters {leniency_spread(table):.3f}")


def prompts_section(corpus: Corpus, folder: Path, key: str, out: Path) -> dict[str, pd.Series]:
    """Zero-shot scores of each prompt, and how far they depend on the prompt."""
    probs = {}
    for path in sorted(folder.glob("*.npz")):
        data = np.load(path)
        probs[f"{key} zero-shot {path.stem}"] = (data["ids"], data["probs"])
    scores = {name: series(ids, expected_score(p)) for name, (ids, p) in probs.items()}
    if scores:
        table, corr = prompt_spread(scores, corpus.truth)
        reliability = pd.DataFrame(
            [{"prompt": n, **sampling_reliability(p)} for n, (_, p) in probs.items()]
        )
        save(table.merge(reliability, on="prompt"), out, f"prompts_{key}")
        save(corr, out, f"prompt_correlation_{key}", index=True)
    return scores


def agreement_section(
    corpus: Corpus, predictions: dict, zero: dict, out: Path, n_boot: int, say: Say
) -> None:
    save(agreement_table(predictions, corpus.ratings, n_boot), out, "agreement_test")
    save(bland_altman_table(predictions, corpus.ratings), out, "bland_altman")
    if zero:
        ids = next(iter(zero.values())).index
        subset = {name: pred.reindex(ids).dropna() for name, pred in predictions.items()}
        for name, pred in zero.items():
            subset[name] = pred
            frame = scoring_frame(corpus.test, pred, [])
            subset[f"{name}, linear map"] = crossfit_calibration(frame)
        save(agreement_table(subset, corpus.ratings, n_boot), out, "agreement_subset")
    say("agreement done")


def fairness_section(corpus: Corpus, predictions: dict, out: Path, n_boot: int, say: Say) -> None:
    items, summary, theta = psychometrics(corpus.essays)
    save(items, out, "irt_items")
    save(pd.DataFrame([summary]), out, "irt_summary")
    test = corpus.test.assign(race=pool_small(corpus.test["race"]))
    for name, pred in predictions.items():
        if name == "mean":
            continue
        frame = scoring_frame(test, pred, ATTRIBUTES)
        for part, table in fairness_tables(frame, ATTRIBUTES, n_boot=min(n_boot, 300)).items():
            save(table.assign(scorer=name), out, f"fairness_{part}_{slug(name)}")
        by_band = error_by_proficiency(frame, theta)
        save(by_band.assign(scorer=name), out, f"proficiency_{slug(name)}")
    say("fairness done")


def mitigation_section(
    corpus: Corpus, embedded: Embedded, predictions: dict, out: Path, say: Say
) -> None:
    test = corpus.test.assign(race=pool_small(corpus.test["race"]))
    train = corpus.train.assign(race=pool_small(corpus.train["race"]))
    train = train.set_index("essay_id").loc[embedded.train_ids]
    frame = scoring_frame(test, predictions[embedded.scorer.name], ATTRIBUTES)
    rows = []
    for attribute in ATTRIBUTES:
        groups = train[attribute].fillna("missing").to_numpy()
        rows.append(
            mitigation_table(frame, attribute, groups, train["overall"].to_numpy(), embedded.refit)
        )
    save(pd.concat(rows), out, f"mitigation_{slug(embedded.scorer.name)}")
    say("mitigation done")


def dif_section(corpus: Corpus, out: Path, n_boot: int, say: Say) -> None:
    essays = corpus.essays.assign(race=pool_small(corpus.essays["race"]))
    tables = []
    for attribute, focal, reference in DIF_PAIRS:
        counts = essays[attribute].value_counts()
        if min(counts.get(focal, 0), counts.get(reference, 0)) < 30:
            continue
        table = dif_table(essays, attribute, focal, reference, n_boot=min(n_boot, 200))
        tables.append(table.assign(attribute=attribute))
    if tables:
        save(pd.concat(tables), out, "dif_traits")
    say("dif done")


def learned_robustness_section(
    corpus: Corpus, fitted: dict, embedded: Embedded, folder: Path, out: Path, n_boot: int, say: Say
) -> None:
    """Scores of changed copies of the first evaluation essays, against those of the originals."""
    names = [n for n in PERTURBATIONS if (folder / f"emb_{n}.npz").exists()]
    if not names:
        return
    ids = np.load(folder / f"emb_{names[0]}.npz")["ids"]
    texts = corpus.test.set_index("essay_id")["text"].loc[ids].tolist()
    expected = {n: p.expected for n, p in PERTURBATIONS.items()}
    tables = []
    for name, scorer in fitted.items():
        if name == "mean":
            continue
        before = series(ids, scorer.predict(texts))
        after = {
            n: series(ids, scorer.predict([p.apply(t) for t in texts]))
            for n, p in PERTURBATIONS.items()
        }
        table = robustness_table(before, after, corpus.truth, expected, n_boot)
        tables.append(table.assign(scorer=name))
    before = series(ids, embedded.predict(embedded.x(ids)))
    after = {
        n: series(
            ids, embedded.predict(np.load(folder / f"emb_{n}.npz")["feats"][:, embedded.columns])
        )
        for n in names
    }
    table = robustness_table(before, after, corpus.truth, expected, n_boot)
    tables.append(table.assign(scorer=embedded.scorer.name))
    save(pd.concat(tables), out, "robustness_learned")
    say("robustness of the learned scorers done")


def zero_robustness_section(
    corpus: Corpus, zero: dict, keys, cache: Path, out: Path, n_boot: int, say: Say
) -> None:
    """The same for the zero-shot scorers, on the changed copies that were scored with a prompt."""
    expected = {n: p.expected for n, p in PERTURBATIONS.items()}
    tables = []
    for key in keys:
        folder = cache / "perturbed" / key
        for scorer, base in zero.items():
            if not scorer.startswith(f"{key} zero-shot "):
                continue
            variant = scorer.removeprefix(f"{key} zero-shot ")
            moved = {}
            for path in folder.glob(f"zs_{variant}_*.npz"):
                data = np.load(path)
                name = path.stem.removeprefix(f"zs_{variant}_")
                moved[name] = series(data["ids"], expected_score(data["probs"]))
            if moved:
                ids = next(iter(moved.values())).index
                table = robustness_table(
                    base.reindex(ids).dropna(), moved, corpus.truth, expected, n_boot
                )
                tables.append(table.assign(scorer=scorer))
    if tables:
        save(pd.concat(tables), out, "robustness_zero_shot")
        say("robustness of the zero-shot scorers done")


def routing_section(corpus: Corpus, embedded: Embedded, out: Path, say: Say) -> None:
    """Send the essays on which bootstrap ridge models disagree most to the first rater."""
    pair = wide(corpus.ratings).reindex(embedded.test_ids).dropna()
    scaler, ridge = embedded.scorer.model[0], embedded.scorer.model[-1]
    train_x = scaler.transform(embedded.x(embedded.train_ids))
    test_x = scaler.transform(embedded.x(pair.index.to_numpy()))
    doubt = ensemble_spread(train_x, embedded.train_scores, test_x, ridge.alpha_)
    machine = embedded.predict(embedded.x(pair.index.to_numpy()))
    curve = coverage_curve(machine, pair["first"], pair["second"], doubt)
    save(curve.assign(scorer=embedded.scorer.name), out, f"routing_{slug(embedded.scorer.name)}")
    say("routing done")


def run(
    root: Path,
    cache: Path,
    out: Path,
    models: tuple[str, ...] = ("qwen0.5b",),
    n_boot: int = 500,
    say: Say = print,
) -> None:
    """Write every table of the audit into `out`.

    `cache` holds `emb/<model>` (pooled hidden states), `zeroshot/<model>` (answer probabilities of
    each prompt) and `perturbed/<model>` (the same for changed essays). Missing outputs are skipped.
    """
    out.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(root)
    say(f"{len(corpus.train)} training essays, {len(corpus.test)} test essays")
    people_section(corpus, out, say)
    fitted, predictions = fit_text_scorers(corpus, say)
    zero: dict[str, pd.Series] = {}
    embedded_models = {}
    for key in models:
        if (cache / "emb" / key).exists():
            embedded = fit_embedded(key, cache / "emb" / key, corpus, out, say)
            ids = embedded.test_ids
            predictions[embedded.scorer.name] = series(ids, embedded.predict(embedded.x(ids)))
            embedded_models[key] = embedded
        zero.update(prompts_section(corpus, cache / "zeroshot" / key, key, out))
    scores = pd.DataFrame(predictions).assign(truth=corpus.truth)
    save(scores.rename_axis("essay_id"), out, "predictions_test", index=True)
    if zero:
        save(pd.DataFrame(zero).rename_axis("essay_id"), out, "zero_shot_scores", index=True)
    agreement_section(corpus, predictions, zero, out, n_boot, say)
    fairness_section(corpus, predictions, out, n_boot, say)
    dif_section(corpus, out, n_boot, say)
    for key, embedded in embedded_models.items():
        mitigation_section(corpus, embedded, predictions, out, say)
        learned_robustness_section(
            corpus, fitted, embedded, cache / "perturbed" / key, out, n_boot, say
        )
        routing_section(corpus, embedded, out, say)
    zero_robustness_section(corpus, zero, models, cache, out, n_boot, say)
    say("done")
