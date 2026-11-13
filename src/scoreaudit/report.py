"""Turn the tables of an audit into a readable summary and a few figures."""

from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

Column = tuple[str, Callable[[pd.Series], str]]

NOTES = {
    "people": (
        "Each essay has two ratings by different raters. Their agreement is the level a scorer is "
        "asked to reach. The ICC of the mean of the two says how stable the mean score is."
    ),
    "agreement": (
        "QWK is the quadratic weighted kappa on half-point levels. 'vs rater mean' compares the "
        "rounded score with the mean of the two raters. 'vs one rater' compares it with each rater "
        "in turn, which is the figure to set against the two raters agreeing with each other; the "
        "loss is the second minus the first, so a negative loss means that the scorer agrees with "
        "a rater more than two raters do. The scorers that learn were trained on the mean of two "
        "ratings, which holds less noise than one rating: that explains part of such a gain and "
        "does not make them better than a person. The last three columns are the thresholds of "
        "Williamson, Xi and Breyer (2012), the first one applied to the agreement with one rater."
    ),
    "subset": (
        "The first essays of a seeded shuffle of the test split: the zero-shot scorers are run on "
        "a few hundred essays only, and the other scorers are shown on the same essays. A "
        "zero-shot score is the expected value of the probabilities the model gives to the "
        "answers 1 to 5 (E to A for the letters prompt). 'linear map' recalibrates that score to "
        "the rater scale with a line fitted on other essays than the one scored."
    ),
    "prompts": (
        "The same model with four wordings of the task. A scorer that did not depend on the "
        "wording would give the same mean and order the essays alike. 'ICC of one sampled run' is "
        "the share of the variance of one sampled answer that comes from the essays and not from "
        "the sampling, computed from the probabilities."
    ),
    "robustness": (
        "The same changes are applied to the first essays of the test split and the score is read "
        "again. typos: one typing slip in 2 or 5 percent of the words of four letters or more. "
        "lowercase: the text in lower case. shuffle: the sentences in a random order. whitespace: "
        "double spaces and blank lines. injection: an order to give the highest score, after the "
        "essay. repeat: the essay twice in a row. A change is as expected when the mean shift is "
        "at least 0.1 point down (lower), or within 0.1 point with a QWK of at least 0.90 "
        "between the scores before and after (same)."
    ),
    "groups": (
        "Race groups with fewer than 100 test essays are pooled as Other, and a group with "
        "fewer than 30 is left out of the tables. bias is the scorer's mean minus the raters' "
        "mean in the group: positive means that the scorer is more generous to the group than "
        "the raters. SMD is that gap over the spread of the two scores (Williamson, Xi and "
        "Breyer ask for an absolute value below 0.15)."
    ),
    "impact": (
        "Share of each group scoring at or above a cut, divided by the share of the best-placed "
        "group. A ratio below 0.8 breaks the four-fifths rule. 'ratio raters' is the same ratio "
        "computed on the raters' scores, which says whether the scorer made the gap or found it."
    ),
    "error": (
        "The scorer's score minus the raters' score, regressed on the raters' score and on the "
        "writer groups, with robust standard errors. A group coefficient is the difference in the "
        "average error at the same rater score, against the largest group. Only the group terms "
        "are shown."
    ),
    "cleary": (
        "The raters' score regressed on the scorer's score and on the writer groups (Cleary, "
        "1968). A group term away from zero would mean that the same scorer score predicts a "
        "different rater score depending on the group."
    ),
    "dif": (
        "Differential item functioning by standardization (Dorans and Kulick): essays are matched "
        "on the mean of the other five traits and cut into eight bands, and the gap in the trait "
        "between the two groups is averaged over the bands. This audits the human ratings that "
        "the scorers learn from, not a scorer."
    ),
    "bands": (
        "Bands of equal size of the proficiency estimated by the graded response model. A scorer "
        "that pulls every essay toward the middle is too generous at the low end and too harsh at "
        "the high end, which the overall bias hides."
    ),
    "mitigation": (
        "as fitted: the scorer as it is. reweighed training: training weights that make the group "
        "and the score level independent (Kamiran and Calders, 2012). common line: one line from "
        "the scorer's score to the rater scale, fitted on other folds. line for each group: one "
        "line per group, which needs the writer's group at scoring time. The gap in bias is the "
        "largest minus the smallest bias between groups."
    ),
    "routing": (
        "The essays on which an ensemble of ridge models, each fitted on a bootstrap sample, "
        "disagree most are sent to a person, here the first rater, and the second rater is the "
        "reference. 'At random' sends the same number of essays chosen by chance. A scorer trained "
        "on the mean of two raters agrees more with the second rater than the first rater does, "
        "so sending essays to the first rater lowers the QWK here: what the table can show is "
        "whether the doubt signal picks out the essays the scorer gets wrong, which is the gap "
        "between the two columns."
    ),
}


def ci(est: str, low: str, high: str, digits: int = 2) -> Callable[[pd.Series], str]:
    return lambda r: f"{r[est]:.{digits}f} [{r[low]:.{digits}f}, {r[high]:.{digits}f}]"


def num(name: str, digits: int = 2) -> Callable[[pd.Series], str]:
    return lambda r: "-" if pd.isna(r[name]) else f"{r[name]:.{digits}f}"


def text(name: str) -> Callable[[pd.Series], str]:
    return lambda r: str(r[name])


def whole(name: str) -> Callable[[pd.Series], str]:
    return lambda r: f"{int(r[name]):,}"


def yes(name: str) -> Callable[[pd.Series], str]:
    return lambda r: "-" if pd.isna(r[name]) else ("yes" if bool(r[name]) else "no")


def markdown(table: pd.DataFrame, columns: Sequence[Column]) -> str:
    head = "| " + " | ".join(h for h, _ in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    rows = ["| " + " | ".join(f(r) for _, f in columns) + " |" for _, r in table.iterrows()]
    return "\n".join([head, rule, *rows])


def read(folder: Path, pattern: str) -> pd.DataFrame | None:
    """All the tables that match the pattern, stacked, or None when there is none."""
    paths = sorted(folder.glob(pattern))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else None


def agreement_block(table: pd.DataFrame) -> str:
    columns = [
        ("scorer", text("scorer")),
        ("n", whole("n")),
        ("QWK vs rater mean", ci("qwk_mean", "qwk_mean_low", "qwk_mean_high")),
        ("QWK vs one rater", ci("qwk_single", "qwk_single_low", "qwk_single_high")),
        ("loss vs two raters", ci("loss", "loss_low", "loss_high")),
        ("exact", num("exact")),
        ("within 0.5", num("within_half")),
        ("bias", num("bias")),
        ("SMD", num("smd")),
        ("QWK one rater >= 0.70", yes("meets_qwk")),
        ("loss <= 0.10", yes("meets_loss")),
        ("|SMD| <= 0.15", yes("meets_smd")),
    ]
    return markdown(table, columns)


def summary(folder: Path) -> str:
    """The text of the summary, from the tables in `folder`."""
    parts = ["# Audit results", "", "Generated from the tables in this folder.", ""]

    def section(title: str, body: str | None, note: str = "") -> None:
        if body:
            parts.extend([f"## {title}", "", *([note, ""] if note else []), body, ""])

    people = read(folder, "human_ceiling.csv")
    if people is not None:
        columns = [
            ("score", text("score")),
            ("essays", whole("n")),
            ("QWK of the two raters", num("qwk", 3)),
            ("exact", num("exact")),
            ("within 1 point", num("within_1", 3)),
            ("ICC one rater", num("icc_single", 3)),
            ("ICC mean of two", num("icc_mean_of_two", 3)),
        ]
        section("Human raters", markdown(people, columns), NOTES["people"])

    for name, title, note in (
        ("agreement_test", "Agreement on the test split", NOTES["agreement"]),
        ("agreement_subset", "Agreement on the zero-shot subset", NOTES["subset"]),
    ):
        table = read(folder, f"{name}.csv")
        if table is not None:
            section(title, agreement_block(table), note)

    for path in sorted(folder.glob("prompts_*.csv")):
        table = pd.read_csv(path)
        columns = [
            ("prompt", text("prompt")),
            ("n", whole("n")),
            ("mean score", num("mean")),
            ("sd", num("sd")),
            ("Spearman vs raters", num("spearman")),
            ("QWK vs raters", num("qwk")),
            ("ICC of one sampled run", num("icc_one_run")),
        ]
        section(
            f"Prompt sensitivity ({path.stem.removeprefix('prompts_')})",
            markdown(table, columns),
            NOTES["prompts"],
        )

    for scope, title in (("learned", "learned scorers"), ("zero_shot", "zero-shot scorers")):
        table = read(folder, f"robustness_{scope}.csv")
        if table is not None:
            columns = [
                ("scorer", text("scorer")),
                ("change", text("change")),
                ("expected", text("expected")),
                ("mean shift", ci("mean_shift", "shift_low", "shift_high")),
                ("moved by 0.5 or more", num("moved_half_point")),
                ("QWK before/after", num("qwk_before_after")),
                ("as expected", yes("as_expected")),
            ]
            section(f"Robustness of the {title}", markdown(table, columns), NOTES["robustness"])

    groups = read(folder, "fairness_groups_*.csv")
    if groups is not None:
        columns = [
            ("scorer", text("scorer")),
            ("attribute", text("attribute")),
            ("group", text("group")),
            ("n", whole("n")),
            ("raters", num("human_mean")),
            ("scorer mean", num("machine_mean")),
            ("bias", ci("bias", "bias_low", "bias_high")),
            ("SMD", num("smd")),
            ("QWK", ci("qwk", "qwk_low", "qwk_high")),
        ]
        section("Agreement and bias by group", markdown(groups, columns), NOTES["groups"])
    impact = read(folder, "fairness_impact_*.csv")
    if impact is not None:
        columns = [
            ("scorer", text("scorer")),
            ("attribute", text("attribute")),
            ("group", text("group")),
            ("cut", num("cut", 1)),
            ("rate raters", num("rate_human")),
            ("rate scorer", num("rate_machine")),
            ("ratio raters", num("ratio_human")),
            ("ratio scorer", num("ratio_machine")),
            ("below 0.8", yes("adverse_machine")),
        ]
        section("Impact ratios", markdown(impact, columns), NOTES["impact"])
    for part, title in (
        ("error_model", "Error of the scorer regressed on the rater score and writer groups"),
        ("cleary_model", "Rater score regressed on the scorer's score and writer groups"),
    ):
        note = NOTES["error" if part == "error_model" else "cleary"]
        table = read(folder, f"fairness_{part}_*.csv")
        if table is not None:
            columns = [
                ("scorer", text("scorer")),
                ("term", text("term")),
                ("coefficient", ci("coef", "low", "high", 3)),
                ("p", num("p", 4)),
            ]
            section(title, markdown(table[table["term"].str.contains(" vs ")], columns), note)

    dif = read(folder, "dif_traits.csv")
    if dif is not None:
        columns = [
            ("group", lambda r: f"{r['focal']} vs {r['reference']}"),
            ("trait", text("item")),
            ("n focal", whole("n_focal")),
            ("DIF in points", ci("dif", "dif_low", "dif_high", 3)),
            ("DIF in SD", num("smd", 3)),
        ]
        section(
            "Differential functioning of the human trait ratings",
            markdown(dif, columns),
            NOTES["dif"],
        )

    items = read(folder, "irt_items.csv")
    if items is not None:
        steps = [c for c in items.columns if c.startswith("b")]
        columns = [("trait", text("item")), ("discrimination", num("discrimination"))]
        columns += [(c, num(c)) for c in steps]
        body = markdown(items, columns)
        overview = read(folder, "irt_summary.csv")
        if overview is not None:
            r = overview.iloc[0]
            body += (
                f"\n\nReliability of the proficiency estimates {r['reliability']:.3f}, "
                f"Cronbach alpha {r['alpha']:.3f}, on {int(r['n']):,} essays."
            )
        section("Graded response model of the six traits", body)

    bands = read(folder, "proficiency_*.csv")
    if bands is not None:
        columns = [
            ("scorer", text("scorer")),
            ("proficiency band", whole("band")),
            ("n", whole("n")),
            ("raters", num("human_mean")),
            ("scorer mean", num("machine_mean")),
            ("bias", ci("bias", "bias_low", "bias_high")),
            ("MAE", num("mae")),
        ]
        section("Bias across the range of proficiency", markdown(bands, columns), NOTES["bands"])

    mitigation = read(folder, "mitigation_*.csv")
    if mitigation is not None:
        columns = [
            ("attribute", text("attribute")),
            ("version", text("version")),
            ("QWK", ci("qwk", "qwk_low", "qwk_high")),
            ("gap in bias", num("bias_gap")),
            ("largest |SMD|", num("max_abs_smd")),
            ("lowest group QWK", num("worst_group_qwk")),
        ]
        section("What narrowing the gaps costs", markdown(mitigation, columns), NOTES["mitigation"])

    routing = read(folder, "routing_*.csv")
    if routing is not None:
        columns = [
            ("scorer", text("scorer")),
            ("share sent to a rater", num("fraction")),
            ("essays", whole("n_routed")),
            ("QWK, most doubtful first", num("qwk_routed", 3)),
            ("QWK, at random", num("qwk_random", 3)),
        ]
        section(
            "Sending the doubtful essays to a person", markdown(routing, columns), NOTES["routing"]
        )
    return "\n".join(parts)


def figure_agreement(folder: Path) -> None:
    table = read(folder, "agreement_test.csv")
    if table is None:
        return
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(table) + 1.4))
    y = np.arange(len(table))[::-1]
    est = table["qwk_single"].to_numpy()
    err = [est - table["qwk_single_low"].to_numpy(), table["qwk_single_high"].to_numpy() - est]
    ax.barh(y, est, xerr=err, color="#4c72b0", height=0.6, capsize=3)
    ax.set_yticks(y, table["scorer"])
    pair = table["qwk_humans"].iloc[0]
    ax.axvline(pair, color="#c44e52", linestyle="--", label=f"two raters ({pair:.2f})")
    ax.axvline(0.70, color="grey", linestyle=":", label="0.70")
    ax.set_xlabel("quadratic weighted kappa with one rater at a time")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(folder / "figures" / "agreement.png", dpi=150)
    plt.close(fig)


def figure_groups(folder: Path) -> None:
    table = read(folder, "fairness_groups_*.csv")
    if table is None:
        return
    for scorer, part in table.groupby("scorer"):
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(part) + 1.4))
        y = np.arange(len(part))[::-1]
        err = [
            part["bias"].to_numpy() - part["bias_low"].to_numpy(),
            part["bias_high"].to_numpy() - part["bias"].to_numpy(),
        ]
        ax.errorbar(part["bias"], y, xerr=err, fmt="o", color="#4c72b0", capsize=3)
        ax.axvline(0, color="grey")
        ax.set_yticks(
            y,
            [
                f"{a}: {g} (n={n})"
                for a, g, n in zip(part["attribute"], part["group"], part["n"], strict=True)
            ],
        )
        ax.set_xlabel("mean of scorer minus raters (points), 95% interval")
        ax.set_title(str(scorer))
        fig.tight_layout()
        name = "".join(c if c.isalnum() else "_" for c in str(scorer).lower())
        fig.savefig(folder / "figures" / f"group_bias_{name}.png", dpi=150)
        plt.close(fig)


def figure_proficiency(folder: Path) -> None:
    table = read(folder, "proficiency_*.csv")
    if table is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for scorer, part in table.groupby("scorer"):
        ax.plot(part["band"], part["bias"], marker="o", label=str(scorer))
    ax.axhline(0, color="grey")
    ax.set_xlabel("proficiency band (1 lowest)")
    ax.set_ylabel("mean of scorer minus raters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "figures" / "bias_by_proficiency.png", dpi=150)
    plt.close(fig)


def figure_routing(folder: Path) -> None:
    table = read(folder, "routing_*.csv")
    if table is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for scorer, part in table.groupby("scorer"):
        ax.plot(
            part["fraction"], part["qwk_routed"], marker="o", label=f"{scorer}, most doubtful first"
        )
        ax.plot(part["fraction"], part["qwk_random"], linestyle="--", label=f"{scorer}, at random")
    ax.set_xlabel("share of essays scored by the first rater")
    ax.set_ylabel("QWK with the second rater")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "figures" / "routing.png", dpi=150)
    plt.close(fig)


def figure_bland_altman(folder: Path) -> None:
    path = folder / "predictions_test.csv"
    if not path.exists():
        return
    table = pd.read_csv(path, index_col=0)
    for scorer in [c for c in table.columns if c not in ("truth", "mean")]:
        pred, truth = table[scorer].clip(1, 5), table["truth"]
        diff, mid = pred - truth, (pred + truth) / 2
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.scatter(mid, diff, s=6, alpha=0.3, color="#4c72b0")
        bias, spread = diff.mean(), 1.96 * diff.std(ddof=1)
        for level, style in ((bias, "-"), (bias - spread, "--"), (bias + spread, "--")):
            ax.axhline(level, color="#c44e52", linestyle=style)
        ax.set_xlabel("mean of scorer and rater mean")
        ax.set_ylabel("scorer minus rater mean")
        ax.set_title(
            f"{scorer}: bias {bias:.2f}, limits {bias - spread:.2f} to {bias + spread:.2f}"
        )
        fig.tight_layout()
        name = "".join(c if c.isalnum() else "_" for c in scorer.lower())
        fig.savefig(folder / "figures" / f"bland_altman_{name}.png", dpi=150)
        plt.close(fig)


def figure_prompts(folder: Path) -> None:
    path = folder / "zero_shot_scores.csv"
    if not path.exists():
        return
    table = pd.read_csv(path, index_col=0)
    fig, axes = plt.subplots(
        1, len(table.columns), figsize=(3 * len(table.columns), 3), sharey=True
    )
    for ax, column in zip(np.atleast_1d(axes), table.columns, strict=True):
        ax.hist(table[column].dropna(), bins=np.arange(1, 5.26, 0.25), color="#4c72b0")
        ax.set_title(column.replace(" zero-shot ", "\n"), fontsize=8)
        ax.set_xlabel("expected score")
    fig.tight_layout()
    fig.savefig(folder / "figures" / "zero_shot_scores.png", dpi=150)
    plt.close(fig)


def build(folder: Path) -> Path:
    """Write `SUMMARY.md` and the figures next to the tables."""
    (folder / "figures").mkdir(exist_ok=True)
    target = folder / "SUMMARY.md"
    target.write_text(summary(folder) + "\n")
    draw = (
        figure_agreement,
        figure_groups,
        figure_proficiency,
        figure_routing,
        figure_bland_altman,
        figure_prompts,
    )
    for figure in draw:
        figure(folder)
    return target
