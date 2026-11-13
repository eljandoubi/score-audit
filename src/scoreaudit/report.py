"""Turn the tables of an audit into a readable summary and a few figures."""

from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

Column = tuple[str, Callable[[pd.Series], str]]


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
        ("QWK >= 0.70", yes("meets_qwk")),
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
        section("Human raters", markdown(people, columns))

    for name, title, note in (
        ("agreement_test", "Agreement on the test split", ""),
        ("agreement_subset", "Agreement on the zero-shot subset", ""),
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
            f"Prompt sensitivity ({path.stem.removeprefix('prompts_')})", markdown(table, columns)
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
            section(f"Robustness of the {title}", markdown(table, columns))

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
        section("Agreement and bias by group", markdown(groups, columns))
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
        section("Impact ratios", markdown(impact, columns))
    for part, title in (
        ("error_model", "Error of the scorer regressed on the rater score and writer groups"),
        ("cleary_model", "Rater score regressed on the scorer's score and writer groups"),
    ):
        table = read(folder, f"fairness_{part}_*.csv")
        if table is not None:
            columns = [
                ("scorer", text("scorer")),
                ("term", text("term")),
                ("coefficient", ci("coef", "low", "high", 3)),
                ("p", num("p", 4)),
            ]
            section(title, markdown(table[table["term"].str.contains(" vs ")], columns))

    dif = read(folder, "dif_traits.csv")
    if dif is not None:
        columns = [
            ("group", lambda r: f"{r['focal']} vs {r['reference']}"),
            ("trait", text("item")),
            ("n focal", whole("n_focal")),
            ("DIF in points", ci("dif", "dif_low", "dif_high", 3)),
            ("DIF in SD", num("smd", 3)),
        ]
        section("Differential functioning of the human trait ratings", markdown(dif, columns))

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
        section("Bias across the range of proficiency", markdown(bands, columns))

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
        section("What narrowing the gaps costs", markdown(mitigation, columns))

    routing = read(folder, "routing_*.csv")
    if routing is not None:
        columns = [
            ("scorer", text("scorer")),
            ("share sent to a rater", num("fraction")),
            ("essays", whole("n_routed")),
            ("QWK, most doubtful first", num("qwk_routed", 3)),
            ("QWK, at random", num("qwk_random", 3)),
        ]
        section("Sending the doubtful essays to a person", markdown(routing, columns))
    return "\n".join(parts)


def figure_agreement(folder: Path) -> None:
    table = read(folder, "agreement_test.csv")
    humans = read(folder, "human_ceiling.csv")
    if table is None:
        return
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(table) + 1.4))
    y = np.arange(len(table))[::-1]
    est = table["qwk_mean"].to_numpy()
    err = [est - table["qwk_mean_low"].to_numpy(), table["qwk_mean_high"].to_numpy() - est]
    ax.barh(y, est, xerr=err, color="#4c72b0", height=0.6, capsize=3)
    ax.set_yticks(y, table["scorer"])
    if humans is not None:
        pair = humans.loc[humans["score"] == "overall", "qwk"].iloc[0]
        ax.axvline(pair, color="#c44e52", linestyle="--", label=f"two raters ({pair:.2f})")
    ax.axvline(0.70, color="grey", linestyle=":", label="0.70")
    ax.set_xlabel("quadratic weighted kappa with the mean of the two raters")
    ax.legend(loc="lower right")
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
