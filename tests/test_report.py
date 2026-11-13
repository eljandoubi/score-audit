import pandas as pd

from scoreaudit import report


def test_the_summary_has_a_section_for_each_table(audit):
    out, _ = audit
    text = report.summary(out)
    for title in (
        "Human raters",
        "Agreement on the test split",
        "Agreement on the zero-shot subset",
        "Prompt sensitivity (fake)",
        "Robustness of the learned scorers",
        "Robustness of the zero-shot scorers",
        "Agreement and bias by group",
        "Impact ratios",
        "Differential functioning of the human trait ratings",
        "Graded response model of the six traits",
        "Bias across the range of proficiency",
        "What narrowing the gaps costs",
        "Sending the doubtful essays to a person",
    ):
        assert f"## {title}" in text, title


def test_the_agreement_table_lists_the_scorers_with_their_intervals(audit):
    out, _ = audit
    text = report.summary(out)
    line = next(line for line in text.splitlines() if line.startswith("| tfidf-ridge | 100 |"))
    assert "[" in line and "]" in line


def test_build_writes_the_summary_and_the_figures(audit, tmp_path):
    out, _ = audit
    copy = tmp_path / "results"
    copy.mkdir()
    for path in out.glob("*.csv"):
        (copy / path.name).write_bytes(path.read_bytes())
    target = report.build(copy)
    assert target.read_text().startswith("# Audit results")
    figures = {p.name for p in (copy / "figures").glob("*.png")}
    assert {
        "agreement.png",
        "bias_by_proficiency.png",
        "routing.png",
        "zero_shot_scores.png",
    } <= figures
    assert any(name.startswith("group_bias_") for name in figures)
    assert any(name.startswith("bland_altman_") for name in figures)


def test_a_folder_without_tables_gives_an_empty_summary_and_no_figures(tmp_path):
    target = report.build(tmp_path)
    assert target.read_text().strip().startswith("# Audit results")
    assert list((tmp_path / "figures").glob("*.png")) == []


def test_formatters():
    row = pd.Series({"a": 0.5, "low": 0.25, "high": 0.75, "flag": True, "gap": None, "n": 12345})
    assert report.ci("a", "low", "high")(row) == "0.50 [0.25, 0.75]"
    assert report.num("a", 1)(row) == "0.5"
    assert report.num("gap")(row) == "-"
    assert report.yes("flag")(row) == "yes"
    assert report.whole("n")(row) == "12,345"
    assert report.yes("gap")(row) == "-"
