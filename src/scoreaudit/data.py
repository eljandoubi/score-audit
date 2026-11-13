"""Read the ELLIPSE corpus (Crossley et al., 2023, CC BY 4.0).

The final corpus has 6,482 essays by English learners in grades 8 to 12. Each essay has an overall
score and six analytic scores, every one the mean of two raters on a 1 to 5 scale. A second
archive keeps the single ratings of those two raters.
"""

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

REPO = "https://raw.githubusercontent.com/scrosseye/ELLIPSE-Corpus"
COMMIT = "710963bbf1acf23a28e40643d9f937d06f047dd4"
FILES = {
    "ELLIPSE_Final_github_train.csv": (
        "782344e99668a3ff508d7410c0eb6e36da70f3b28f81c96e367f1ca04924b06c"
    ),
    "ELLIPSE_Final_github_test.zip": (
        "9ecddfcda83f6a99c8a24dda47fb1c30673fedeaf578a0fadac6645fd151a2f8"
    ),
    "ellipsis_raw_rater_scores_anon_all_essay.zip": (
        "f9c3cede2d54144225e0651af6415137f7da07f2d2955e8f8587896c293d5e85"
    ),
}
# The corpus authors publish the passwords of the two archives in their README.
TEST_PASSWORD = b"ellipse_test"
RAW_PASSWORD = b"ellipse_raw_data"

TRAITS = ("cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions")
SCORES = ("overall", *TRAITS)

_RENAME = {
    "text_id_kaggle": "essay_id",
    "full_text": "text",
    "race_ethnicity": "race",
    "SES": "ses",
    "set": "split",
    **{s.capitalize(): s for s in SCORES},
}
_RACE = {
    "Hispanic/Latino": "Hispanic",
    "Asian/Pacific Islander": "Asian/PI",
    "Black/African American": "Black",
    "White": "White",
    "Two or more races/Other": "Two+/Other",
    "American Indian/Alaskan Native": "AI/AN",
}
_SES = {
    "Economically disadvantaged": "disadvantaged",
    "Not economically disadvantaged": "not disadvantaged",
}
_COLUMNS = ["essay_id", "text", "split", "prompt", "gender", "race", "ses", "grade", *SCORES]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(root: Path) -> None:
    """Fetch the three files of the corpus at a fixed commit and check their hashes."""
    root.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        target = root / name
        if target.exists() and sha256(target) == digest:
            continue
        urllib.request.urlretrieve(f"{REPO}/{COMMIT}/{name}", target)
        if sha256(target) != digest:
            raise ValueError(f"{name} does not match the pinned hash")


def _csv_in_zip(path: Path, password: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        (name,) = [n for n in zf.namelist() if n.endswith(".csv")]
        with zf.open(name, pwd=password) as fh:
            return pd.read_csv(fh)


def load_essays(root: Path) -> pd.DataFrame:
    """One row per essay of the final corpus, with its split, writer group and scores."""
    train = pd.read_csv(root / "ELLIPSE_Final_github_train.csv")
    test = _csv_in_zip(root / "ELLIPSE_Final_github_test.zip", TEST_PASSWORD)
    essays = pd.concat([train, test], ignore_index=True).rename(columns=_RENAME)
    essays["race"] = essays["race"].map(_RACE)
    essays["ses"] = essays["ses"].map(_SES)
    return essays[_COLUMNS].reset_index(drop=True)


def load_ratings(root: Path, essays: pd.DataFrame) -> pd.DataFrame:
    """The single ratings behind the scores: two rows per essay, one for each rater.

    Ratings of essays that are not in the final corpus are dropped, and so is an essay that has a
    rating of 0, which the raters used for a text they could not score.
    """
    raw = _csv_in_zip(root / "ellipsis_raw_rater_scores_anon_all_essay.zip", RAW_PASSWORD)
    parts = []
    for slot in (1, 2):
        part = pd.DataFrame({"essay_id": raw["text_id_kaggle"], "slot": slot})
        part["rater"] = raw[f"Rater_{slot}"]
        for score in SCORES:
            part[score] = raw[f"{score.capitalize()}_{slot}"]
        parts.append(part)
    ratings = pd.concat(parts, ignore_index=True)
    ratings = ratings[ratings["essay_id"].isin(essays["essay_id"])]
    unusable = ratings.loc[(ratings[list(SCORES)] == 0).any(axis=1), "essay_id"]
    ratings = ratings[~ratings["essay_id"].isin(unusable)]
    return ratings.sort_values(["essay_id", "slot"]).reset_index(drop=True)
