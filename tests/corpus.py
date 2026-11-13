"""A tiny corpus with the same files and columns as ELLIPSE, for the tests."""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from scoreaudit.data import SCORES

CAPITAL = {s: s.capitalize() for s in SCORES}


def _essays(ids, split, rng):
    n = len(ids)
    frame = pd.DataFrame(
        {
            "text_id_kaggle": ids,
            "full_text": [
                " ".join(f"w{(k + j) % 40}" for j in range(int(rng.integers(8, 30))))
                for k in range(n)
            ],
            "gender": rng.choice(["Male", "Female"], n),
            "grade": rng.choice([8, 11], n),
            "race_ethnicity": rng.choice(["Hispanic/Latino", "Asian/Pacific Islander"], n),
            "task": "Independent",
            "SES": rng.choice(["Economically disadvantaged", "Not economically disadvantaged"], n),
            "prompt": rng.choice(["Distance learning", "Being busy"], n),
            "set": split,
        }
    )
    for score in SCORES:
        frame[CAPITAL[score]] = rng.integers(2, 10, n) / 2
    return frame


def write_corpus(root: Path, n_train=6, n_test=4, seed=0):
    """Write the three files. The archives have no password: zipfile cannot write one, and
    reading an archive without a password ignores the one it is given."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    ids = [f"ID{i:03d}" for i in range(n_train + n_test)]
    train = _essays(ids[:n_train], "train", rng)
    test = _essays(ids[n_train:], "test", rng)
    train.to_csv(root / "ELLIPSE_Final_github_train.csv", index=False)
    test.to_csv(root / "ELLIPSE_Final_github_test.csv", index=False)
    with zipfile.ZipFile(root / "ELLIPSE_Final_github_test.zip", "w") as zf:
        zf.write(root / "ELLIPSE_Final_github_test.csv", "ELLIPSE_Final_github_test.csv")

    everything = pd.concat([train, test], ignore_index=True)
    raw = pd.DataFrame({"Filename": everything["text_id_kaggle"] + ".txt"})
    raw["text_id_kaggle"] = everything["text_id_kaggle"]
    raw["Text"] = everything["full_text"]
    for slot in (1, 2):
        raw[f"Rater_{slot}"] = rng.choice(["rater_1", "rater_2", "rater_3"], len(raw))
        for score in SCORES:
            raw[f"{CAPITAL[score]}_{slot}"] = rng.integers(1, 6, len(raw))
        raw[f"Identifying_Info_{slot}"] = 0
    return raw


def write_raw_zip(root: Path, raw: pd.DataFrame):
    raw.to_csv(root / "ratings.csv", index=False)
    with zipfile.ZipFile(root / "ellipsis_raw_rater_scores_anon_all_essay.zip", "w") as zf:
        zf.write(root / "ratings.csv", "ratings.csv")
