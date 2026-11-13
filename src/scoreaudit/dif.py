"""Do raters score a trait differently for two groups of writers alike on the other traits?"""

import numpy as np
import pandas as pd

from scoreaudit.data import TRAITS


def _smd_dif(data: pd.DataFrame, item: str, by: str, focal: str, strata: int, min_cell: int):
    stratum = pd.qcut(data["match"], strata, duplicates="drop", labels=False)
    is_focal = data[by] == focal
    diffs, weights = [], []
    for k in np.unique(stratum):
        cell = stratum == k
        n_focal, n_ref = (cell & is_focal).sum(), (cell & ~is_focal).sum()
        if n_focal < min_cell or n_ref < min_cell:
            continue
        gap = data.loc[cell & is_focal, item].mean() - data.loc[cell & ~is_focal, item].mean()
        diffs.append(gap)
        weights.append(n_focal)
    if not weights:
        return float("nan")
    return float(np.average(diffs, weights=weights))


def smd_dif(
    frame: pd.DataFrame,
    item: str,
    by: str,
    focal: str,
    reference: str,
    rest: list[str] | None = None,
    strata: int = 8,
    min_cell: int = 5,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Differential item functioning by standardization (Dorans and Kulick).

    Essays are matched on the mean of the other traits and cut into `strata` bands. The score of
    the `item` trait is compared between the two groups inside each band, and the gaps are
    averaged with the number of focal-group essays as weights. A value away from zero says that
    the raters gave this trait a different score to the focal group at the same level of the
    rest. It is reported in points and in standard deviations of the trait.
    """
    rest = rest if rest is not None else [t for t in TRAITS if t != item]
    data = frame[[item, by, *rest]].dropna()
    data = data[data[by].isin([focal, reference])].copy()
    data["match"] = data[rest].mean(axis=1)
    points = _smd_dif(data, item, by, focal, strata, min_cell)
    rng = np.random.default_rng(seed)
    parts = [data[data[by] == focal], data[data[by] == reference]]
    draws = []
    for _ in range(n_boot):
        resampled = pd.concat([p.iloc[rng.integers(0, len(p), len(p))] for p in parts])
        draws.append(_smd_dif(resampled, item, by, focal, strata, min_cell))
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    spread = data[item].std(ddof=1)
    return {
        "item": item,
        "focal": focal,
        "reference": reference,
        "n_focal": len(parts[0]),
        "n_reference": len(parts[1]),
        "dif": points,
        "dif_low": float(low),
        "dif_high": float(high),
        "smd": points / spread,
    }


def dif_table(
    frame: pd.DataFrame, by: str, focal: str, reference: str, items=TRAITS, **kwargs
) -> pd.DataFrame:
    """The DIF of each trait for one focal group against a reference group."""
    return pd.DataFrame([smd_dif(frame, i, by, focal, reference, **kwargs) for i in items])
