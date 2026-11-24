from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    name: str
    train: np.ndarray
    test: np.ndarray


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_splits(df: pd.DataFrame, seed: int = 42, fraction: float = 0.2) -> dict[str, Split]:
    """Create four scientifically distinct evaluation protocols."""
    n = len(df)
    indices = np.arange(n)
    rng = _rng(seed)

    # A: held-out pairs, while requiring every test cell and drug to occur in train.
    shuffled = rng.permutation(indices)
    proposed_test = shuffled[: int(fraction * n)]
    proposed_train = shuffled[int(fraction * n) :]
    train_cells = set(df.iloc[proposed_train]["SANGER_MODEL_ID"])
    train_drugs = set(df.iloc[proposed_train]["DRUG_ID"])
    valid = df.iloc[proposed_test]["SANGER_MODEL_ID"].isin(train_cells) & df.iloc[proposed_test]["DRUG_ID"].isin(train_drugs)
    a_test = proposed_test[valid.to_numpy()]
    a_train = np.setdiff1d(indices, a_test)

    # B: cell-line holdout.
    cells = np.array(sorted(df["SANGER_MODEL_ID"].dropna().unique()))
    held_cells = set(rng.choice(cells, max(1, int(fraction * len(cells))), replace=False))
    b_mask = df["SANGER_MODEL_ID"].isin(held_cells).to_numpy()

    # C: compound holdout.
    drugs = np.array(sorted(df["DRUG_ID"].dropna().unique()))
    held_drugs = set(rng.choice(drugs, max(1, int(fraction * len(drugs))), replace=False))
    c_mask = df["DRUG_ID"].isin(held_drugs).to_numpy()

    # D: train excludes both held entity sets; test is their intersection.
    d_test = b_mask & c_mask
    d_train = (~b_mask) & (~c_mask)
    return {
        "A_known_entities_new_pairs": Split("A_known_entities_new_pairs", a_train, a_test),
        "B_unseen_cells": Split("B_unseen_cells", indices[~b_mask], indices[b_mask]),
        "C_unseen_compounds": Split("C_unseen_compounds", indices[~c_mask], indices[c_mask]),
        "D_unseen_cells_and_compounds": Split("D_unseen_cells_and_compounds", indices[d_train], indices[d_test]),
    }


def validate_split(df: pd.DataFrame, split: Split) -> dict:
    train, test = df.iloc[split.train], df.iloc[split.test]
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_cells": train["SANGER_MODEL_ID"].nunique(),
        "test_cells": test["SANGER_MODEL_ID"].nunique(),
        "overlap_cells": len(set(train["SANGER_MODEL_ID"]) & set(test["SANGER_MODEL_ID"])),
        "train_drugs": train["DRUG_ID"].nunique(),
        "test_drugs": test["DRUG_ID"].nunique(),
        "overlap_drugs": len(set(train["DRUG_ID"]) & set(test["DRUG_ID"])),
    }

