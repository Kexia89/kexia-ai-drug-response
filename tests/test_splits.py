from __future__ import annotations

import pandas as pd

from kexia_cancer.models.splits import make_splits, validate_split


def grid() -> pd.DataFrame:
    return pd.DataFrame(
        [(f"C{cell}", drug) for cell in range(12) for drug in range(15)],
        columns=["SANGER_MODEL_ID", "DRUG_ID"],
    )


def test_group_split_invariants() -> None:
    data = grid()
    splits = make_splits(data, seed=7)
    b = validate_split(data, splits["B_unseen_cells"])
    c = validate_split(data, splits["C_unseen_compounds"])
    d = validate_split(data, splits["D_unseen_cells_and_compounds"])
    a = validate_split(data, splits["A_known_entities_new_pairs"])
    assert a["overlap_cells"] == a["test_cells"]
    assert a["overlap_drugs"] == a["test_drugs"]
    assert b["overlap_cells"] == 0
    assert c["overlap_drugs"] == 0
    assert d["overlap_cells"] == 0
    assert d["overlap_drugs"] == 0


def test_splits_are_reproducible() -> None:
    a, b = make_splits(grid(), seed=42), make_splits(grid(), seed=42)
    for name in a:
        assert (a[name].train == b[name].train).all()
        assert (a[name].test == b[name].test).all()

