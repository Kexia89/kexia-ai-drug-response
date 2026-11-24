from __future__ import annotations

import numpy as np
import pandas as pd

from kexia_cancer.evaluation.phase2 import (
    _expression_scores, chemical_identity, group_bootstrap_ci, make_independent_splits,
)
from kexia_cancer.evaluation.temporal_validation import logistic


def _split_frame() -> pd.DataFrame:
    rows = []
    for cell in range(10):
        for drug in range(12):
            rows.append({
                "SANGER_MODEL_ID": f"SIDM{cell:03d}", "DRUG_ID": drug,
                "INCHI_KEY": f"ABCDEFGHIJKLMN-ABCDEFGHIJ-{chr(65 + drug % 12)}",
                "CHEMBL_ID": f"CHEMBL{drug}", "LN_IC50": float(cell + drug),
            })
    frame = pd.DataFrame(rows)
    frame["CHEMICAL_IDENTITY"] = chemical_identity(frame)
    return frame


def test_verified_chemical_identity_overrides_drug_id() -> None:
    frame = pd.DataFrame({
        "DRUG_ID": [1, 2], "CHEMBL_ID": ["CHEMBL25", "CHEMBL25"],
        "INCHI_KEY": ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"],
    })
    identities = chemical_identity(frame)
    assert identities.nunique() == 1
    assert identities.iloc[0].startswith("INCHIKEY:")


def test_compound_split_has_no_chemical_identity_overlap() -> None:
    frame = _split_frame()
    split = make_independent_splits(frame)["C_unseen_chemical_identities"]
    train = set(frame.iloc[split.train]["CHEMICAL_IDENTITY"])
    test = set(frame.iloc[split.test]["CHEMICAL_IDENTITY"])
    assert train.isdisjoint(test)


def test_expression_pca_uses_only_training_cells() -> None:
    expression = pd.DataFrame({
        "SANGER_MODEL_ID": ["A", "B", "C", "D"],
        "g1": [0.0, 1.0, 1000.0, 2000.0], "g2": [1.0, 0.0, -1000.0, -2000.0],
        "g3": [2.0, 3.0, 4.0, 5.0],
    })
    rows = pd.Series(["A", "B", "C", "D"])
    scores, metadata = _expression_scores(expression, {"A", "B"}, rows, components=1)
    assert metadata["training_cells_with_expression"] == 2
    assert scores.shape == (4, 1)


def test_group_bootstrap_is_reproducible() -> None:
    y = np.arange(12, dtype=float)
    prediction = y + np.tile([-.5, .5], 6)
    groups = np.repeat(["a", "b", "c"], 4)
    first = group_bootstrap_ci(y, prediction, groups, seed=7, replicates=20)
    second = group_bootstrap_ci(y, prediction, groups, seed=7, replicates=20)
    assert first == second


def test_logistic_temporal_model_is_nonnegative_and_bounded() -> None:
    time = np.linspace(0, 1000, 1001)
    values = logistic(time, carrying_capacity=1_000_000, growth_rate=.03, initial=1_000)
    assert np.all(values >= 0)
    assert np.all(values <= 1_000_000 * (1 + 1e-12))
    assert np.all(np.diff(values) >= 0)
