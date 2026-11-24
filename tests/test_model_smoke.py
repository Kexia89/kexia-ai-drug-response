from __future__ import annotations

import numpy as np
import pandas as pd

from kexia_cancer.models.baselines import make_pipeline


def test_ridge_pipeline_handles_missing_and_unknown_categories() -> None:
    rng = np.random.default_rng(1)
    train = pd.DataFrame(
        {
            "MIN_CONC": rng.uniform(0, 1, 40),
            "MAX_CONC": rng.uniform(1, 10, 40),
            "mutational_burden": [np.nan] + rng.uniform(0, 20, 39).tolist(),
            "MUT_TP53": rng.integers(0, 2, 40),
            "CN_ERBB2": rng.uniform(0, 8, 40),
            "PUTATIVE_TARGET": ["EGFR", "PI3K"] * 20,
            "PATHWAY_NAME": ["RTK", "PI3K"] * 20,
            "SCREENING_SITE": "SANGER",
        }
    )
    y = rng.normal(size=40)
    pipeline = make_pipeline(train, "ridge")
    pipeline.fit(train, y)
    test = train.iloc[:2].copy()
    test["PUTATIVE_TARGET"] = "UNSEEN_TARGET"
    prediction = pipeline.predict(test)
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()

