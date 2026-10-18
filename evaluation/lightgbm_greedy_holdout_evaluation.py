# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# evaluation/lightgbm_greedy_holdout_evaluation.py

from typing import cast

import lightgbm as lgb
import numpy as np
from sklearn.metrics import brier_score_loss

from evaluation.validation import (
    validate_binary_classification_evaluation_inputs
)
from tuning.validation import validate_random_seed


ModelMetadata = dict[str, object]
GreedyMetadata = dict[str, object]


def evaluate_lightgbm_greedy_ensemble_on_holdout(
    greedy_metadata: GreedyMetadata,
    X_train,
    y_train,
    X_holdout,
    y_holdout,
    seed: int = 42
) -> tuple[float, np.ndarray]:
    """
    Retrain one LightGBM-only greedy ensemble on a meta-training set and
    evaluate it on the corresponding untouched meta-holdout set.
    """
    validate_random_seed(seed, "seed")

    X_train, y_train, X_holdout, y_holdout = (
        validate_binary_classification_evaluation_inputs(
            X_train, y_train, X_holdout, y_holdout
        )
    )

    models = cast(list[ModelMetadata], greedy_metadata["models"])
    weights = np.asarray(
        cast(list[float], greedy_metadata["weights"]), dtype=np.float64
    )

    if not models:
        raise ValueError(
            "Greedy metadata must contain at least one selected model."
        )

    if weights.ndim != 1:
        raise ValueError(
            f"Greedy weights must be 1D, got {weights.ndim}D."
        )

    if len(models) != len(weights):
        raise ValueError(
            "Numbers of selected LightGBM models and weights must match."
        )

    if not np.all(np.isfinite(weights)):
        raise ValueError("Greedy weights must be finite.")

    if np.any(weights < 0.0):
        raise ValueError("Greedy weights must be nonnegative.")

    if not np.isclose(float(np.sum(weights)), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Greedy weights must sum to 1.")

    y_probability_ensemble = np.zeros(len(y_holdout), dtype=np.float64)

    for weight, model_metadata in zip(weights, models):
        model_name = cast(str, model_metadata["model_name"])
        feature_set = cast(str, model_metadata["feature_set"])

        if model_name != "lightgbm":
            raise ValueError(
                "LightGBM stopping evaluation received a non-LightGBM model."
            )

        if feature_set != "original":
            raise ValueError(
                "LightGBM stopping currently supports only "
                "feature_set='original'."
            )

        hyperparams = dict(
            cast(dict[str, object], model_metadata["hyperparams"])
        )

        model = lgb.LGBMClassifier(
            objective="binary",
            random_state=seed,
            feature_fraction_seed=seed,
            bagging_seed=seed,
            extra_seed=seed,
            n_jobs=1,
            verbosity=-1,
            **hyperparams
        )

        model.fit(X_train, y_train)

        y_probability = np.asarray(
            model.predict_proba(X_holdout)[:, 1], dtype=np.float64
        )

        if not np.all(np.isfinite(y_probability)):
            raise ValueError(
                "LightGBM holdout probabilities must be finite."
            )

        if np.any(y_probability < 0.0) or np.any(y_probability > 1.0):
            raise ValueError(
                "LightGBM holdout probabilities must be between 0 and 1."
            )

        y_probability_ensemble += weight * y_probability

    brier_score = float(
        brier_score_loss(
            y_holdout, y_probability_ensemble, pos_label=1, scale_by_half=True
        )
    )

    return brier_score, y_probability_ensemble
