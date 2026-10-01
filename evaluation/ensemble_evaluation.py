# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# evaluation/ensemble_evaluation.py

from typing import cast

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score
)
from sklearn.preprocessing import StandardScaler

from evaluation.validation import (
    validate_binary_classification_evaluation_inputs
)
from tuning.feature_sets import (
    get_development_and_test_data,
    validate_lightgbm_feature_set,
    validate_logistic_regression_feature_set,
    validate_xgboost_feature_set
)
from tuning.validation import validate_positive_integer, validate_random_seed


ModelMetadata = dict[str, object]
TripletMetadata = dict[str, object]
GreedyMetadata = dict[str, object]


def _validate_logistic_regression_settings(
    max_iter: int, tol: float
) -> None:
    """Validate Logistic Regression settings used for final retraining."""
    validate_positive_integer(max_iter, "logistic_regression_max_iter")

    if (
        isinstance(tol, bool)
        or not isinstance(tol, (int, float))
        or not np.isfinite(tol)
        or tol <= 0
    ):
        raise ValueError(
            "logistic_regression_tol must be a positive finite number, "
            f"got {tol}."
        )


def _build_model_from_metadata(
    model_metadata: ModelMetadata,
    seed: int,
    logistic_regression_max_iter: int,
    logistic_regression_tol: float
):
    """Build a classifier from saved base model column metadata."""
    model_name = cast(str, model_metadata["model_name"])
    hyperparams = dict(cast(dict[str, object], model_metadata["hyperparams"]))

    if model_name == "xgboost":
        return xgb.XGBClassifier(
            objective="binary:logistic",
            random_state=seed,
            tree_method="hist",
            n_jobs=1,
            verbosity=0,
            **hyperparams
        )

    if model_name == "lightgbm":
        return lgb.LGBMClassifier(
            objective="binary",
            random_state=seed,
            feature_fraction_seed=seed,
            bagging_seed=seed,
            extra_seed=seed,
            n_jobs=1,
            verbosity=-1,
            **hyperparams
        )

    if model_name == "logistic_regression":
        _validate_logistic_regression_settings(
            logistic_regression_max_iter, logistic_regression_tol
        )

        return LogisticRegression(
            solver="saga",
            max_iter=logistic_regression_max_iter,
            tol=logistic_regression_tol,
            random_state=seed,
            **hyperparams
        )

    raise ValueError(f"Unsupported model_name in metadata: {model_name}.")


def _validate_model_feature_set(
    model_name: str, feature_set: str
) -> None:
    """Validate the feature set recorded for one base model."""
    if model_name == "xgboost":
        validate_xgboost_feature_set(feature_set)
        return

    if model_name == "lightgbm":
        validate_lightgbm_feature_set(feature_set)
        return

    if model_name == "logistic_regression":
        validate_logistic_regression_feature_set(feature_set)
        return

    raise ValueError(f"Unsupported model_name in metadata: {model_name}.")


def _predict_base_model_on_test(
    model_metadata: ModelMetadata,
    arrays: dict[str, np.ndarray],
    seed: int,
    logistic_regression_max_iter: int,
    logistic_regression_tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Retrain one selected base model and predict test probabilities."""
    model_name = cast(str, model_metadata["model_name"])
    feature_set = cast(str, model_metadata["feature_set"])

    _validate_model_feature_set(model_name, feature_set)

    X_dev, y_dev, X_test, y_test = get_development_and_test_data(
        arrays, feature_set
    )

    X_dev, y_dev, X_test, y_test = (
        validate_binary_classification_evaluation_inputs(
            X_dev, y_dev, X_test, y_test
        )
    )

    model = _build_model_from_metadata(
        model_metadata,
        seed=seed,
        logistic_regression_max_iter=logistic_regression_max_iter,
        logistic_regression_tol=logistic_regression_tol
    )

    if model_name == "logistic_regression":
        scaler = StandardScaler()
        X_dev_model = scaler.fit_transform(X_dev)
        X_test_model = scaler.transform(X_test)
    else:
        X_dev_model = X_dev
        X_test_model = X_test

    model.fit(X_dev_model, y_dev)

    y_probability_test = np.asarray(
        model.predict_proba(X_test_model)[:, 1], dtype=np.float64
    )
    y_test = np.asarray(y_test, dtype=np.int64)

    if not np.all(np.isfinite(y_probability_test)):
        raise ValueError("Base model test probabilities must be finite.")

    if np.any(y_probability_test < 0.0) or np.any(y_probability_test > 1.0):
        raise ValueError(
            "Base model test probabilities must be between 0 and 1."
        )

    return y_probability_test, y_test


def _get_triplet_models(triplet: TripletMetadata) -> list[ModelMetadata]:
    """Return the three base model metadata dictionaries for one triplet."""
    if "models" not in triplet:
        raise KeyError(
            "Triplet metadata does not contain 'models'. "
            "Run run_triplet_search.py before test evaluation."
        )

    models = cast(list[ModelMetadata], triplet["models"])

    if len(models) != 3:
        raise ValueError(
            f"Triplet metadata must contain 3 models, got {len(models)}."
        )

    return models


def _get_triplet_weights(triplet: TripletMetadata) -> np.ndarray:
    """Return validated convex weights for one triplet."""
    weights = np.array(
        [
            cast(float, triplet["wi"]),
            cast(float, triplet["wj"]),
            cast(float, triplet["wk"])
        ],
        dtype=np.float64
    )

    if not np.all(np.isfinite(weights)):
        raise ValueError("Triplet weights must be finite.")

    if np.any(weights < 0.0):
        raise ValueError("Triplet weights must be nonnegative.")

    weight_sum = float(np.sum(weights))

    if not np.isclose(weight_sum, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"Triplet weights must sum to 1, got {weight_sum}.")

    return weights


def evaluate_triplet_ensembles_on_test(
    triplet_metadata: list[TripletMetadata],
    arrays: dict[str, np.ndarray],
    seed: int = 42,
    logistic_regression_max_iter: int = 20000,
    logistic_regression_tol: float = 1e-4
) -> tuple[list[TripletMetadata], np.ndarray]:
    """
    Retrain selected triplet ensembles on the full development set and evaluate
    them on the test set.

    Parameters
    ----------
    triplet_metadata
        Metadata saved by run_triplet_search.py.
    arrays
        Prepared dataset arrays loaded from storage.
    seed
        Random seed used for final base model retraining.
    logistic_regression_max_iter
        Maximum iterations for final Logistic Regression retraining.
    logistic_regression_tol
        Optimization tolerance for final Logistic Regression retraining.

    Returns
    -------
    evaluated_metadata
        Copy of triplet metadata with added test Brier score, log loss,
        ROC AUC, and accuracy values.
    triplet_test_probabilities
        Test positive-class probability matrix for evaluated triplets.
        Shape: (n_samples_test, n_kept_triplets).
    """
    validate_random_seed(seed, "seed")

    if not triplet_metadata:
        raise ValueError("triplet_metadata must contain at least one triplet.")

    probability_cache: dict[int, np.ndarray] = {}
    y_test_reference = None

    evaluated_metadata = []
    triplet_test_probabilities = []

    for triplet in triplet_metadata:
        models = _get_triplet_models(triplet)
        weights = _get_triplet_weights(triplet)

        y_probability_ensemble = None

        for weight, model_metadata in zip(weights, models):
            column_index = cast(int, model_metadata["column_index"])

            if column_index not in probability_cache:
                y_probability_test, y_test = _predict_base_model_on_test(
                    model_metadata,
                    arrays,
                    seed=seed,
                    logistic_regression_max_iter=logistic_regression_max_iter,
                    logistic_regression_tol=logistic_regression_tol
                )

                probability_cache[column_index] = y_probability_test

                if y_test_reference is None:
                    y_test_reference = y_test
                elif not np.array_equal(y_test_reference, y_test):
                    raise ValueError(
                        "Inconsistent y_test values encountered across "
                        "feature sets."
                    )

            y_probability_test = probability_cache[column_index]

            if y_probability_ensemble is None:
                y_probability_ensemble = weight * y_probability_test
            else:
                y_probability_ensemble += weight * y_probability_test

        if y_test_reference is None or y_probability_ensemble is None:
            raise RuntimeError(
                "Failed to construct triplet ensemble test probabilities."
            )

        y_prediction_ensemble = np.asarray(
            y_probability_ensemble >= 0.5, dtype=np.int64
        )

        test_brier_score = float(
            brier_score_loss(
                y_test_reference,
                y_probability_ensemble,
                pos_label=1,
                scale_by_half=True
            )
        )
        test_log_loss = float(
            log_loss(y_test_reference, y_probability_ensemble, labels=[0, 1])
        )
        test_roc_auc = float(
            roc_auc_score(y_test_reference, y_probability_ensemble)
        )
        test_accuracy = float(
            accuracy_score(y_test_reference, y_prediction_ensemble)
        )

        evaluated_triplet = triplet.copy()
        evaluated_triplet["test_brier_score"] = test_brier_score
        evaluated_triplet["test_log_loss"] = test_log_loss
        evaluated_triplet["test_roc_auc"] = test_roc_auc
        evaluated_triplet["test_accuracy"] = test_accuracy

        evaluated_metadata.append(evaluated_triplet)
        triplet_test_probabilities.append(y_probability_ensemble)

    triplet_test_probabilities_matrix = np.column_stack(
        triplet_test_probabilities
    ).astype(np.float32)

    return evaluated_metadata, triplet_test_probabilities_matrix


def _get_greedy_models(
    greedy_metadata: GreedyMetadata
) -> list[ModelMetadata]:
    """Return selected base model metadata for a greedy ensemble."""
    if "models" not in greedy_metadata:
        raise KeyError(
            "Greedy metadata does not contain 'models'. "
            "Run run_greedy_ensemble_selection.py before test evaluation."
        )

    models = cast(list[ModelMetadata], greedy_metadata["models"])

    if not models:
        raise ValueError("Greedy metadata must contain at least one model.")

    return models


def _get_greedy_weights(
    greedy_metadata: GreedyMetadata
) -> np.ndarray:
    """Return validated convex weights for a greedy ensemble."""
    if "weights" not in greedy_metadata:
        raise KeyError("Greedy metadata does not contain 'weights'.")

    weights = np.asarray(
        cast(list[float], greedy_metadata["weights"]), dtype=np.float64
    )

    if weights.ndim != 1:
        raise ValueError(f"Greedy weights must be 1D, got {weights.ndim}D.")

    if len(weights) == 0:
        raise ValueError("Greedy weights must contain at least one value.")

    if not np.all(np.isfinite(weights)):
        raise ValueError("Greedy weights must be finite.")

    if np.any(weights < 0.0):
        raise ValueError("Greedy weights must be nonnegative.")

    weight_sum = float(np.sum(weights))

    if not np.isclose(weight_sum, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"Greedy weights must sum to 1, got {weight_sum}.")

    return weights


def evaluate_greedy_ensemble_on_test(
    greedy_metadata: GreedyMetadata,
    arrays: dict[str, np.ndarray],
    seed: int = 42,
    logistic_regression_max_iter: int = 20000,
    logistic_regression_tol: float = 1e-4
) -> tuple[GreedyMetadata, np.ndarray]:
    """
    Retrain the selected greedy ensemble on the full development set and
    evaluate it on the held-out test set.

    Parameters
    ----------
    greedy_metadata
        Metadata saved by run_greedy_ensemble_selection.py.
    arrays
        Prepared dataset arrays loaded from storage.
    seed
        Random seed used for final base model retraining.
    logistic_regression_max_iter
        Maximum iterations for final Logistic Regression retraining.
    logistic_regression_tol
        Optimization tolerance for final Logistic Regression retraining.

    Returns
    -------
    evaluated_metadata
        Copy of greedy metadata with added test Brier score, log loss,
        ROC AUC, and accuracy.
    greedy_test_probabilities
        Test positive-class probability matrix for the evaluated greedy
        ensemble.
        Shape: (n_samples_test, 1).
    """
    validate_random_seed(seed, "seed")

    models = _get_greedy_models(greedy_metadata)
    weights = _get_greedy_weights(greedy_metadata)

    if len(models) != len(weights):
        raise ValueError(
            f"Number of greedy models and weights must match, "
            f"got {len(models)} != {len(weights)}."
        )

    probability_cache: dict[int, np.ndarray] = {}
    y_test_reference = None
    y_probability_ensemble = None

    for weight, model_metadata in zip(weights, models):
        column_index = cast(int, model_metadata["column_index"])

        if column_index not in probability_cache:
            y_probability_test, y_test = _predict_base_model_on_test(
                model_metadata,
                arrays,
                seed=seed,
                logistic_regression_max_iter=logistic_regression_max_iter,
                logistic_regression_tol=logistic_regression_tol
            )

            probability_cache[column_index] = y_probability_test

            if y_test_reference is None:
                y_test_reference = y_test
            elif not np.array_equal(y_test_reference, y_test):
                raise ValueError(
                    "Inconsistent y_test values encountered across "
                    "feature sets."
                )

        y_probability_test = probability_cache[column_index]

        if y_probability_ensemble is None:
            y_probability_ensemble = weight * y_probability_test
        else:
            y_probability_ensemble += weight * y_probability_test

    if y_test_reference is None or y_probability_ensemble is None:
        raise RuntimeError(
            "Failed to construct greedy ensemble test probabilities."
        )

    y_prediction_ensemble = np.asarray(
        y_probability_ensemble >= 0.5, dtype=np.int64
    )

    test_brier_score = float(
        brier_score_loss(
            y_test_reference,
            y_probability_ensemble,
            pos_label=1,
            scale_by_half=True
        )
    )
    test_log_loss = float(
        log_loss(y_test_reference, y_probability_ensemble, labels=[0, 1])
    )
    test_roc_auc = float(
        roc_auc_score(y_test_reference, y_probability_ensemble)
    )
    test_accuracy = float(
        accuracy_score(y_test_reference, y_prediction_ensemble)
    )

    evaluated_metadata = greedy_metadata.copy()
    evaluated_metadata["test_brier_score"] = test_brier_score
    evaluated_metadata["test_log_loss"] = test_log_loss
    evaluated_metadata["test_roc_auc"] = test_roc_auc
    evaluated_metadata["test_accuracy"] = test_accuracy

    test_probabilities_matrix = np.asarray(
        y_probability_ensemble, dtype=np.float32
    ).reshape(-1, 1)

    return evaluated_metadata, test_probabilities_matrix
