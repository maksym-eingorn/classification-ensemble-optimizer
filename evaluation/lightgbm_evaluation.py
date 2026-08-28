# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# evaluation/lightgbm_evaluation.py

import numpy as np
import optuna
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score
)

from evaluation.validation import (
    validate_binary_classification_evaluation_inputs
)
from tuning.validation import validate_random_seed


def evaluate_best_lightgbm_on_test(
    study_lgb: optuna.study.Study,
    X_dev,
    y_dev,
    X_test,
    y_test,
    seed: int = 42
) -> dict[str, float]:
    """
    Retrain the best LightGBM model from Optuna and evaluate it on the test set.

    Parameters
    ----------
    study_lgb : optuna.study.Study
        Completed Optuna study for LightGBM.
    X_dev
        Development feature matrix used for final model retraining.
        Expected shape: (n_samples_dev, n_features).
    y_dev
        Binary development target vector corresponding to X_dev.
        Expected shape: (n_samples_dev,), with classes 0 and 1.
    X_test
        Test feature matrix used for evaluation.
        Expected shape: (n_samples_test, n_features).
    y_test
        Binary test target vector corresponding to X_test.
        Expected shape: (n_samples_test,), with classes 0 and 1.
    seed : int, default=42
        Random seed used by the final LightGBM model.

    Returns
    -------
    test_metrics : dict[str, float]
        Test Brier score, log loss, ROC AUC, and accuracy of the best
        LightGBM model.
        Accuracy uses a positive-class probability threshold of 0.5.
    """
    X_dev, y_dev, X_test, y_test = (
        validate_binary_classification_evaluation_inputs(
            X_dev, y_dev, X_test, y_test
        )
    )
    validate_random_seed(seed, "seed")

    best_params = study_lgb.best_params.copy()

    model = lgb.LGBMClassifier(
        objective="binary",
        random_state=seed,
        feature_fraction_seed=seed,
        bagging_seed=seed,
        extra_seed=seed,
        n_jobs=1,
        verbosity=-1,
        **best_params
    )

    model.fit(X_dev, y_dev)

    y_probability_test = model.predict_proba(X_test)[:, 1]
    y_prediction_test = np.asarray(
        y_probability_test >= 0.5, dtype=np.int64
    )

    return {
        "brier_score": float(brier_score_loss(
            y_test, y_probability_test, pos_label=1, scale_by_half=True
        )),
        "log_loss": float(log_loss(y_test, y_probability_test, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_test, y_probability_test)),
        "accuracy": float(accuracy_score(y_test, y_prediction_test))
    }
