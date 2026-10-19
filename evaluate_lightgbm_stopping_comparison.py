# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# evaluate_lightgbm_stopping_comparison.py

from environment import configure_environment

configure_environment()

from pathlib import Path
from typing import cast

import joblib

import config
from evaluation.ensemble_evaluation import evaluate_greedy_ensemble_on_test
from storage import get_dataset_output_dir, load_prepared_data
from stopping.lightgbm_ensemble_stopping import (
    get_lightgbm_stopping_comparison_output_dir
)
from tuning.feature_sets import validate_lightgbm_feature_set


def _evaluate_one(
    output_dir: Path, arrays: dict
) -> dict[str, object]:
    """Evaluate one saved LightGBM greedy ensemble on the test set."""
    greedy_metadata = cast(
        dict[str, object], joblib.load(output_dir / "greedy_oof_metadata.pkl")
    )

    evaluated_metadata, test_probabilities = (
        evaluate_greedy_ensemble_on_test(
            greedy_metadata,
            arrays,
            seed=config.RANDOM_SEED,
            logistic_regression_max_iter=config.LOGISTIC_REGRESSION_MAX_ITER,
            logistic_regression_tol=config.LOGISTIC_REGRESSION_TOL
        )
    )

    joblib.dump(
        evaluated_metadata,
        output_dir / "greedy_test_metadata.pkl",
        compress=3
    )
    joblib.dump(
        test_probabilities,
        output_dir / "greedy_test_probabilities.pkl",
        compress=3
    )

    return evaluated_metadata


def main() -> None:
    """
    Compare the independently selected LightGBM candidate budget with the
    complete stored LightGBM candidate zoo on the untouched test set.
    """
    validate_lightgbm_feature_set(config.LIGHTGBM_FEATURE_SET)

    prepared_data_dir = get_dataset_output_dir(
        config.PREPARED_DATA_DIR, config.DATASET_NAME, config.DATA_SPLIT_SEED
    )

    arrays, _ = load_prepared_data(prepared_data_dir)

    comparison_dir = (
        get_lightgbm_stopping_comparison_output_dir(
            config.ENSEMBLE_RESULTS_DIR,
            config.DATASET_NAME,
            config.DATA_SPLIT_SEED,
            config.LIGHTGBM_FEATURE_SET
        )
    )

    selected_metadata = _evaluate_one(
        comparison_dir / "selected_budget",
        arrays
    )

    full_metadata = _evaluate_one(
        comparison_dir / "full_budget",
        arrays
    )

    selected_budget = cast(int, selected_metadata["candidate_budget"])
    full_budget = cast(int, full_metadata["candidate_budget"])

    selected_oof_brier = cast(float, selected_metadata["oof_brier_score"])
    selected_test_brier = cast(float, selected_metadata["test_brier_score"])
    selected_test_log_loss = cast(float, selected_metadata["test_log_loss"])
    selected_test_roc_auc = cast(float, selected_metadata["test_roc_auc"])
    selected_test_accuracy = cast(float, selected_metadata["test_accuracy"])

    full_oof_brier = cast(float, full_metadata["oof_brier_score"])
    full_test_brier = cast(float, full_metadata["test_brier_score"])
    full_test_log_loss = cast(float, full_metadata["test_log_loss"])
    full_test_roc_auc = cast(float, full_metadata["test_roc_auc"])
    full_test_accuracy = cast(float, full_metadata["test_accuracy"])

    print("LightGBM ensemble stopping comparison")
    print()

    print(f"Selected budget ({selected_budget} trials):")
    print(f"  OOF Brier = {selected_oof_brier:.5f}")
    print(f"  Test Brier = {selected_test_brier:.5f}")
    print(f"  Test log loss = {selected_test_log_loss:.5f}")
    print(f"  Test ROC AUC = {selected_test_roc_auc:.5f}")
    print(f"  Test accuracy = {selected_test_accuracy:.5f}")

    print()

    print(f"Full budget ({full_budget} trials):")
    print(f"  OOF Brier = {full_oof_brier:.5f}")
    print(f"  Test Brier = {full_test_brier:.5f}")
    print(f"  Test log loss = {full_test_log_loss:.5f}")
    print(f"  Test ROC AUC = {full_test_roc_auc:.5f}")
    print(f"  Test accuracy = {full_test_accuracy:.5f}")

    print()

    difference = selected_test_brier - full_test_brier

    comparison_summary = {
        "data_split_seed": config.DATA_SPLIT_SEED,
        "selected_budget": selected_budget,
        "full_budget": full_budget,
        "selected_oof_brier": selected_oof_brier,
        "full_oof_brier": full_oof_brier,
        "selected_test_brier": selected_test_brier,
        "full_test_brier": full_test_brier,
        "test_brier_difference": difference,
        "selected_test_log_loss": selected_test_log_loss,
        "full_test_log_loss": full_test_log_loss,
        "selected_test_roc_auc": selected_test_roc_auc,
        "full_test_roc_auc": full_test_roc_auc,
        "selected_test_accuracy": selected_test_accuracy,
        "full_test_accuracy": full_test_accuracy
    }

    joblib.dump(
        comparison_summary,
        comparison_dir / "comparison_summary.pkl",
        compress=3
    )

    print(f"Selected minus full test Brier = {difference:+.5f}")

    print(
        f"Comparison summary saved to: "
        f"{comparison_dir / 'comparison_summary.pkl'}"
    )


if __name__ == "__main__":
    main()
