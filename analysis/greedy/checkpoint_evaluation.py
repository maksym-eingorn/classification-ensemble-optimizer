# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# analysis/greedy/checkpoint_evaluation.py

from pathlib import Path
from typing import cast

import joblib
import numpy as np

from analysis.schemas import CheckpointEvaluationResult
from ensemble.greedy_selection import (
    find_greedy_ensemble_from_precomputed_dot_products, precompute_dot_products
)
from ensemble.oof_matrix import OOFResultBlock, build_oof_matrix
from evaluation.ensemble_evaluation import evaluate_greedy_ensemble_on_test
from tuning.feature_sets import get_development_data


GreedyMetadata = dict[str, object]


def _validate_budget_sequence(
    budgets: list[int], available_trials: int
) -> None:
    """
    Validate an ordered checkpoint budget sequence against stored trials.
    """
    if not budgets:
        raise ValueError("At least one checkpoint budget is required.")

    if budgets != sorted(set(budgets)):
        raise ValueError(
            "Checkpoint budgets must be unique and in increasing order."
        )

    if budgets[0] < 1:
        raise ValueError("Checkpoint budgets must be positive.")

    if budgets[-1] > available_trials:
        raise RuntimeError(
            f"Largest checkpoint budget is {budgets[-1]}, "
            f"but only {available_trials} Optuna trials are stored."
        )


def _load_and_validate_optuna_artifacts(
    optuna_results: dict[str, object], n_samples_dev: int
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, object]]
]:
    """Load and validate full-development Optuna artifacts."""
    trial_numbers = np.asarray(optuna_results["trial_numbers"], dtype=int)

    oof_probabilities = np.asarray(
        optuna_results["oof_probabilities"], dtype=np.float64
    )

    oof_brier_scores = np.asarray(
        optuna_results["oof_brier_scores"], dtype=np.float64
    )

    fold_brier_scores = np.asarray(
        optuna_results["fold_brier_scores"], dtype=np.float64
    )

    hyperparams = cast(list[dict[str, object]], optuna_results["hyperparams"])

    if trial_numbers.ndim != 1 or len(trial_numbers) < 1:
        raise RuntimeError(
            "Stored trial_numbers must be a nonempty one-dimensional array."
        )

    expected_trial_numbers = np.arange(len(trial_numbers), dtype=int)

    if not np.array_equal(trial_numbers, expected_trial_numbers):
        raise RuntimeError(
            "Stored full-development trials must be a complete "
            "contiguous sequence 0, 1, ..., N-1."
        )

    n_trials = len(trial_numbers)

    if oof_probabilities.shape != (n_samples_dev, n_trials):
        raise RuntimeError(
            "Stored out-of-fold probability matrix has an unexpected shape: "
            f"{oof_probabilities.shape}; expected "
            f"({n_samples_dev}, {n_trials})."
        )

    if oof_brier_scores.shape != (n_trials,):
        raise RuntimeError(
            "Stored OOF Brier scores have an unexpected shape: "
            f"{oof_brier_scores.shape}."
        )

    if fold_brier_scores.ndim != 2 or fold_brier_scores.shape[0] != n_trials:
        raise RuntimeError(
            "Stored fold Brier scores must be a 2D array "
            "with one row per Optuna trial."
        )

    if len(hyperparams) != n_trials:
        raise RuntimeError(
            "Stored hyperparameter list must contain "
            "one entry per Optuna trial."
        )

    if not np.all(np.isfinite(oof_probabilities)):
        raise RuntimeError("Stored out-of-fold probabilities must be finite.")

    if np.any(oof_probabilities < 0.0) or np.any(oof_probabilities > 1.0):
        raise RuntimeError(
            "Stored out-of-fold probabilities must lie in [0, 1]."
        )

    if not np.all(np.isfinite(oof_brier_scores)):
        raise RuntimeError("Stored OOF Brier scores must be finite.")

    if not np.all(np.isfinite(fold_brier_scores)):
        raise RuntimeError("Stored fold Brier scores must be finite.")

    return (
        trial_numbers,
        oof_probabilities,
        oof_brier_scores,
        fold_brier_scores,
        hyperparams
    )


def _load_existing_checkpoint_artifacts(
    output_path: Path
) -> tuple[
    list[CheckpointEvaluationResult],
    dict[int, GreedyMetadata],
    dict[int, np.ndarray],
    dict[int, np.ndarray]
]:
    """
    Load resumable checkpoint-evaluation artifacts when present.
    """
    results_path = output_path / "checkpoint_results.pkl"
    metadata_path = output_path / "checkpoint_greedy_metadata.pkl"
    oof_path = output_path / "checkpoint_oof_probabilities.pkl"
    test_path = output_path / "checkpoint_test_probabilities.pkl"

    results = joblib.load(results_path) if results_path.exists() else []

    metadata = joblib.load(metadata_path) if metadata_path.exists() else {}

    oof_probabilities = joblib.load(oof_path) if oof_path.exists() else {}

    test_probabilities = joblib.load(test_path) if test_path.exists() else {}

    committed_budgets = [row["budget"] for row in results]

    if len(set(committed_budgets)) != len(committed_budgets):
        raise RuntimeError(
            "Saved checkpoint_results.pkl contains duplicate budgets."
        )

    for budget in committed_budgets:
        if (
            budget not in metadata
            or budget not in oof_probabilities
            or budget not in test_probabilities
        ):
            raise RuntimeError(
                f"Checkpoint budget {budget} is marked complete but one "
                "or more detailed checkpoint artifacts are missing."
            )

    return results, metadata, oof_probabilities, test_probabilities


def evaluate_greedy_checkpoints_for_seed(
    arrays: dict[str, np.ndarray],
    optuna_results: dict[str, object],
    output_dir: str | Path,
    data_split_seed: int,
    model_name: str,
    feature_set: str,
    budgets: list[int],
    pool_size: int | None,
    ensemble_size: int,
    allow_repeats: bool,
    seed: int,
    logistic_regression_max_iter: int,
    logistic_regression_tol: float
) -> list[CheckpointEvaluationResult]:
    """
    Build and test a single-family greedy ensemble at every candidate budget.

    The stored full-development Optuna trajectory is truncated to each budget,
    greedy selection is rerun on that prefix, selected base models are retrained
    on the full development set, and the resulting ensemble is evaluated on the
    untouched outer test set.
    """
    _, y_dev = get_development_data(arrays, feature_set)

    y_dev = np.asarray(y_dev, dtype=np.float64)

    (
        trial_numbers,
        oof_probabilities,
        oof_brier_scores,
        fold_brier_scores,
        hyperparams
    ) = _load_and_validate_optuna_artifacts(
        optuna_results, n_samples_dev=len(y_dev)
    )

    _validate_budget_sequence(budgets, available_trials=len(trial_numbers))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (
        results,
        metadata_by_budget,
        oof_probabilities_by_budget,
        test_probabilities_by_budget
    ) = _load_existing_checkpoint_artifacts(output_path)

    saved_budgets = [row["budget"] for row in results]

    if saved_budgets != budgets[:len(saved_budgets)]:
        raise RuntimeError(
            f"Saved checkpoint budgets {saved_budgets} "
            f"are not the expected leading portion of {budgets}."
        )

    if saved_budgets == budgets:
        return results

    completed_budgets = set(saved_budgets)

    full_budget = budgets[-1]

    full_block = OOFResultBlock(
        model_name=model_name,
        feature_set=feature_set,
        trial_numbers=trial_numbers[:full_budget],
        oof_probabilities=oof_probabilities[:, :full_budget],
        oof_brier_scores=oof_brier_scores[:full_budget],
        fold_brier_scores=fold_brier_scores[:full_budget],
        hyperparams=hyperparams[:full_budget]
    )

    P_full, column_metadata_full = build_oof_matrix([full_block])

    G_full, s_full, y2 = precompute_dot_products(P_full, y_dev)

    for budget in budgets:
        if budget in completed_budgets:
            continue

        P = P_full[:, :budget]
        column_metadata = column_metadata_full[:budget]

        (
            greedy_metadata, greedy_oof_probabilities
        ) = (
            find_greedy_ensemble_from_precomputed_dot_products(
                P,
                y_dev,
                column_metadata,
                G_full[:budget, :budget],
                s_full[:budget],
                y2,
                pool_size=pool_size,
                ensemble_size=ensemble_size,
                seed=seed,
                allow_repeats=allow_repeats
            )
        )

        greedy_metadata["candidate_budget"] = budget

        (
            evaluated_metadata, test_probabilities
        ) = (
            evaluate_greedy_ensemble_on_test(
                greedy_metadata,
                arrays,
                seed=seed,
                logistic_regression_max_iter=logistic_regression_max_iter,
                logistic_regression_tol=logistic_regression_tol
            )
        )

        row: CheckpointEvaluationResult = {
            "data_split_seed": data_split_seed,
            "budget": budget,
            "model_name": model_name,
            "feature_set": feature_set,
            "ensemble_method": "greedy",
            "oof_brier_score": float(
                cast(float, evaluated_metadata["oof_brier_score"])
            ),
            "test_brier_score": float(
                cast(float, evaluated_metadata["test_brier_score"])
            ),
            "test_log_loss": float(
                cast(float, evaluated_metadata["test_log_loss"])
            ),
            "test_roc_auc": float(
                cast(float, evaluated_metadata["test_roc_auc"])
            ),
            "test_accuracy": float(
                cast(float, evaluated_metadata["test_accuracy"])
            )
        }

        metadata_by_budget[budget] = evaluated_metadata

        oof_probabilities_by_budget[budget] = np.asarray(
            greedy_oof_probabilities, dtype=np.float32
        )

        test_probabilities_by_budget[budget] = np.asarray(
            test_probabilities[:, 0], dtype=np.float32
        )

        # Detailed artifacts are written before checkpoint_results.pkl,
        # which acts as the commit marker for a completed budget.
        joblib.dump(
            metadata_by_budget,
            output_path / "checkpoint_greedy_metadata.pkl",
            compress=3
        )

        joblib.dump(
            oof_probabilities_by_budget,
            output_path / "checkpoint_oof_probabilities.pkl",
            compress=3
        )

        joblib.dump(
            test_probabilities_by_budget,
            output_path / "checkpoint_test_probabilities.pkl",
            compress=3
        )

        results.append(row)
        results.sort(key=lambda item: item["budget"])

        joblib.dump(
            results, output_path / "checkpoint_results.pkl", compress=3
        )

        completed_budgets.add(budget)

    saved_budgets = [row["budget"] for row in results]

    if saved_budgets != budgets:
        raise RuntimeError(
            f"Completed checkpoint budgets {saved_budgets} "
            f"do not match expected budgets {budgets}."
        )

    return results


def load_greedy_checkpoint_results(
    output_dir: str | Path
) -> list[CheckpointEvaluationResult]:
    """
    Load completed scalar checkpoint-evaluation results for one split seed.
    """
    path = Path(output_dir) / "checkpoint_results.pkl"

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint evaluation results were not found: {path}"
        )

    return cast(list[CheckpointEvaluationResult], joblib.load(path))
