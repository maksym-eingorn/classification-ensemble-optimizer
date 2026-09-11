# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# ensemble/oof_matrix.py

from dataclasses import dataclass
from typing import cast

import numpy as np

from tuning.result_storage import get_optuna_result_dir, load_optuna_results


@dataclass(frozen=True)
class OOFResultBlock:
    """Container for one model family's saved out-of-fold probabilities."""
    model_name: str
    feature_set: str
    trial_numbers: np.ndarray
    oof_probabilities: np.ndarray
    oof_brier_scores: np.ndarray
    fold_brier_scores: np.ndarray
    hyperparams: list[dict[str, object]]


def load_oof_result_block(
    root_dir: str,
    dataset_name: str,
    data_split_seed: int,
    feature_set: str,
    model_name: str
) -> OOFResultBlock:
    """Load one model family's saved Optuna out-of-fold results."""
    result_dir = get_optuna_result_dir(
        root_dir, dataset_name, data_split_seed, feature_set, model_name
    )

    results = load_optuna_results(result_dir)

    hyperparams = cast(list[dict[str, object]], results["hyperparams"])

    return OOFResultBlock(
        model_name=model_name,
        feature_set=feature_set,
        trial_numbers=np.asarray(results["trial_numbers"]),
        oof_probabilities=np.asarray(results["oof_probabilities"]),
        oof_brier_scores=np.asarray(results["oof_brier_scores"]),
        fold_brier_scores=np.asarray(results["fold_brier_scores"]),
        hyperparams=hyperparams
    )


def build_oof_matrix(
    blocks: list[OOFResultBlock]
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """
    Build a unified out-of-fold probability matrix from saved Optuna outputs.

    Parameters
    ----------
    blocks
        Loaded out-of-fold result blocks from model-specific Optuna runs.

    Returns
    -------
    P : np.ndarray
        Unified out-of-fold positive-class probability matrix.
        Shape: (n_samples_dev, n_models).
    column_metadata : list[dict[str, object]]
        Metadata describing which model/trial each column corresponds to.
    """
    if not blocks:
        raise ValueError("At least one out-of-fold result block is required.")

    matrices = []
    column_metadata = []
    n_samples = None
    column_offset = 0

    for block in blocks:
        oof_probabilities = np.asarray(
            block.oof_probabilities, dtype=np.float64
        )

        if oof_probabilities.ndim != 2:
            raise ValueError(
                f"Out-of-fold probabilities for {block.model_name} must be 2D, "
                f"got {oof_probabilities.ndim}D."
            )

        if not np.all(np.isfinite(oof_probabilities)):
            raise ValueError(
                f"Out-of-fold probabilities for {block.model_name} must be "
                "finite."
            )

        if np.any(oof_probabilities < 0.0) or np.any(oof_probabilities > 1.0):
            raise ValueError(
                f"Out-of-fold probabilities for {block.model_name} must be "
                "between 0 and 1."
            )

        if n_samples is None:
            n_samples = oof_probabilities.shape[0]
        elif oof_probabilities.shape[0] != n_samples:
            raise ValueError(
                "All out-of-fold probability matrices must have the same "
                f"number of rows, got {n_samples} and "
                f"{oof_probabilities.shape[0]}."
            )

        n_trials = oof_probabilities.shape[1]

        if n_trials < 1:
            raise ValueError(
                f"Out-of-fold probabilities for {block.model_name} must "
                "contain at least one trial column."
            )

        trial_numbers = np.asarray(block.trial_numbers)

        if trial_numbers.ndim != 1 or len(trial_numbers) != n_trials:
            raise ValueError(
                f"Trial numbers for {block.model_name} must be 1D with length "
                f"{n_trials}, got shape {trial_numbers.shape}."
            )

        oof_brier_scores = np.asarray(block.oof_brier_scores, dtype=np.float64)

        if oof_brier_scores.ndim != 1 or len(oof_brier_scores) != n_trials:
            raise ValueError(
                f"OOF Brier scores for {block.model_name} must be 1D with "
                f"length {n_trials}, got shape {oof_brier_scores.shape}."
            )

        if not np.all(np.isfinite(oof_brier_scores)):
            raise ValueError(
                f"OOF Brier scores for {block.model_name} must be finite."
            )

        fold_brier_scores = np.asarray(
            block.fold_brier_scores, dtype=np.float64
        )

        if fold_brier_scores.ndim != 2:
            raise ValueError(
                f"Fold Brier scores for {block.model_name} must be 2D, "
                f"got {fold_brier_scores.ndim}D."
            )

        if fold_brier_scores.shape[0] != n_trials:
            raise ValueError(
                f"Fold Brier scores for {block.model_name} must have "
                f"{n_trials} rows, got {fold_brier_scores.shape[0]}."
            )

        if not np.all(np.isfinite(fold_brier_scores)):
            raise ValueError(
                f"Fold Brier scores for {block.model_name} must be finite."
            )

        if len(block.hyperparams) != n_trials:
            raise ValueError(
                f"Hyperparameter entries for {block.model_name} must have "
                f"length {n_trials}, got {len(block.hyperparams)}."
            )

        for local_column in range(n_trials):
            column_metadata.append({
                "column_index": column_offset + local_column,
                "model_name": block.model_name,
                "feature_set": block.feature_set,
                "trial_number": int(trial_numbers[local_column]),
                "oof_brier_score": float(oof_brier_scores[local_column]),
                "fold_brier_scores": (
                    fold_brier_scores[local_column].astype(float).tolist()
                ),
                "hyperparams": dict(block.hyperparams[local_column])
            })

        matrices.append(oof_probabilities)
        column_offset += n_trials

    P = np.hstack(matrices).astype(np.float64)

    return P, column_metadata
