# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# stopping/lightgbm_ensemble_stopping.py

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import optuna
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

from ensemble.greedy_selection import find_greedy_ensemble
from ensemble.oof_matrix import OOFResultBlock, build_oof_matrix
from evaluation.lightgbm_greedy_holdout_evaluation import (
    evaluate_lightgbm_greedy_ensemble_on_holdout
)
from storage import get_data_split_name
from tuning.lightgbm_incremental_optuna import run_optuna_kfold_lightgbm_block
from tuning.result_storage import load_optuna_results, save_optuna_results
from tuning.validation import (
    validate_binary_classification_inputs,
    validate_positive_integer,
    validate_random_seed,
    validate_stratified_kfold_settings
)


def get_lightgbm_stopping_output_dir(
    root_dir: str | Path,
    dataset_name: str,
    data_split_seed: int,
    feature_set: str
) -> Path:
    """Return the root directory for LightGBM ensemble-aware stopping."""
    return (
        Path(root_dir)
        / dataset_name
        / get_data_split_name(data_split_seed)
        / f"lightgbm-{feature_set}"
        / "greedy_ensemble_stopping"
    )


def get_lightgbm_stopping_comparison_output_dir(
    root_dir: str | Path,
    dataset_name: str,
    data_split_seed: int,
    feature_set: str
) -> Path:
    """Return output directory for stopped-vs-full LightGBM ensembles."""
    return (
        Path(root_dir)
        / dataset_name
        / get_data_split_name(data_split_seed)
        / f"lightgbm-{feature_set}"
        / "stopping_comparison"
    )


def _concatenate_optuna_results(
    previous: dict[str, object] | None,
    study: optuna.study.Study,
    new_trial_numbers: np.ndarray,
    new_oof_probabilities: np.ndarray,
    new_oof_brier_scores: np.ndarray,
    new_fold_brier_scores: np.ndarray,
    new_hyperparams: list[dict[str, object]]
) -> tuple[
    optuna.study.Study,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, object]]
]:
    """Append one new trial block to previously saved Optuna artifacts."""
    if previous is None:
        return (
            study,
            new_trial_numbers,
            new_oof_probabilities,
            new_oof_brier_scores,
            new_fold_brier_scores,
            new_hyperparams
        )

    old_trial_numbers = np.asarray(previous["trial_numbers"], dtype=int)
    old_oof_probabilities = np.asarray(
        previous["oof_probabilities"], dtype=np.float32
    )
    old_oof_brier_scores = np.asarray(previous["oof_brier_scores"], dtype=float)
    old_fold_brier_scores = np.asarray(
        previous["fold_brier_scores"], dtype=float
    )
    old_hyperparams = cast(list[dict[str, object]], previous["hyperparams"])

    return (
        study,
        np.concatenate([old_trial_numbers, new_trial_numbers]),
        np.hstack(
            [old_oof_probabilities, new_oof_probabilities]
        ).astype(np.float32),
        np.concatenate([old_oof_brier_scores, new_oof_brier_scores]),
        np.vstack([old_fold_brier_scores, new_fold_brier_scores]),
        old_hyperparams + new_hyperparams
    )


def _validate_trial_sequence(trial_numbers: np.ndarray) -> None:
    """Require complete sequential Optuna trial numbering from zero."""
    expected = np.arange(len(trial_numbers), dtype=int)

    if not np.array_equal(trial_numbers, expected):
        raise RuntimeError(
            "Meta-fold Optuna trials are not a complete contiguous sequence "
            "starting at trial 0."
        )


def _run_meta_fold_checkpoint(
    fold_number: int,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    fold_dir_string: str,
    expected_previous_trials: int,
    checkpoint_trials: int,
    feature_set: str,
    inner_n_splits: int,
    n_estimators_min: int,
    n_estimators_max: int,
    n_estimators_step: int,
    greedy_pool_size: int | None,
    greedy_ensemble_size: int,
    greedy_allow_repeats: bool,
    seed: int,
    verbose: bool
) -> dict[str, object]:
    """
    Advance one meta-fold Optuna trajectory by one block, rebuild its greedy
    ensemble from the entire accumulated trial zoo, and evaluate the ensemble
    on the untouched meta-holdout fold.
    """
    fold_dir = Path(fold_dir_string)
    fold_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_total = expected_previous_trials + checkpoint_trials
    checkpoint_path = fold_dir / f"checkpoint_{checkpoint_total}.pkl"

    optuna_dir = fold_dir / "optuna"

    study_path = optuna_dir / "study.pkl"

    if study_path.exists():
        previous = load_optuna_results(optuna_dir)

        previous_trial_numbers = np.asarray(
            previous["trial_numbers"], dtype=int
        )

        _validate_trial_sequence(previous_trial_numbers)

        current_trials = len(previous_trial_numbers)
    else:
        previous = None
        current_trials = 0

    # Crash-safe resume: the fold may already have completed this checkpoint
    # even if the parent process had not yet written the central history.
    if current_trials == checkpoint_total and checkpoint_path.exists():
        return cast(dict[str, object], joblib.load(checkpoint_path))

    if current_trials != expected_previous_trials:
        raise RuntimeError(
            f"Meta-fold {fold_number} contains {current_trials} saved trials, "
            f"but {expected_previous_trials} were expected."
        )

    X_train = X_dev[train_idx]
    y_train = y_dev[train_idx]
    X_holdout = X_dev[valid_idx]
    y_holdout = y_dev[valid_idx]

    if previous is None:
        existing_study: optuna.study.Study | None = None
    else:
        existing_study = cast(optuna.study.Study, previous["study"])

    (
        updated_study,
        new_trial_numbers,
        new_oof_probabilities,
        new_oof_brier_scores,
        new_fold_brier_scores,
        new_hyperparams
    ) = run_optuna_kfold_lightgbm_block(
        X_train,
        y_train,
        study=existing_study,
        n_trials=checkpoint_trials,
        n_splits=inner_n_splits,
        n_estimators_min=n_estimators_min,
        n_estimators_max=n_estimators_max,
        n_estimators_step=n_estimators_step,
        seed=seed,
        verbose=verbose
    )

    (
        final_study,
        trial_numbers,
        oof_probabilities,
        oof_brier_scores,
        fold_brier_scores,
        hyperparams
    ) = _concatenate_optuna_results(
        previous,
        updated_study,
        new_trial_numbers,
        new_oof_probabilities,
        new_oof_brier_scores,
        new_fold_brier_scores,
        new_hyperparams
    )

    _validate_trial_sequence(trial_numbers)

    if len(trial_numbers) != checkpoint_total:
        raise RuntimeError(
            f"Meta-fold {fold_number} expected {checkpoint_total} trials "
            f"after this block, got {len(trial_numbers)}."
        )

    block = OOFResultBlock(
        model_name="lightgbm",
        feature_set=feature_set,
        trial_numbers=trial_numbers,
        oof_probabilities=oof_probabilities,
        oof_brier_scores=oof_brier_scores,
        fold_brier_scores=fold_brier_scores,
        hyperparams=hyperparams
    )

    P, column_metadata = build_oof_matrix([block])

    greedy_metadata, _ = find_greedy_ensemble(
        P,
        y_train,
        column_metadata,
        pool_size=greedy_pool_size,
        ensemble_size=greedy_ensemble_size,
        seed=seed,
        allow_repeats=greedy_allow_repeats
    )

    holdout_brier_score, holdout_probabilities = (
        evaluate_lightgbm_greedy_ensemble_on_holdout(
            greedy_metadata,
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            seed=seed
        )
    )

    result: dict[str, object] = {
        "fold_number": fold_number,
        "n_trials": checkpoint_total,
        "valid_idx": valid_idx.astype(int),
        "holdout_brier_score": holdout_brier_score,
        "holdout_probabilities": (
            np.asarray(holdout_probabilities, dtype=np.float32)
        ),
        "greedy_metadata": greedy_metadata
    }

    joblib.dump(result, checkpoint_path, compress=3)

    save_optuna_results(
        optuna_dir,
        final_study,
        trial_numbers,
        oof_probabilities,
        oof_brier_scores,
        fold_brier_scores,
        hyperparams
    )

    return result


def run_lightgbm_ensemble_aware_stopping(
    X_dev,
    y_dev,
    output_dir: str | Path,
    feature_set: str,
    meta_n_splits: int,
    checkpoint_trials: int,
    patience: int,
    max_trials: int,
    max_workers: int,
    inner_n_splits: int,
    n_estimators_min: int,
    n_estimators_max: int,
    n_estimators_step: int,
    greedy_pool_size: int | None,
    greedy_ensemble_size: int,
    greedy_allow_repeats: bool,
    seed: int,
    verbose: bool = False
) -> dict[str, object]:
    """
    Select an Optuna trial budget by cross-fitted downstream-ensemble
    performance, using strict improvement and checkpoint patience.
    """
    X_dev, y_dev = validate_binary_classification_inputs(X_dev, y_dev)

    validate_random_seed(seed, "seed")
    validate_positive_integer(meta_n_splits, "meta_n_splits")
    validate_positive_integer(checkpoint_trials, "checkpoint_trials")
    validate_positive_integer(patience, "patience")
    validate_positive_integer(max_trials, "max_trials")
    validate_positive_integer(max_workers, "max_workers")
    validate_positive_integer(
        greedy_ensemble_size,
        "greedy_ensemble_size"
    )

    validate_stratified_kfold_settings(meta_n_splits, y_dev)

    if max_trials % checkpoint_trials != 0:
        raise ValueError(
            "max_trials must be an exact multiple of checkpoint_trials."
        )

    if max_workers > meta_n_splits:
        raise ValueError(
            "max_workers cannot exceed meta_n_splits."
        )

    if feature_set != "original":
        raise ValueError(
            "LightGBM ensemble-aware stopping currently supports only "
            "feature_set='original'."
        )

    if not isinstance(greedy_allow_repeats, bool):
        raise ValueError(
            "greedy_allow_repeats must be a boolean."
        )

    if not isinstance(verbose, bool):
        raise ValueError("verbose must be a boolean.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    settings = {
        "feature_set": feature_set,
        "meta_n_splits": meta_n_splits,
        "checkpoint_trials": checkpoint_trials,
        "patience": patience,
        "max_trials": max_trials,
        "inner_n_splits": inner_n_splits,
        "n_estimators_min": n_estimators_min,
        "n_estimators_max": n_estimators_max,
        "n_estimators_step": n_estimators_step,
        "greedy_pool_size": greedy_pool_size,
        "greedy_ensemble_size": greedy_ensemble_size,
        "greedy_allow_repeats": greedy_allow_repeats,
        "seed": seed
    }

    settings_path = output_path / "stopping_settings.pkl"

    if settings_path.exists():
        saved_settings = joblib.load(settings_path)

        if saved_settings != settings:
            raise RuntimeError(
                "Existing stopping results were created with different "
                "settings. Remove the stopping output directory before "
                "starting a new experiment."
            )
    else:
        joblib.dump(settings, settings_path, compress=3)

    summary_path = output_path / "stopping_summary.pkl"

    if summary_path.exists():
        summary = cast(dict[str, object], joblib.load(summary_path))

        if bool(summary.get("completed", False)):
            return summary

    history_path = output_path / "stopping_history.pkl"
    probabilities_path = (
        output_path / "checkpoint_cross_fitted_probabilities.pkl"
    )

    if history_path.exists():
        history = cast(list[dict[str, object]], joblib.load(history_path))
    else:
        history = []

    if probabilities_path.exists():
        checkpoint_probabilities = cast(
            dict[int, np.ndarray], joblib.load(probabilities_path)
        )
    else:
        checkpoint_probabilities = {}

    meta_splitter = StratifiedKFold(
        n_splits=meta_n_splits, shuffle=True, random_state=seed
    )

    meta_splits = list(meta_splitter.split(X_dev, y_dev))

    if history:
        current_trials = cast(int, history[-1]["n_trials"])

        best_entry = min(
            history,
            key=lambda item: cast(
                float, item["cross_fitted_brier_score"]
            )
        )

        best_score = cast(float, best_entry["cross_fitted_brier_score"])
        best_budget = cast(int, best_entry["n_trials"])

        consecutive_nonimprovements = cast(
            int, history[-1]["consecutive_nonimprovements"]
        )
    else:
        current_trials = 0
        best_score = np.inf
        best_budget = 0
        consecutive_nonimprovements = 0

    stop_reason = None

    while current_trials < max_trials:
        next_trials = current_trials + checkpoint_trials

        futures = []

        with ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            for fold_offset, (train_idx, valid_idx) in enumerate(
                meta_splits, start=1
            ):
                fold_dir = output_path / f"meta_fold_{fold_offset}"

                future = executor.submit(
                    _run_meta_fold_checkpoint,
                    fold_offset,
                    X_dev,
                    y_dev,
                    np.asarray(train_idx, dtype=int),
                    np.asarray(valid_idx, dtype=int),
                    str(fold_dir),
                    current_trials,
                    checkpoint_trials,
                    feature_set,
                    inner_n_splits,
                    n_estimators_min,
                    n_estimators_max,
                    n_estimators_step,
                    greedy_pool_size,
                    greedy_ensemble_size,
                    greedy_allow_repeats,
                    seed,
                    verbose
                )

                futures.append(future)

            fold_results = [future.result() for future in futures]

        fold_results.sort(
            key=lambda item: cast(int, item["fold_number"])
        )

        cross_fitted_probabilities = np.full(
            len(y_dev), np.nan, dtype=np.float64
        )

        fold_brier_scores = []

        for fold_result in fold_results:
            valid_idx = np.asarray(fold_result["valid_idx"], dtype=int)

            holdout_probabilities = np.asarray(
                fold_result["holdout_probabilities"], dtype=np.float64
            )

            if len(valid_idx) != len(holdout_probabilities):
                raise RuntimeError(
                    "Meta-fold validation indices and probabilities have "
                    "inconsistent lengths."
                )

            cross_fitted_probabilities[valid_idx] = holdout_probabilities

            fold_brier_scores.append(float(fold_result["holdout_brier_score"]))

        if not np.all(np.isfinite(cross_fitted_probabilities)):
            raise RuntimeError(
                "Cross-fitted checkpoint probabilities are not finite."
            )

        cross_fitted_brier_score = float(
            brier_score_loss(
                y_dev,
                cross_fitted_probabilities,
                pos_label=1,
                scale_by_half=True
            )
        )

        is_new_best = cross_fitted_brier_score < best_score

        if is_new_best:
            best_score = cross_fitted_brier_score
            best_budget = next_trials
            consecutive_nonimprovements = 0
        else:
            consecutive_nonimprovements += 1

        history_entry: dict[str, object] = {
            "n_trials": next_trials,
            "cross_fitted_brier_score": cross_fitted_brier_score,
            "fold_brier_scores": fold_brier_scores,
            "is_new_best": is_new_best,
            "best_budget_after_checkpoint": best_budget,
            "best_brier_after_checkpoint": best_score,
            "consecutive_nonimprovements": consecutive_nonimprovements
        }

        history.append(history_entry)

        checkpoint_probabilities[next_trials] = (
            cross_fitted_probabilities.astype(np.float32)
        )

        joblib.dump(checkpoint_probabilities, probabilities_path, compress=3)
        joblib.dump(history, history_path, compress=3)

        print(
            f"Checkpoint {next_trials}: "
            f"cross-fitted ensemble Brier = "
            f"{cross_fitted_brier_score:.5f}"
        )
        print(
            "Meta-fold Brier scores: "
            + ", ".join(f"{score:.5f}" for score in fold_brier_scores)
        )

        if is_new_best:
            print(f"New best checkpoint: {next_trials} trials.")
        else:
            print(
                "No improvement. "
                f"Consecutive non-improving checkpoints: "
                f"{consecutive_nonimprovements}/{patience}."
            )

        print()

        current_trials = next_trials

        if consecutive_nonimprovements >= patience:
            stop_reason = "patience"
            break

    if stop_reason is None:
        stop_reason = "max_trials"

    summary: dict[str, object] = {
        "method": "cross_fitted_lightgbm_greedy_ensemble_aware_hpo_stopping",
        "completed": True,
        "selected_budget": best_budget,
        "selected_cross_fitted_brier_score": best_score,
        "stopping_checkpoint": current_trials,
        "stop_reason": stop_reason,
        "checkpoint_trials": checkpoint_trials,
        "patience": patience,
        "meta_n_splits": meta_n_splits,
        "history": history
    }

    joblib.dump(summary, summary_path, compress=3)

    return summary
