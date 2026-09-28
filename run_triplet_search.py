# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# run_triplet_search.py

from environment import configure_environment

configure_environment()

from pathlib import Path
from typing import cast

import joblib
import numpy as np

import config
from ensemble.oof_matrix import build_oof_matrix, load_oof_result_block
from ensemble.triplet_search import find_top_triplet_ensembles
from storage import (
    get_data_split_name, get_dataset_output_dir, load_prepared_data
)
from tuning.feature_sets import (
    validate_lightgbm_feature_set,
    validate_logistic_regression_feature_set,
    validate_xgboost_feature_set
)


def _get_model_specs() -> list[tuple[str, str]]:
    """Return model names and configured feature sets for triplet search."""
    return [
        ("xgboost", config.XGBOOST_FEATURE_SET),
        ("lightgbm", config.LIGHTGBM_FEATURE_SET),
        ("logistic_regression", config.LOGISTIC_REGRESSION_FEATURE_SET)
    ]


def _validate_model_feature_sets() -> None:
    """Validate configured feature sets for all triplet base model families."""
    validate_xgboost_feature_set(config.XGBOOST_FEATURE_SET)
    validate_lightgbm_feature_set(config.LIGHTGBM_FEATURE_SET)
    validate_logistic_regression_feature_set(
        config.LOGISTIC_REGRESSION_FEATURE_SET
    )


def _get_triplet_output_dir(
    root_dir: str | Path,
    dataset_name: str,
    data_split_seed: int,
    model_specs: list[tuple[str, str]]
) -> Path:
    """Return the output directory for convex triplet search results."""
    spec_name = "__".join(
        f"{model_name}-{feature_set}"
        for model_name, feature_set in model_specs
    )

    return (
        Path(root_dir)
        / dataset_name
        / get_data_split_name(data_split_seed)
        / spec_name
        / "triplet_convex"
    )


def main() -> None:
    """Run exact exhaustive Brier-optimal convex triplet ensemble search."""
    _validate_model_feature_sets()
    model_specs = _get_model_specs()

    prepared_data_dir = get_dataset_output_dir(
        config.PREPARED_DATA_DIR, config.DATASET_NAME, config.DATA_SPLIT_SEED
    )

    arrays, _ = load_prepared_data(prepared_data_dir)

    y_dev = np.asarray(arrays["y_dev"], dtype=np.float64)

    blocks = [
        load_oof_result_block(
            config.OPTUNA_RESULTS_DIR,
            config.DATASET_NAME,
            config.DATA_SPLIT_SEED,
            feature_set,
            model_name
        )
        for model_name, feature_set in model_specs
    ]

    P, column_metadata = build_oof_matrix(blocks)

    triplet_metadata, triplet_oof_probabilities = find_top_triplet_ensembles(
        P,
        y_dev,
        top_n=config.TRIPLET_TOP_N,
        n_threads=config.TRIPLET_N_THREADS
    )

    if not triplet_metadata:
        raise RuntimeError("Triplet search did not return any valid triplets.")

    for triplet in triplet_metadata:
        i = cast(int, triplet["i"])
        j = cast(int, triplet["j"])
        k = cast(int, triplet["k"])

        triplet["models"] = [
            dict(column_metadata[i]),
            dict(column_metadata[j]),
            dict(column_metadata[k])
        ]

    output_dir = _get_triplet_output_dir(
        config.ENSEMBLE_RESULTS_DIR,
        config.DATASET_NAME,
        config.DATA_SPLIT_SEED,
        model_specs
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        column_metadata,
        output_dir / "base_model_column_metadata.pkl",
        compress=3
    )
    joblib.dump(
        triplet_metadata,
        output_dir / "triplet_oof_metadata.pkl",
        compress=3
    )
    joblib.dump(
        triplet_oof_probabilities.astype(np.float32),
        output_dir / "triplet_oof_probabilities.pkl",
        compress=3
    )

    best_triplet = triplet_metadata[0]

    best_i = cast(int, best_triplet["i"])
    best_j = cast(int, best_triplet["j"])
    best_k = cast(int, best_triplet["k"])
    best_wi = cast(float, best_triplet["wi"])
    best_wj = cast(float, best_triplet["wj"])
    best_wk = cast(float, best_triplet["wk"])
    best_oof_brier_score = cast(float, best_triplet["oof_brier_score"])

    print(f"Unified out-of-fold probability matrix shape: {P.shape}")
    print(
        "Triplet out-of-fold probability matrix shape: "
        f"{triplet_oof_probabilities.shape}"
    )
    print(f"Triplet search results saved to: {output_dir}")
    print()
    print(
        f"Best triplet OOF Brier score = {best_oof_brier_score:.5f}, "
        f"columns ({best_i}, {best_j}, {best_k})"
    )
    print(
        f"Best triplet weights: ({best_wi:.5f}, {best_wj:.5f}, {best_wk:.5f})"
    )


if __name__ == "__main__":
    main()
