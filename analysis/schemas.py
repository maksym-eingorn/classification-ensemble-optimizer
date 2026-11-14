# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# analysis/schemas.py

from typing import TypedDict


class DistributionSummary(TypedDict):
    mean: float
    median: float
    std: float
    q10: float
    q25: float
    q75: float
    q90: float
    minimum: float
    maximum: float


class CheckpointEvaluationResult(TypedDict):
    data_split_seed: int
    budget: int
    model_name: str
    feature_set: str
    ensemble_method: str
    oof_brier_score: float
    test_brier_score: float
    test_log_loss: float
    test_roc_auc: float
    test_accuracy: float


class StoppingCheckpointRecord(TypedDict):
    data_split_seed: int
    budget: int
    cross_fitted_brier_score: float
    fold_brier_scores: list[float]


class StoppingSeedRecord(TypedDict):
    data_split_seed: int
    selected_budget: int
    selected_cross_fitted_brier_score: float


class CombinedCheckpointRecord(TypedDict):
    data_split_seed: int
    budget: int
    cross_fitted_brier_score: float
    oof_brier_score: float
    test_brier_score: float
    test_log_loss: float
    test_roc_auc: float
    test_accuracy: float
    is_cf_selected: bool


class PerSeedSummary(TypedDict):
    data_split_seed: int

    selected_budget: int
    selected_cross_fitted_brier_score: float
    selected_oof_brier_score: float
    selected_test_brier_score: float
    selected_test_log_loss: float
    selected_test_roc_auc: float
    selected_test_accuracy: float

    full_budget: int
    full_oof_brier_score: float
    full_test_brier_score: float
    full_test_log_loss: float
    full_test_roc_auc: float
    full_test_accuracy: float

    selected_minus_full_test_brier: float
    selected_minus_full_test_log_loss: float
    selected_minus_full_test_roc_auc: float
    selected_minus_full_test_accuracy: float

    oracle_budget: int
    oracle_budget_count: int
    oracle_test_brier_score: float
    selection_regret: float

    selected_budget_is_oracle_budget: bool
    selected_has_oracle_test_brier: bool
    full_has_oracle_test_brier: bool
    test_oracle_better_than_full: bool

    budget_distance_to_oracle: int
    cf_test_pearson_correlation: float | None

    full_oof_better_than_selected: bool
    selected_test_better_than_full: bool
    oof_full_but_selected_test_wins: bool


class FixedBudgetComparison(TypedDict):
    budget: int
    n_seeds: int

    adaptive_wins: int
    adaptive_losses: int
    ties: int

    mean_adaptive_test_brier: float
    mean_fixed_test_brier: float

    mean_selected_minus_fixed_test_brier: float
    median_selected_minus_fixed_test_brier: float
    std_selected_minus_fixed_test_brier: float
    q10_selected_minus_fixed_test_brier: float
    q25_selected_minus_fixed_test_brier: float
    q75_selected_minus_fixed_test_brier: float
    q90_selected_minus_fixed_test_brier: float
    min_selected_minus_fixed_test_brier: float
    max_selected_minus_fixed_test_brier: float


class BudgetCurveSummary(TypedDict):
    budget: int
    n_seeds: int

    mean_cross_fitted_brier: float
    median_cross_fitted_brier: float
    std_cross_fitted_brier: float

    mean_oof_brier: float
    median_oof_brier: float
    std_oof_brier: float

    mean_test_brier: float
    median_test_brier: float
    std_test_brier: float

    mean_test_log_loss: float
    mean_test_roc_auc: float
    mean_test_accuracy: float
