# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# config.py

# --- General settings ---

RANDOM_SEED = 42

# Controls only the development/test split.
DATA_SPLIT_SEED = 21

# --- Dataset settings ---

# Currently supported: "breast_cancer", "digits_binary"
DATASET_NAME = "breast_cancer"

TEST_SIZE = 0.2

# --- XGBoost Optuna tuning ---

# Supported feature set: "original"
XGBOOST_FEATURE_SET = "original"

XGBOOST_N_TRIALS = 1000
XGBOOST_N_JOBS = 1
XGBOOST_N_SPLITS = 5
XGBOOST_N_ESTIMATORS_MIN = 100
XGBOOST_N_ESTIMATORS_MAX = 1000
XGBOOST_N_ESTIMATORS_STEP = 50
XGBOOST_VERBOSE = True

# --- LightGBM Optuna tuning ---

# Supported feature set: "original"
LIGHTGBM_FEATURE_SET = "original"

LIGHTGBM_N_TRIALS = 1000
LIGHTGBM_N_JOBS = 1
LIGHTGBM_N_SPLITS = 5
LIGHTGBM_N_ESTIMATORS_MIN = 100
LIGHTGBM_N_ESTIMATORS_MAX = 1000
LIGHTGBM_N_ESTIMATORS_STEP = 50
LIGHTGBM_VERBOSE = True

# --- Logistic Regression Optuna tuning ---

# Supported feature set: "original"
# Standardization is fitted separately within each CV training fold.
LOGISTIC_REGRESSION_FEATURE_SET = "original"

LOGISTIC_REGRESSION_N_TRIALS = 100
LOGISTIC_REGRESSION_N_JOBS = 1
LOGISTIC_REGRESSION_N_SPLITS = 5
LOGISTIC_REGRESSION_MAX_ITER = 20000
LOGISTIC_REGRESSION_TOL = 1e-4
LOGISTIC_REGRESSION_VERBOSE = True

# --- Output paths ---

PREPARED_DATA_DIR = "prepared_data"
OPTUNA_RESULTS_DIR = "optuna_results"
