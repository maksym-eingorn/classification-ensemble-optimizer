# Classification Ensemble Optimizer

A modular Python machine learning project for building a binary classification ensemble optimization pipeline.

## Overview

The project implements data preparation for the Breast Cancer Wisconsin Diagnostic and binary Digits classification datasets, XGBoost and LightGBM hyperparameter tuning using Optuna, and final XGBoost and LightGBM evaluation on the held-out test set.

The data preparation pipeline includes dataset loading, binary target encoding, stratified development/test splitting, feature scaling, and saving prepared arrays and the fitted scaler.

The Breast Cancer Wisconsin target is encoded so that benign samples are class `0` and malignant samples are class `1`. This makes malignant the positive class for probability-based classification metrics and future ensemble evaluation.

The binary Digits dataset contains handwritten zeros and ones, encoded as class `0` and class `1`, respectively.

The XGBoost and LightGBM workflows load prepared development data, select the configured original feature set, run Optuna with stratified K-fold cross-validation, store out-of-fold positive-class probabilities for future ensemble search, save tuning results locally, and evaluate the best tuned models on the held-out test set.

## Project Structure

`main.py` — data preparation pipeline\
`run_xgboost_optuna.py` — XGBoost Optuna tuning workflow\
`run_lightgbm_optuna.py` — LightGBM Optuna tuning workflow\
`evaluate_xgboost.py` — XGBoost evaluation on the test set\
`evaluate_lightgbm.py` — LightGBM evaluation on the test set\
`config.py` — project settings and user-configurable parameters\
`environment.py` — numerical library thread settings for improved reproducibility\
`datasets/loader.py` — dataset dispatcher\
`datasets/breast_cancer.py` — Breast Cancer Wisconsin Diagnostic dataset loading and target encoding\
`datasets/digits_binary.py` — binary Digits dataset loading and filtering\
`preprocessing.py` — stratified development/test splitting and standard feature scaling\
`storage.py` — saving and loading prepared NumPy arrays and fitted preprocessing objects\
`evaluation/validation.py` — validation helpers for final model evaluation\
`evaluation/xgboost_evaluation.py` — XGBoost retraining on the full development set and evaluation on the test set\
`evaluation/lightgbm_evaluation.py` — LightGBM retraining on the full development set and evaluation on the test set\
`tuning/feature_sets.py` — feature set selection for tuning and evaluation workflows\
`tuning/validation.py` — shared validation helpers for feature and target arrays\
`tuning/result_storage.py` — saving and loading Optuna result artifacts\
`tuning/xgboost_optuna.py` — XGBoost Optuna tuning logic\
`tuning/lightgbm_optuna.py` — LightGBM Optuna tuning logic\
`requirements.txt` — Python package dependencies

## How It Works

The data preparation pipeline:

* loads the selected binary classification dataset
* converts the Breast Cancer Wisconsin target to `0 = benign` and `1 = malignant`
* filters the Digits dataset to handwritten zeros and ones
* splits the data into development and test sets
* preserves class proportions through stratified splitting
* scales the original features
* fits the scaler only on the development set
* applies the fitted scaler to both development and test feature matrices
* saves prepared arrays as `.npy` files
* saves the fitted scaler as a `.pkl` file

Generated data artifacts are saved locally in dataset- and split-specific subfolders:

`prepared_data/breast_cancer/split_seed_<DATA_SPLIT_SEED>/`\
`prepared_data/digits_binary/split_seed_<DATA_SPLIT_SEED>/`

The XGBoost and LightGBM Optuna tuning workflows:

* load prepared data for the selected dataset
* choose the configured original feature set
* run model-specific hyperparameter tuning with Optuna
* use stratified K-fold cross-validation on the development set
* compute full out-of-fold and per-fold Brier scores
* store one out-of-fold positive-class probability column per Optuna trial
* save the completed Optuna study, trial numbers, out-of-fold probabilities, Brier scores, and hyperparameters

Generated Optuna artifacts are saved locally in dataset-, split-, feature-set-, and model-specific subfolders:

`optuna_results/breast_cancer/split_seed_<DATA_SPLIT_SEED>/original/xgboost/`\
`optuna_results/breast_cancer/split_seed_<DATA_SPLIT_SEED>/original/lightgbm/`\
`optuna_results/digits_binary/split_seed_<DATA_SPLIT_SEED>/original/xgboost/`\
`optuna_results/digits_binary/split_seed_<DATA_SPLIT_SEED>/original/lightgbm/`

The XGBoost and LightGBM test evaluation workflows:

* load the prepared development and test data for the selected dataset
* choose the same configured original feature set used during tuning
* load the saved Optuna study
* retrain the best model on the full development set
* evaluate the retrained model on the held-out test set
* print the final test Brier score, log loss, ROC AUC, and accuracy

## How to Run

This project is configured for Python 3.14.

On macOS, XGBoost and LightGBM require the OpenMP runtime. If Homebrew is available, install it with:

`brew install libomp`

Install dependencies from the project directory with:

`pip install -r requirements.txt`

Then run:

`python main.py`

The script prepares the selected dataset and saves processed outputs into the corresponding dataset- and split-specific subfolder within `prepared_data/`.

Next, run XGBoost Optuna tuning:

`python run_xgboost_optuna.py`

The XGBoost tuning script loads the prepared development data, runs Optuna-based XGBoost tuning with stratified K-fold cross-validation, and saves results into the corresponding dataset-, split-, feature-set-, and model-specific subfolder within `optuna_results/`.

Also run LightGBM Optuna tuning:

`python run_lightgbm_optuna.py`

The LightGBM tuning script loads the prepared development data, runs Optuna-based LightGBM tuning with stratified K-fold cross-validation, and saves results into the corresponding dataset-, split-, feature-set-, and model-specific subfolder within `optuna_results/`.

To evaluate the best tuned XGBoost model on the test set, run:

`python evaluate_xgboost.py`

To evaluate the best tuned LightGBM model on the test set, run:

`python evaluate_lightgbm.py`

Each evaluation script loads the saved model-specific Optuna study, retrains the best model on the full development set, evaluates it on the held-out test set, and prints the test Brier score, log loss, ROC AUC, and accuracy.

## Configuration

Main user-facing settings are stored in `config.py`.

Important general and data preparation settings include:

`DATASET_NAME` — selected dataset name\
`TEST_SIZE` — test set fraction\
`DATA_SPLIT_SEED` — random seed used only for the development/test split; it also determines the split-specific output subfolder name\
`RANDOM_SEED` — random seed used by Optuna sampling, stratified K-fold splitting, and model randomness, including final retraining\
`PREPARED_DATA_DIR` — root output folder for prepared data

Important model-specific Optuna settings are grouped by model family.

For XGBoost, the settings use the `XGBOOST_` prefix. For LightGBM, the settings use the `LIGHTGBM_` prefix.

`XGBOOST_FEATURE_SET` / `LIGHTGBM_FEATURE_SET` — selected feature set for tuning and evaluation\
`XGBOOST_N_TRIALS` / `LIGHTGBM_N_TRIALS` — number of Optuna trials\
`XGBOOST_N_JOBS` / `LIGHTGBM_N_JOBS` — number of parallel Optuna workers\
`XGBOOST_N_SPLITS` / `LIGHTGBM_N_SPLITS` — number of stratified K-fold cross-validation splits\
`XGBOOST_N_ESTIMATORS_MIN` / `LIGHTGBM_N_ESTIMATORS_MIN` — minimum number of estimators considered by Optuna\
`XGBOOST_N_ESTIMATORS_MAX` / `LIGHTGBM_N_ESTIMATORS_MAX` — maximum number of estimators considered by Optuna\
`XGBOOST_N_ESTIMATORS_STEP` / `LIGHTGBM_N_ESTIMATORS_STEP` — step size for the Optuna search over estimators\
`XGBOOST_VERBOSE` / `LIGHTGBM_VERBOSE` — whether to print trial-level Brier scores

`OPTUNA_RESULTS_DIR` represents the root output folder for Optuna result artifacts.

The currently supported datasets are:

`breast_cancer`\
`digits_binary`

The currently supported XGBoost and LightGBM feature set is:

`original`

## Breast Cancer Wisconsin Diagnostic Dataset

The project uses the Breast Cancer Wisconsin Diagnostic dataset provided by scikit-learn.

The dataset contains 569 samples and 30 numerical input features.

Scikit-learn originally encodes malignant samples as class `0` and benign samples as class `1`. During loading, the project reverses this encoding so that:

`0 = benign`  
`1 = malignant`

The project therefore treats malignant as the positive class.

The full feature matrix and target vector are preserved, together with the development/test split and scaled feature matrices.

## Binary Digits Dataset

The project uses a binary subset of the Digits classification dataset provided by scikit-learn.

The subset contains 360 samples and 64 numerical input features representing flattened 8 × 8 images.

Only handwritten zeros and ones are retained:

`0 = digit zero`  
`1 = digit one`

The project therefore treats digit one as the positive class.

The full filtered feature matrix and target vector are preserved, together with the development/test split and scaled feature matrices.

## XGBoost and LightGBM Optuna Tuning

The XGBoost and LightGBM tuning workflows use Optuna to search over model-specific hyperparameters.

XGBoost and LightGBM tuning currently use the original feature matrix. Scaled original features remain available in the prepared data for future scale-sensitive models but are not supported by the XGBoost or LightGBM tuning workflows.

Each Optuna trial trains one model configuration across all stratified K folds and produces one full out-of-fold positive-class probability vector for the development set.

The full out-of-fold Brier score is minimized by Optuna. Per-fold Brier scores are also retained for each trial.

The final out-of-fold probability matrix has one column per trial and is saved for future ensemble search.

The test set is not used during Optuna tuning. After tuning is complete, the best saved configuration for each model family can be retrained on the full development set and evaluated once on the held-out test set.

## XGBoost and LightGBM Test Evaluation

The XGBoost and LightGBM test evaluation workflows load the saved Optuna study for the configured dataset, development/test split, feature set, and model family.

The best hyperparameters from the corresponding Optuna study are used to retrain a final model on the full development set.

The retrained model is then evaluated on the held-out test set. Each workflow prints the test Brier score, log loss, ROC AUC, and accuracy. Accuracy uses a positive-class probability threshold of 0.5.

This keeps the test set separate from hyperparameter tuning and reserves test-set performance measurement for final model evaluation.

## Reproducibility

The project limits hidden parallelism in numerical libraries through `environment.py`.

For stricter reproducibility, `PYTHONHASHSEED` can be set before launching Python.

The development/test split is controlled by `DATA_SPLIT_SEED`. Changing `DATA_SPLIT_SEED` creates a different held-out split and writes generated artifacts under `split_seed_<DATA_SPLIT_SEED>` subfolders, allowing multiple split experiments for each dataset without overwriting previous results.

The split uses stratification so that the class proportions in the development and test sets remain similar to those in the full dataset.

`RANDOM_SEED` is defined separately so that Optuna sampling, stratified K-fold splitting, and model randomness can remain fixed while the development/test split is varied independently.

XGBoost and LightGBM tuning use seeded Optuna samplers, seeded stratified K-fold splitting, and seeded models. Final retraining for both XGBoost and LightGBM also uses `RANDOM_SEED`. `XGBOOST_N_JOBS = 1` and `LIGHTGBM_N_JOBS = 1` keep Optuna trial execution sequential by default for improved reproducibility.

## Generated Files

The pipeline may generate files such as:

* `.npy` prepared arrays
* `.pkl` fitted preprocessing objects
* `.pkl` Optuna studies and result artifacts
* the `prepared_data/` directory
* the `optuna_results/` directory

These files are ignored by Git because they are generated artifacts rather than source code.

## Why This Project

This project is designed as a clean, extensible foundation for binary classification ensemble experimentation.

It currently emphasizes:

* modular Python architecture
* reproducible data preparation
* stratified development/test splitting
* leakage-aware preprocessing
* Optuna-based XGBoost and LightGBM hyperparameter tuning
* stratified K-fold out-of-fold probability generation
* Brier-score optimization
* final evaluation on the held-out test set
* clean separation of dataset loading, preprocessing, tuning, evaluation, and storage
* a scalable structure for future model comparison and ensemble optimization

## License

Copyright 2026 Maksym Eingorn

Licensed under the Apache License, Version 2.0. See the `LICENSE` file for details.