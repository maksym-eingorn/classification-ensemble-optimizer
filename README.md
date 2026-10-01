# Classification Ensemble Optimizer

A modular Python machine learning project (with C++/OpenMP acceleration) for building a binary classification ensemble optimization pipeline.

## Overview

The project implements data preparation for the Breast Cancer Wisconsin Diagnostic and binary Digits classification datasets, XGBoost, LightGBM, and Logistic Regression hyperparameter tuning using Optuna, final individual model evaluation on the held-out test set, exact Brier-optimal convex triplet ensemble search, triplet ensemble evaluation on the held-out test set, Caruana-style greedy ensemble selection, and greedy ensemble evaluation on the held-out test set.

The data preparation pipeline includes dataset loading, binary target encoding, stratified development/test splitting, feature scaling, and saving prepared arrays and the fitted scaler.

The Breast Cancer Wisconsin target is encoded so that benign samples are class `0` and malignant samples are class `1`. This makes malignant the positive class for probability-based classification metrics and ensemble evaluation.

The binary Digits dataset contains handwritten zeros and ones, encoded as class `0` and class `1`, respectively.

The XGBoost, LightGBM, and Logistic Regression workflows load prepared development data, select the configured original feature set, run Optuna with stratified K-fold cross-validation, store out-of-fold positive-class probabilities for ensemble search and selection, save tuning results locally, and evaluate the best tuned models on the held-out test set.

The triplet ensemble workflow combines saved out-of-fold positive-class probability columns from the configured model families, searches all 3-model combinations exactly, optimizes nonnegative weights constrained to sum to 1 to minimize the OOF Brier score, saves the retained OOF-ranked triplets, and evaluates the saved triplet ensembles on the held-out test set.

The greedy ensemble workflow uses the same saved out-of-fold probability matrix, builds a convex ensemble through Caruana-style greedy selection, allows repeated model selections when configured, converts selection counts into final ensemble weights, saves the chosen greedy ensemble, and evaluates it on the held-out test set.

## Project Structure

`main.py` — data preparation pipeline\
`run_xgboost_optuna.py` — XGBoost Optuna tuning workflow\
`run_lightgbm_optuna.py` — LightGBM Optuna tuning workflow\
`run_logistic_regression_optuna.py` — Logistic Regression Optuna tuning workflow\
`run_triplet_search.py` — exact Brier-optimal convex triplet ensemble search workflow\
`run_greedy_ensemble_selection.py` — Caruana-style greedy ensemble selection workflow\
`evaluate_xgboost.py` — XGBoost evaluation on the test set\
`evaluate_lightgbm.py` — LightGBM evaluation on the test set\
`evaluate_logistic_regression.py` — Logistic Regression evaluation on the test set\
`evaluate_triplet_ensembles.py` — saved triplet ensemble evaluation on the test set\
`evaluate_greedy_ensemble.py` — saved greedy ensemble evaluation on the test set\
`config.py` — project settings and user-configurable parameters\
`environment.py` — numerical library thread settings for improved reproducibility\
`datasets/loader.py` — dataset dispatcher\
`datasets/breast_cancer.py` — Breast Cancer Wisconsin Diagnostic dataset loading and target encoding\
`datasets/digits_binary.py` — binary Digits dataset loading and filtering\
`preprocessing.py` — stratified development/test splitting and standard feature scaling\
`storage.py` — saving and loading prepared NumPy arrays and fitted preprocessing objects\
`ensemble/oof_matrix.py` — construction of a unified out-of-fold probability matrix\
`ensemble/triplet_search.py` — Python wrapper for the native triplet search extension\
`ensemble/greedy_selection.py` — Caruana-style greedy ensemble selection logic\
`evaluation/validation.py` — validation helpers for final model evaluation\
`evaluation/xgboost_evaluation.py` — XGBoost retraining on the full development set and evaluation on the test set\
`evaluation/lightgbm_evaluation.py` — LightGBM retraining on the full development set and evaluation on the test set\
`evaluation/logistic_regression_evaluation.py` — Logistic Regression retraining on the full development set and evaluation on the test set\
`evaluation/ensemble_evaluation.py` — triplet and greedy ensemble retraining on the full development set and evaluation on the test set\
`tuning/feature_sets.py` — feature set selection for tuning and evaluation workflows\
`tuning/validation.py` — shared validation helpers for feature and target arrays\
`tuning/result_storage.py` — saving and loading Optuna result artifacts\
`tuning/xgboost_optuna.py` — XGBoost Optuna tuning logic\
`tuning/lightgbm_optuna.py` — LightGBM Optuna tuning logic\
`tuning/logistic_regression_optuna.py` — Logistic Regression Optuna tuning logic\
`native/triplet_search.cpp` — C++/OpenMP implementation of exact exhaustive convex triplet search\
`setup_triplet_search.py` — build script for the native triplet search extension\
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

The XGBoost, LightGBM, and Logistic Regression Optuna tuning workflows:

* load prepared data for the selected dataset
* choose the configured original feature set
* run model-specific hyperparameter tuning with Optuna
* use stratified K-fold cross-validation on the development set
* standardize Logistic Regression features separately within each cross-validation training fold and apply the fitted scaler to the corresponding validation fold
* compute full out-of-fold and per-fold Brier scores
* store one out-of-fold positive-class probability column per Optuna trial
* save the completed Optuna study, trial numbers, out-of-fold probabilities, Brier scores, and hyperparameters

Generated Optuna artifacts are saved locally in dataset-, split-, feature-set-, and model-specific subfolders:

`optuna_results/breast_cancer/split_seed_<DATA_SPLIT_SEED>/original/xgboost/`\
`optuna_results/breast_cancer/split_seed_<DATA_SPLIT_SEED>/original/lightgbm/`\
`optuna_results/breast_cancer/split_seed_<DATA_SPLIT_SEED>/original/logistic_regression/`\
`optuna_results/digits_binary/split_seed_<DATA_SPLIT_SEED>/original/xgboost/`\
`optuna_results/digits_binary/split_seed_<DATA_SPLIT_SEED>/original/lightgbm/`\
`optuna_results/digits_binary/split_seed_<DATA_SPLIT_SEED>/original/logistic_regression/`

The XGBoost, LightGBM, and Logistic Regression test evaluation workflows:

* load the prepared development and test data for the selected dataset
* choose the same configured original feature set used during tuning
* load the saved model-specific Optuna study
* for Logistic Regression, fit a `StandardScaler` on the full development feature matrix and apply it to both the development and test feature matrices before final model retraining and evaluation
* retrain the best model on the full development set
* evaluate the retrained model on the held-out test set
* print the final test Brier score, log loss, ROC AUC, and accuracy

The convex triplet ensemble search workflow:

* loads saved out-of-fold positive-class probabilities from the configured XGBoost, LightGBM, and Logistic Regression model families
* builds one unified out-of-fold probability matrix
* searches all 3-model combinations exactly
* finds the Brier-optimal convex weights for each triplet, with nonnegative weights constrained to sum to 1
* ranks triplets by OOF Brier score
* retains up to the configured number of top OOF-ranked triplets
* saves the retained triplet metadata and out-of-fold probabilities

The greedy ensemble selection workflow:

* loads saved out-of-fold positive-class probabilities from the configured XGBoost, LightGBM, and Logistic Regression model families
* builds one unified out-of-fold probability matrix
* runs Caruana-style greedy ensemble selection
* optionally restricts the candidate pool with `GREEDY_POOL_SIZE`
* allows repeated model selections when `GREEDY_ALLOW_REPEATS = True`
* converts selection counts into convex weights
* saves the greedy ensemble metadata and out-of-fold probabilities

Generated ensemble search and selection artifacts are saved locally within dataset- and split-specific subfolders of `ensemble_results/`.

`ensemble_results/breast_cancer/split_seed_<DATA_SPLIT_SEED>/xgboost-original__lightgbm-original__logistic_regression-original/triplet_convex/`\
`ensemble_results/breast_cancer/split_seed_<DATA_SPLIT_SEED>/xgboost-original__lightgbm-original__logistic_regression-original/greedy_selection/`\
`ensemble_results/digits_binary/split_seed_<DATA_SPLIT_SEED>/xgboost-original__lightgbm-original__logistic_regression-original/triplet_convex/`\
`ensemble_results/digits_binary/split_seed_<DATA_SPLIT_SEED>/xgboost-original__lightgbm-original__logistic_regression-original/greedy_selection/`

The triplet ensemble test evaluation workflow:

* loads the saved OOF-ranked triplet metadata
* retrains the selected base models on the full development set using their saved trial hyperparameters
* for selected Logistic Regression models, fits a `StandardScaler` on the full development feature matrix and applies it to both the development and test feature matrices
* combines the selected models' positive-class test probabilities using the saved convex weights
* evaluates the retained triplet ensembles on the held-out test set
* saves triplet test metadata and test probabilities
* prints the OOF Brier score and final test Brier score, log loss, ROC AUC, and accuracy for the best OOF-ranked triplet and, for diagnostic purposes only, for the retained triplet with the lowest test Brier score

The greedy ensemble test evaluation workflow:

* loads the saved greedy ensemble metadata
* retrains the selected base models on the full development set using their saved trial hyperparameters
* for selected Logistic Regression models, fits a `StandardScaler` on the full development feature matrix and applies it to both the development and test feature matrices
* combines the selected models' positive-class test probabilities using the saved convex weights
* evaluates the greedy ensemble on the held-out test set
* saves greedy ensemble test metadata and test probabilities
* prints the OOF Brier score and final test Brier score, log loss, ROC AUC, and accuracy

## How to Run

This project is configured for Python 3.14.

On macOS, XGBoost, LightGBM, and the native triplet search require the OpenMP runtime, and the native triplet search also requires Eigen. If Homebrew is available, install them with:

`brew install libomp eigen`

Install dependencies from the project directory with:

`pip install -r requirements.txt`

Then run:

`python main.py`

The script prepares the selected dataset and saves processed outputs into the corresponding dataset- and split-specific subfolder within `prepared_data/`.

Next, run XGBoost Optuna tuning:

`python run_xgboost_optuna.py`

Also run LightGBM Optuna tuning:

`python run_lightgbm_optuna.py`

In addition, run Logistic Regression Optuna tuning:

`python run_logistic_regression_optuna.py`

Each tuning script loads the prepared development data, runs Optuna-based tuning with stratified K-fold cross-validation (and fold-local standardization in the case of Logistic Regression), and saves results into the corresponding dataset-, split-, feature-set-, and model-specific subfolder within `optuna_results/`.

To evaluate the best tuned XGBoost model on the test set, run:

`python evaluate_xgboost.py`

To evaluate the best tuned LightGBM model on the test set, run:

`python evaluate_lightgbm.py`

To evaluate the best tuned Logistic Regression model on the test set, run:

`python evaluate_logistic_regression.py`

Each evaluation script loads the saved model-specific Optuna study, retrains the best model on the full development set, evaluates it on the held-out test set, and prints the test Brier score, log loss, ROC AUC, and accuracy.

Before running triplet ensemble search, build the native C++ extension:

`python setup_triplet_search.py build_ext --inplace`

The native extension requires Eigen and OpenMP. If Eigen or the macOS OpenMP runtime is not located in one of the default paths used by the build script, set `EIGEN_INCLUDE_DIR` or `LIBOMP_PREFIX`, respectively.

Then run exact Brier-optimal convex triplet ensemble search:

`python run_triplet_search.py`

The triplet search script loads saved Optuna out-of-fold positive-class probabilities, builds a unified out-of-fold probability matrix, runs exact exhaustive convex triplet search, and saves the retained OOF-ranked triplet results into the corresponding dataset- and split-specific subfolder within `ensemble_results/`.

Evaluate the retained triplet ensembles on the held-out test set:

`python evaluate_triplet_ensembles.py`

The triplet evaluation script retrains the selected base models on the full development set, combines their positive-class test probabilities using the saved convex weights, evaluates the retained triplets on the held-out test set, and saves the resulting test artifacts within `ensemble_results/`.

Also, run Caruana-style greedy ensemble selection:

`python run_greedy_ensemble_selection.py`

The greedy ensemble selection script does not require building the native C++ extension. It loads the saved Optuna out-of-fold probability artifacts, builds a unified out-of-fold probability matrix, runs greedy ensemble selection, and saves the selected ensemble into the corresponding dataset- and split-specific subfolder within `ensemble_results/`.

Finally, evaluate the saved greedy ensemble on the held-out test set:

`python evaluate_greedy_ensemble.py`

The greedy ensemble evaluation script retrains the selected base models on the full development set, combines their positive-class test probabilities using the saved convex weights, evaluates the ensemble on the held-out test set, and saves the resulting test artifacts within `ensemble_results/`.

## Configuration

Main user-facing settings are stored in `config.py`.

Important general and data preparation settings include:

`DATASET_NAME` — selected dataset name\
`TEST_SIZE` — test set fraction\
`DATA_SPLIT_SEED` — random seed used only for the development/test split; it also determines the split-specific output subfolder name\
`RANDOM_SEED` — random seed used by Optuna sampling, stratified K-fold splitting, model randomness, greedy ensemble selection tie-breaking, and final retraining\
`PREPARED_DATA_DIR` — root output folder for prepared data

Important model-specific Optuna settings are grouped by model family.

For XGBoost, the settings use the `XGBOOST_` prefix. For LightGBM, the settings use the `LIGHTGBM_` prefix. For Logistic Regression, the settings use the `LOGISTIC_REGRESSION_` prefix.

`XGBOOST_FEATURE_SET` / `LIGHTGBM_FEATURE_SET` / `LOGISTIC_REGRESSION_FEATURE_SET` — selected feature set for tuning and evaluation\
`XGBOOST_N_TRIALS` / `LIGHTGBM_N_TRIALS` / `LOGISTIC_REGRESSION_N_TRIALS` — number of Optuna trials\
`XGBOOST_N_JOBS` / `LIGHTGBM_N_JOBS` / `LOGISTIC_REGRESSION_N_JOBS` — number of parallel Optuna workers\
`XGBOOST_N_SPLITS` / `LIGHTGBM_N_SPLITS` / `LOGISTIC_REGRESSION_N_SPLITS` — number of stratified K-fold cross-validation splits\
`XGBOOST_VERBOSE` / `LIGHTGBM_VERBOSE` / `LOGISTIC_REGRESSION_VERBOSE` — whether to print trial-level Brier scores

Important tree-model estimator settings are:

`XGBOOST_N_ESTIMATORS_MIN` / `LIGHTGBM_N_ESTIMATORS_MIN` — minimum number of estimators considered by Optuna\
`XGBOOST_N_ESTIMATORS_MAX` / `LIGHTGBM_N_ESTIMATORS_MAX` — maximum number of estimators considered by Optuna\
`XGBOOST_N_ESTIMATORS_STEP` / `LIGHTGBM_N_ESTIMATORS_STEP` — step size for the Optuna search over estimators

Important Logistic Regression optimization settings are:

`LOGISTIC_REGRESSION_MAX_ITER` — maximum number of Logistic Regression optimization iterations\
`LOGISTIC_REGRESSION_TOL` — optimization tolerance for Logistic Regression

Important convex triplet ensemble search settings are:

`TRIPLET_TOP_N` — maximum number of top OOF-ranked convex triplets retained from the native exhaustive search\
`TRIPLET_N_THREADS` — number of OpenMP threads used by the native triplet search

Important greedy ensemble selection settings are:

`GREEDY_POOL_SIZE` — number of the best individual models considered by greedy selection; set to `None` to use the full model zoo\
`GREEDY_ENSEMBLE_SIZE` — maximum number of greedy selection steps\
`GREEDY_ALLOW_REPEATS` — whether the same model may be selected multiple times

`OPTUNA_RESULTS_DIR` represents the root output folder for Optuna result artifacts.

`ENSEMBLE_RESULTS_DIR` represents the root output folder for triplet search, greedy selection, and ensemble evaluation artifacts.

The currently supported datasets are:

`breast_cancer`\
`digits_binary`

The currently supported XGBoost, LightGBM, and Logistic Regression feature set is:

`original`

For Logistic Regression, `original` denotes the raw prepared feature matrix. Standardization is performed within the Logistic Regression workflow rather than by selecting the prepared `original_scaled` feature set.

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

## XGBoost, LightGBM, and Logistic Regression Optuna Tuning

The XGBoost, LightGBM, and Logistic Regression tuning workflows use Optuna to search over model-specific hyperparameters.

XGBoost, LightGBM, and Logistic Regression tuning currently select the original feature matrix. XGBoost and LightGBM train directly on these features. Logistic Regression standardizes the features separately within each cross-validation training fold and applies the fitted scaler to the corresponding validation fold. The prepared scaled original feature matrices remain available but are not selected directly by the current model-specific tuning workflows.

Each Optuna trial trains one model configuration across all stratified K folds and produces one full out-of-fold positive-class probability vector for the development set.

The full out-of-fold Brier score is minimized by Optuna. Per-fold Brier scores are also retained for each trial.

The final out-of-fold probability matrix for each model family has one column per trial and is saved for triplet ensemble search and greedy ensemble selection.

The test set is not used during Optuna tuning. After tuning is complete, the best saved configuration for each model family can be retrained on the full development set and evaluated once on the held-out test set.

## XGBoost, LightGBM, and Logistic Regression Test Evaluation

The XGBoost, LightGBM, and Logistic Regression test evaluation workflows load the saved Optuna study for the configured dataset, development/test split, feature set, and model family.

The best hyperparameters from the corresponding Optuna study are used to retrain a final model on the full development set.

The retrained model is then evaluated on the held-out test set. Each workflow prints the test Brier score, log loss, ROC AUC, and accuracy. Accuracy uses a positive-class probability threshold of 0.5.

This keeps the test set separate from hyperparameter tuning and reserves test-set performance measurement for final model evaluation.

## Convex Triplet Ensemble Search

The convex triplet ensemble search workflow uses saved out-of-fold positive-class probabilities from the configured XGBoost, LightGBM, and Logistic Regression Optuna runs.

It builds a unified out-of-fold probability matrix where each column corresponds to one saved Optuna trial. It then searches every 3-model combination exactly. For each triplet, the search minimizes the OOF Brier score subject to nonnegative weights whose sum is 1. The retained triplets are ranked by increasing OOF Brier score.

The exhaustive search is implemented in native C++ with Eigen and OpenMP acceleration.

The test set is not used during triplet ensemble search.

Triplet search artifacts are saved within the corresponding dataset- and split-specific `triplet_convex` subfolder of `ensemble_results/` and include:

* `base_model_column_metadata.pkl`
* `triplet_oof_metadata.pkl`
* `triplet_oof_probabilities.pkl`

## Greedy Ensemble Selection

The greedy ensemble selection workflow uses saved out-of-fold positive-class probabilities from the configured XGBoost, LightGBM, and Logistic Regression Optuna runs.

It builds a unified out-of-fold probability matrix where each column corresponds to one saved Optuna trial. It then runs Caruana-style greedy ensemble selection. The algorithm first restricts the candidate set to the configured pool of the best individual models, unless the full model zoo is used. It then builds a convex ensemble by repeatedly adding the model that gives the lowest OOF Brier score after averaging the selected probability vectors. If repeats are enabled, the same model may be selected multiple times. Final convex weights are selection counts divided by the total number of completed greedy selections.

The test set is not used during greedy ensemble selection.

Greedy selection artifacts are saved within the corresponding dataset- and split-specific subfolder of `ensemble_results/` and include:

* `base_model_column_metadata.pkl`
* `greedy_oof_metadata.pkl`
* `greedy_oof_probabilities.pkl`

## Triplet Ensemble Test Evaluation

The triplet ensemble test evaluation workflow loads the saved OOF-ranked triplet metadata, retrains the selected base models on the full development set using the saved trial hyperparameters, combines their positive-class test probabilities using the saved convex weights, and evaluates each retained triplet ensemble on the held-out test set.

For selected Logistic Regression models, a `StandardScaler` is fitted on the full development feature matrix and applied to both the development and test feature matrices before final retraining and probability prediction.

The script reports the OOF Brier score and final test Brier score, log loss, ROC AUC, and accuracy for the best OOF-ranked triplet. It also reports the retained triplet with the lowest test Brier score as diagnostic information only. The test set should not be used to choose the final model selection rule. Accuracy uses a positive-class probability threshold of 0.5.

Triplet ensemble test artifacts are saved within the corresponding dataset- and split-specific `triplet_convex` subfolder of `ensemble_results/` and include:

* `triplet_test_metadata.pkl`
* `triplet_test_probabilities.pkl`

## Greedy Ensemble Test Evaluation

The greedy ensemble test evaluation workflow loads the saved greedy ensemble metadata, retrains the selected base models on the full development set using the saved selected-trial hyperparameters, combines their positive-class test probabilities using the saved convex weights, and evaluates the greedy ensemble on the held-out test set.

For selected Logistic Regression models, a `StandardScaler` is fitted on the full development feature matrix and applied to both the development and test feature matrices before final retraining and probability prediction.

The script reports the greedy ensemble OOF Brier score and final test Brier score, log loss, ROC AUC, and accuracy. Accuracy uses a positive-class probability threshold of 0.5.

Greedy ensemble test artifacts are saved within the corresponding dataset- and split-specific subfolder of `ensemble_results/` and include:

* `greedy_test_metadata.pkl`
* `greedy_test_probabilities.pkl`

## Reproducibility

The project limits hidden parallelism in numerical libraries through `environment.py`.

For stricter reproducibility, `PYTHONHASHSEED` can be set before launching Python.

The development/test split is controlled by `DATA_SPLIT_SEED`. Changing `DATA_SPLIT_SEED` creates a different held-out split and writes generated artifacts under `split_seed_<DATA_SPLIT_SEED>` subfolders, allowing multiple split experiments for each dataset without overwriting previous results.

The split uses stratification so that the class proportions in the development and test sets remain similar to those in the full dataset.

`RANDOM_SEED` is defined separately so that Optuna sampling, stratified K-fold splitting, and model randomness can remain fixed while the development/test split is varied independently.

XGBoost, LightGBM, and Logistic Regression tuning use seeded Optuna samplers, seeded stratified K-fold splitting, and seeded models. Final retraining for all three model families also uses `RANDOM_SEED`. Greedy ensemble selection uses `RANDOM_SEED` for deterministic tie-breaking, and triplet and greedy ensemble test evaluation use the same seed when retraining the selected base models. `XGBOOST_N_JOBS = 1`, `LIGHTGBM_N_JOBS = 1`, and `LOGISTIC_REGRESSION_N_JOBS = 1` keep Optuna trial execution sequential by default for improved reproducibility.

The native triplet search uses OpenMP for parallel exhaustive search. Its thread count is controlled by `TRIPLET_N_THREADS`.

## Generated Files

The pipeline may generate files such as:

* `.npy` prepared arrays
* `.pkl` fitted preprocessing objects
* `.pkl` Optuna studies and result artifacts
* `.pkl` ensemble search, selection, and evaluation artifacts
* compiled native extension files
* the `prepared_data/` directory
* the `optuna_results/` directory
* the `ensemble_results/` directory
* the `build/` directory

These files are ignored by Git because they are generated artifacts rather than source code.

## Why This Project

This project is designed as a clean, extensible foundation for binary classification ensemble experimentation.

It currently emphasizes:

* modular Python architecture
* reproducible data preparation
* stratified development/test splitting
* leakage-aware preprocessing
* Optuna-based XGBoost, LightGBM, and Logistic Regression hyperparameter tuning
* stratified K-fold out-of-fold probability generation
* exact exhaustive Brier-optimal convex triplet ensemble search over saved out-of-fold probabilities
* native C++/OpenMP acceleration for combinatorial triplet search
* Caruana-style greedy ensemble selection over saved out-of-fold probabilities
* Brier-score optimization
* final evaluation of individual models, triplet ensembles, and greedy ensembles on the held-out test set
* clean separation of dataset loading, preprocessing, tuning, ensemble search, selection, evaluation, and storage
* a scalable structure for future model comparison and ensemble optimization

## License

Copyright 2026 Maksym Eingorn

Licensed under the Apache License, Version 2.0. See the `LICENSE` file for details.