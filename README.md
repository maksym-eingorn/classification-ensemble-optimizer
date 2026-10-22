# Classification Ensemble Optimizer

A modular Python machine learning project (with C++/OpenMP acceleration) for building a binary classification ensemble optimization pipeline.

## Overview

The project implements data preparation for the Breast Cancer Wisconsin Diagnostic and binary Digits classification datasets, XGBoost, LightGBM, and Logistic Regression hyperparameter tuning using Optuna, final individual model evaluation on the held-out test set, exact Brier-optimal convex triplet ensemble search, triplet ensemble evaluation on the held-out test set, Caruana-style greedy ensemble selection, greedy ensemble evaluation on the held-out test set, and a LightGBM-specific ensemble-aware HPO budget-selection workflow based on cross-fitted downstream greedy ensemble performance.

The data preparation pipeline includes dataset loading, binary target encoding, stratified development/test splitting, feature scaling, and saving prepared arrays and the fitted scaler.

The Breast Cancer Wisconsin target is encoded so that benign samples are class `0` and malignant samples are class `1`. This makes malignant the positive class for probability-based classification metrics and ensemble evaluation.

The binary Digits dataset contains handwritten zeros and ones, encoded as class `0` and class `1`, respectively.

The XGBoost, LightGBM, and Logistic Regression workflows load prepared development data, select the configured original feature set, run Optuna with stratified K-fold cross-validation, store out-of-fold positive-class probabilities for ensemble search and selection, save tuning results locally, and evaluate the best tuned models on the held-out test set.

The triplet ensemble workflow combines saved out-of-fold positive-class probability columns from the configured model families, searches all 3-model combinations exactly, optimizes nonnegative weights constrained to sum to 1 to minimize the OOF Brier score, saves the retained OOF-ranked triplets, and evaluates the saved triplet ensembles on the held-out test set.

The greedy ensemble workflow uses the same saved out-of-fold probability matrix, builds a convex ensemble through Caruana-style greedy selection, allows repeated model selections when configured, converts selection counts into final ensemble weights, saves the chosen greedy ensemble, and evaluates it on the held-out test set.

The LightGBM ensemble-aware HPO budget-selection workflow evaluates candidate library size using meta-holdout observations that are not used by the corresponding Optuna trajectories. It partitions the development set into stratified meta-folds, advances one independent LightGBM Optuna trajectory per meta-fold in checkpoint blocks, constructs a LightGBM-only greedy ensemble from the candidates available at each checkpoint, retrains the selected ensemble members on the corresponding meta-training set, and evaluates the ensemble on the untouched meta-holdout set. The held-out predictions from all meta-folds are combined into one cross-fitted development-set probability vector, whose Brier score determines the preferred Optuna candidate budget without using the outer test set.

The selected candidate budget can then be applied to the already stored sequential full-development LightGBM Optuna trajectory. The project builds one greedy ensemble from the first `N` stored trials, where `N` is the selected budget, and another from the complete stored candidate zoo, retrains both on the full development set, and compares them on the untouched outer test set.

## Project Structure

`main.py` — data preparation pipeline\
`run_xgboost_optuna.py` — XGBoost Optuna tuning workflow\
`run_lightgbm_optuna.py` — LightGBM Optuna tuning workflow\
`run_logistic_regression_optuna.py` — Logistic Regression Optuna tuning workflow\
`run_triplet_search.py` — exact Brier-optimal convex triplet ensemble search workflow\
`run_greedy_ensemble_selection.py` — Caruana-style greedy ensemble selection workflow\
`run_lightgbm_ensemble_stopping.py` — cross-fitted LightGBM ensemble-aware HPO budget-selection workflow\
`run_lightgbm_stopped_greedy_selection.py` — selected-budget and full-budget LightGBM greedy ensemble construction\
`evaluate_xgboost.py` — XGBoost evaluation on the test set\
`evaluate_lightgbm.py` — LightGBM evaluation on the test set\
`evaluate_logistic_regression.py` — Logistic Regression evaluation on the test set\
`evaluate_triplet_ensembles.py` — saved triplet ensemble evaluation on the test set\
`evaluate_greedy_ensemble.py` — saved greedy ensemble evaluation on the test set\
`evaluate_lightgbm_stopping_comparison.py` — selected-budget versus full-budget LightGBM greedy ensemble test comparison\
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
`evaluation/lightgbm_greedy_holdout_evaluation.py` — LightGBM-only greedy ensemble retraining and evaluation on an untouched meta-holdout set\
`stopping/lightgbm_ensemble_stopping.py` — checkpointed cross-fitted LightGBM greedy ensemble-aware HPO budget-selection logic\
`tuning/feature_sets.py` — feature set selection for tuning and evaluation workflows\
`tuning/validation.py` — shared validation helpers for feature and target arrays\
`tuning/result_storage.py` — saving and loading Optuna result artifacts\
`tuning/xgboost_optuna.py` — XGBoost Optuna tuning logic\
`tuning/lightgbm_optuna.py` — LightGBM Optuna tuning logic and shared trial suggestion/out-of-fold evaluation helpers\
`tuning/lightgbm_incremental_optuna.py` — incremental block execution and continuation of sequential LightGBM Optuna studies\
`tuning/logistic_regression_optuna.py` — Logistic Regression Optuna tuning logic\
`native/triplet_search.cpp` — C++/OpenMP implementation of exact exhaustive convex triplet search\
`setup_triplet_search.py` — build script for the native triplet search extension\
`requirements.txt` — Python package dependencies

## How It Works

### The data preparation pipeline:

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

### The XGBoost, LightGBM, and Logistic Regression Optuna tuning workflows:

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

### The XGBoost, LightGBM, and Logistic Regression test evaluation workflows:

* load the prepared development and test data for the selected dataset
* choose the same configured original feature set used during tuning
* load the saved model-specific Optuna study
* for Logistic Regression, fit a `StandardScaler` on the full development feature matrix and apply it to both the development and test feature matrices before final model retraining and evaluation
* retrain the best model on the full development set
* evaluate the retrained model on the held-out test set
* print the final test Brier score, log loss, ROC AUC, and accuracy

### The convex triplet ensemble search workflow:

* loads saved out-of-fold positive-class probabilities from the configured XGBoost, LightGBM, and Logistic Regression model families
* builds one unified out-of-fold probability matrix
* searches all 3-model combinations exactly
* finds the Brier-optimal convex weights for each triplet, with nonnegative weights constrained to sum to 1
* ranks triplets by OOF Brier score
* retains up to the configured number of top OOF-ranked triplets
* saves the retained triplet metadata and out-of-fold probabilities

### The greedy ensemble selection workflow:

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

### The LightGBM ensemble-aware HPO budget-selection workflow:

* is currently implemented for the original LightGBM feature set and the Caruana-style greedy ensemble
* uses only the development set when selecting the candidate budget
* partitions the development set into stratified meta-folds
* runs one independent seeded LightGBM Optuna trajectory for each meta-fold
* keeps the corresponding meta-holdout fold completely outside that Optuna trajectory
* runs every Optuna trajectory sequentially in checkpoint blocks while preserving the same Optuna study and TPE history across checkpoints
* performs the usual inner stratified K-fold out-of-fold evaluation for every LightGBM trial using only the meta-training portion
* builds a LightGBM-only greedy ensemble from the candidates accumulated in that meta-training trajectory at each checkpoint, subject to the configured greedy pool size
* retrains the selected greedy ensemble members on the complete meta-training portion
* evaluates the retrained ensemble on the untouched meta-holdout portion
* combines all meta-holdout probability vectors into one complete cross-fitted probability vector over the development set
* computes the checkpoint Brier score from that complete cross-fitted probability vector
* treats a strictly lower cross-fitted Brier score as an improvement
* selects the best checkpoint encountered
* can terminate candidate generation after the configured number of consecutive non-improving checkpoints
* never uses the outer test set to select the candidate budget

The independent meta-fold Optuna trajectories may run in parallel processes, while every individual Optuna study and every LightGBM model fit remains single-threaded.

Stopping artifacts are saved locally in:

`stopping_results/<dataset>/split_seed_<DATA_SPLIT_SEED>/lightgbm-original/greedy_ensemble_stopping/`

The saved artifacts include the stopping settings, complete checkpoint history, cross-fitted checkpoint probabilities, final stopping summary, and per-meta-fold Optuna and checkpoint artifacts.

The selected-budget comparison workflow then uses the already stored sequential full-development LightGBM Optuna trajectory. The first `N` saved trials represent the candidate zoo that existed after `N` sequential trials, so they can be used to reconstruct the candidate zoo corresponding to the selected budget.

`run_lightgbm_stopped_greedy_selection.py` builds:

* a LightGBM-only greedy ensemble from the first `N` stored full-development Optuna trials, where `N` is the selected budget
* a LightGBM-only greedy ensemble from the complete stored full-development Optuna trajectory

These comparison ensembles are saved in:

`ensemble_results/<dataset>/split_seed_<DATA_SPLIT_SEED>/lightgbm-original/stopping_comparison/`

The outer test set remains untouched until `evaluate_lightgbm_stopping_comparison.py` retrains and evaluates the selected-budget and full-budget ensembles. The comparison script saves test metadata, test probability vectors, and a compact `comparison_summary.pkl`.

With the current configuration of 100-trial checkpoints, a 1000-trial maximum, and `LIGHTGBM_STOPPING_PATIENCE = 10`, the workflow intentionally reaches the 1000-trial maximum and selects the checkpoint with the lowest cross-fitted ensemble Brier score among 100, 200, ..., 1000 trials. This configuration can be used to study candidate-budget selection over the complete checkpoint curve rather than to realize computational savings through early termination.

### The triplet ensemble test evaluation workflow:

* loads the saved OOF-ranked triplet metadata
* retrains the selected base models on the full development set using their saved trial hyperparameters
* for selected Logistic Regression models, fits a `StandardScaler` on the full development feature matrix and applies it to both the development and test feature matrices
* combines the selected models' positive-class test probabilities using the saved convex weights
* evaluates the retained triplet ensembles on the held-out test set
* saves triplet test metadata and test probabilities
* prints the OOF Brier score and final test Brier score, log loss, ROC AUC, and accuracy for the best OOF-ranked triplet and, for diagnostic purposes only, for the retained triplet with the lowest test Brier score

### The greedy ensemble test evaluation workflow:

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

Evaluate the saved greedy ensemble on the held-out test set:

`python evaluate_greedy_ensemble.py`

The greedy ensemble evaluation script retrains the selected base models on the full development set, combines their positive-class test probabilities using the saved convex weights, evaluates the ensemble on the held-out test set, and saves the resulting test artifacts within `ensemble_results/`.

The LightGBM stopping experiment is an additional LightGBM-only greedy ensemble workflow. It requires prepared data for the selected development/test split.

First, run the independent cross-fitted LightGBM ensemble-aware HPO budget-selection procedure:

`python run_lightgbm_ensemble_stopping.py`

The script creates independent checkpointed Optuna trajectories within the configured development meta-folds, evaluates the downstream greedy ensemble on untouched meta-holdout observations at every checkpoint, reconstructs one complete cross-fitted development-set probability vector per checkpoint, and selects the candidate budget with the best cross-fitted Brier score encountered.

The corresponding outer test set is not used by this script.

To construct the selected-budget and full-budget final LightGBM greedy ensembles, a complete sequential full-development LightGBM Optuna trajectory must already exist in `optuna_results/`. If it has not already been generated, execute `run_lightgbm_optuna.py` first. Then run:

`python run_lightgbm_stopped_greedy_selection.py`

This script uses the independently selected candidate budget `N`, takes the first `N` trials from the stored full-development LightGBM Optuna trajectory, builds the corresponding greedy ensemble, and also builds a baseline greedy ensemble from the complete stored trajectory.

Finally, evaluate both ensembles on the untouched outer test set:

`python evaluate_lightgbm_stopping_comparison.py`

The comparison script retrains the selected base models on the full development set, evaluates the selected-budget and full-budget greedy ensembles on the held-out test set, prints their out-of-fold and test metrics, prints the selected-minus-full test Brier difference, and saves the comparison artifacts within `ensemble_results/`.

A negative selected-minus-full test Brier difference means that the independently selected candidate budget produced a lower test Brier score than the full stored candidate zoo.

## Configuration

Main user-facing settings are stored in `config.py`.

Important general and data preparation settings include:

`DATASET_NAME` — selected dataset name\
`TEST_SIZE` — test set fraction\
`DATA_SPLIT_SEED` — random seed used only for the development/test split; it also determines the split-specific output subfolder name\
`RANDOM_SEED` — random seed used by Optuna sampling, stratified K-fold and meta-fold splitting, model randomness, greedy ensemble selection tie-breaking, and final retraining\
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

Important LightGBM ensemble-aware HPO budget-selection settings are:

`LIGHTGBM_STOPPING_META_N_SPLITS` — number of stratified development meta-folds used for independent downstream ensemble validation\
`LIGHTGBM_STOPPING_CHECKPOINT_TRIALS` — number of additional Optuna trials run per meta-fold between checkpoint evaluations\
`LIGHTGBM_STOPPING_PATIENCE` — number of consecutive non-improving checkpoints allowed before candidate generation terminates\
`LIGHTGBM_STOPPING_MAX_TRIALS` — maximum number of Optuna trials in each meta-fold trajectory\
`LIGHTGBM_STOPPING_MAX_WORKERS` — maximum number of independent meta-fold Optuna trajectories executed concurrently

Each individual Optuna trajectory in the stopping workflow remains sequential, and every LightGBM model uses a single job. Parallelism is applied only across separate meta-fold trajectories.

With the current settings—`LIGHTGBM_STOPPING_CHECKPOINT_TRIALS = 100`, `LIGHTGBM_STOPPING_PATIENCE = 10`, and `LIGHTGBM_STOPPING_MAX_TRIALS = 1000`—the experiment runs the complete 100-to-1000 checkpoint curve and selects the checkpoint with the lowest cross-fitted Brier score rather than terminating early through patience.

`OPTUNA_RESULTS_DIR` represents the root output folder for Optuna result artifacts.

`ENSEMBLE_RESULTS_DIR` represents the root output folder for triplet search, greedy selection, and ensemble evaluation artifacts.

`STOPPING_RESULTS_DIR` represents the root output folder for ensemble-aware HPO stopping artifacts.

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

The full OOF Brier score is minimized by Optuna. Per-fold Brier scores are also retained for each trial.

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

## LightGBM Ensemble-Aware HPO Budget Selection

The ensemble-aware HPO budget-selection workflow addresses candidate library size as a downstream ensemble selection problem rather than stopping Optuna according to the best individual model objective alone.

For each development/test split, the test set is held completely aside. The development set is partitioned into stratified meta-folds. For every meta-fold, LightGBM hyperparameter optimization is performed only on the corresponding meta-training portion, with inner stratified K-fold out-of-fold predictions generated for every trial.

At each configured checkpoint, all LightGBM candidates accumulated in the meta-training trajectory are passed to the same Caruana-style greedy ensemble selector used elsewhere in the project. The selected ensemble members are then retrained on the complete meta-training portion and evaluated on the untouched meta-holdout portion.

Each development observation therefore receives one ensemble probability from a model selection process that did not use that observation for either Optuna tuning or greedy ensemble selection. The resulting held-out probabilities are concatenated into one full cross-fitted development-set prediction vector, and its Brier score is used as the checkpoint-level downstream ensemble generalization estimate.

The selected candidate budget is the checkpoint with the lowest cross-fitted Brier score encountered. A strict decrease is required for improvement. The implementation also supports patience-based termination after a configured number of consecutive non-improving checkpoints.

The outer test set is not used to choose the candidate budget.

For final comparison, the selected budget is applied to the stored sequential full-development LightGBM Optuna trajectory by taking its first `N` trials. A greedy ensemble is built from those selected candidates and compared with a greedy ensemble built from the complete stored candidate zoo. Both are retrained on the full development set and evaluated on the same untouched outer test set.

This workflow currently supports LightGBM with the `original` feature set and greedy ensemble selection. It is implemented separately from the existing multi-model greedy and exact convex triplet workflows.

## Reproducibility

The project limits hidden parallelism in numerical libraries through `environment.py`.

For stricter reproducibility, `PYTHONHASHSEED` can be set before launching Python.

The development/test split is controlled by `DATA_SPLIT_SEED`. Changing `DATA_SPLIT_SEED` creates a different held-out split and writes generated artifacts under `split_seed_<DATA_SPLIT_SEED>` subfolders, allowing multiple split experiments for each dataset without overwriting previous results.

The split uses stratification so that the class proportions in the development and test sets remain similar to those in the full dataset.

`RANDOM_SEED` is defined separately so that Optuna sampling, stratified K-fold splitting, and model randomness can remain fixed while the development/test split is varied independently.

XGBoost, LightGBM, and Logistic Regression tuning use seeded Optuna samplers, seeded stratified K-fold splitting, and seeded models. Final retraining for all three model families also uses `RANDOM_SEED`. Greedy ensemble selection uses `RANDOM_SEED` for deterministic tie-breaking, and triplet and greedy ensemble test evaluation use the same seed when retraining the selected base models. `XGBOOST_N_JOBS = 1`, `LIGHTGBM_N_JOBS = 1`, and `LOGISTIC_REGRESSION_N_JOBS = 1` keep Optuna trial execution sequential by default for improved reproducibility.

The native triplet search uses OpenMP for parallel exhaustive search. Its thread count is controlled by `TRIPLET_N_THREADS`.

The LightGBM ensemble-aware HPO budget-selection workflow uses seeded stratified meta-fold splitting and preserves each meta-fold Optuna study across checkpoint blocks. The independent meta-fold trajectories may execute concurrently in separate processes, but every individual Optuna study is sequential and every LightGBM model remains single-threaded. Resuming a saved meta-fold study preserves its accumulated Optuna history rather than restarting the search.

Changing `DATA_SPLIT_SEED` changes the development/test split while `RANDOM_SEED` continues to control the meta-fold split, Optuna sampling, inner cross-validation, LightGBM randomness, and greedy tie-breaking within that split.

## Generated Files

The pipeline may generate files such as:

* `.npy` prepared arrays
* `.pkl` fitted preprocessing objects
* `.pkl` Optuna studies and result artifacts
* `.pkl` ensemble search, selection, evaluation, stopping, and comparison artifacts
* compiled native extension files
* the `prepared_data/` directory
* the `optuna_results/` directory
* the `ensemble_results/` directory
* the `stopping_results/` directory
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
* checkpointed continuation of sequential LightGBM Optuna trajectories
* cross-fitted downstream ensemble evaluation for HPO candidate budget selection
* strict separation of candidate budget selection from the test set
* reconstruction and comparison of selected-budget and full-budget LightGBM greedy ensembles
* clean separation of dataset loading, preprocessing, tuning, ensemble search, selection, evaluation, and storage
* a scalable structure for future model comparison and ensemble optimization

## License

Copyright 2026 Maksym Eingorn

Licensed under the Apache License, Version 2.0. See the `LICENSE` file for details.