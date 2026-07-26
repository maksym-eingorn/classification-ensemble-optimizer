# Classification Ensemble Optimizer

A modular Python machine learning project for building a binary classification ensemble optimization pipeline.

## Overview

The project currently implements data preparation for the Breast Cancer Wisconsin Diagnostic dataset.

The data preparation pipeline includes dataset loading, binary target encoding, stratified development/test splitting, feature scaling, and saving prepared arrays and the fitted scaler.

The Breast Cancer Wisconsin target is encoded so that benign samples are class `0` and malignant samples are class `1`. This makes malignant the positive class for later probability-based classification metrics and ensemble evaluation.

## Project Structure

`main.py` — data preparation pipeline\
`config.py` — project settings and user-configurable parameters\
`environment.py` — numerical library thread settings for improved reproducibility\
`datasets/loader.py` — dataset dispatcher\
`datasets/breast_cancer.py` — Breast Cancer Wisconsin Diagnostic dataset loading and target encoding\
`preprocessing.py` — stratified development/test splitting and standard feature scaling\
`storage.py` — saving and loading prepared NumPy arrays and fitted preprocessing objects\
`requirements.txt` — Python package dependencies

## How It Works

The data preparation pipeline:

* loads the selected binary classification dataset
* converts the Breast Cancer Wisconsin target to `0 = benign` and `1 = malignant`
* splits the data into development and test sets
* preserves class proportions through stratified splitting
* scales the original features
* fits the scaler only on the development set
* applies the fitted scaler to both development and test feature matrices
* saves prepared arrays as `.npy` files
* saves the fitted scaler as a `.pkl` file

Generated data artifacts are saved locally in dataset- and split-specific subfolders:

`prepared_data/breast_cancer/split_seed_<DATA_SPLIT_SEED>/`

## How to Run

This project is configured for Python 3.14.

Install dependencies from the project directory with:

`pip install -r requirements.txt`

Then run:

`python main.py`

The script prepares the selected dataset and saves processed outputs into the corresponding dataset- and split-specific subfolder within `prepared_data/`.

## Configuration

Main user-facing settings are stored in `config.py`.

Important data preparation settings include:

`DATASET_NAME` — selected dataset name\
`TEST_SIZE` — test set fraction\
`DATA_SPLIT_SEED` — random seed used only for the development/test split; it also determines the split-specific output subfolder name\
`RANDOM_SEED` — random seed reserved for later model tuning, K-fold cross-validation, model training, and other non-split randomness\
`PREPARED_DATA_DIR` — root output folder for prepared data

The currently supported dataset is:

`breast_cancer`

## Breast Cancer Wisconsin Diagnostic Dataset

The project uses the Breast Cancer Wisconsin Diagnostic dataset provided by scikit-learn.

The dataset contains 569 samples and 30 numerical input features.

Scikit-learn originally encodes malignant samples as class `0` and benign samples as class `1`. During loading, the project reverses this encoding so that:

`0 = benign`  
`1 = malignant`

The project therefore treats malignant as the positive class.

The full feature matrix and target vector are preserved, together with the development/test split and scaled feature matrices.

## Reproducibility

The project limits hidden parallelism in numerical libraries through `environment.py`.

For stricter reproducibility, `PYTHONHASHSEED` can be set before launching Python.

The development/test split is controlled by `DATA_SPLIT_SEED`. Changing `DATA_SPLIT_SEED` creates a different held-out split and writes generated artifacts under `split_seed_<DATA_SPLIT_SEED>` subfolders, allowing multiple split experiments for the same dataset without overwriting previous results.

The split uses stratification so that the class proportions in the development and test sets remain similar to those in the full dataset.

`RANDOM_SEED` is defined separately so that later model and tuning randomness can remain fixed while the development/test split is varied independently.

## Generated Files

The pipeline may generate files such as:

* `.npy` prepared arrays
* `.pkl` fitted preprocessing objects
* the `prepared_data/` directory

These files are ignored by Git because they are generated artifacts rather than source code.

## Why This Project

This project is designed as a clean, extensible foundation for binary classification ensemble experimentation.

It currently emphasizes:

* modular Python architecture
* reproducible data preparation
* stratified development/test splitting
* leakage-aware preprocessing
* explicit positive class encoding
* clean separation of dataset loading, preprocessing, and storage
* a scalable structure for future model tuning, comparison, and ensemble optimization

## License

Copyright 2026 Maksym Eingorn

Licensed under the Apache License, Version 2.0. See the `LICENSE` file for details.