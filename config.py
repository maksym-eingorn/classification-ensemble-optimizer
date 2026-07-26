# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# config.py

# --- General settings ---

RANDOM_SEED = 42

# Controls only the development/test split.
DATA_SPLIT_SEED = 21

# --- Dataset settings ---

# Currently supported: "breast_cancer"
DATASET_NAME = "breast_cancer"

TEST_SIZE = 0.2

# --- Output paths ---

PREPARED_DATA_DIR = "prepared_data"
