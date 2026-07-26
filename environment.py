# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# environment.py

import os


def configure_environment() -> None:
    """
    Limit hidden parallelism in numerical libraries to improve reproducibility.

    This should be called before importing NumPy, scikit-learn, XGBoost,
    LightGBM, or other numerical libraries.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["XGB_NUM_THREADS"] = "1"
    os.environ["LGBM_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
