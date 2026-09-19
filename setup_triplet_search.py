# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# setup_triplet_search.py

import os
import sys
from pathlib import Path

import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext


def _first_existing_path(candidates: list[Path]) -> Path | None:
    """Return the first existing path from a list of candidates."""
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def _get_eigen_path() -> Path:
    """Return the Eigen include directory."""
    configured = os.environ.get("EIGEN_INCLUDE_DIR")

    candidates = []
    if configured:
        candidates.append(Path(configured))

    if sys.platform == "darwin":
        candidates.extend([
            Path("/opt/homebrew/include/eigen3"),
            Path("/usr/local/include/eigen3")
        ])
    elif sys.platform == "win32":
        candidates.append(Path(r"C:\Libraries\eigen-3.4.1"))
    else:
        candidates.extend([
            Path("/usr/include/eigen3"),
            Path("/usr/local/include/eigen3")
        ])

    found_path = _first_existing_path(candidates)

    if found_path is None:
        raise RuntimeError(
            "Eigen include directory was not found. "
            "Install Eigen or set EIGEN_INCLUDE_DIR to its include directory."
        )

    return found_path


def _get_macos_libomp_prefix() -> Path:
    """Return the Homebrew libomp prefix on macOS."""
    configured = os.environ.get("LIBOMP_PREFIX")

    candidates = []
    if configured:
        candidates.append(Path(configured))

    candidates.extend([
        Path("/opt/homebrew/opt/libomp"),
        Path("/usr/local/opt/libomp")
    ])

    found_prefix = _first_existing_path(candidates)

    if found_prefix is None:
        raise RuntimeError(
            "libomp was not found. Install it with 'brew install libomp' "
            "or set LIBOMP_PREFIX to the libomp prefix."
        )

    return found_prefix


eigen_path = _get_eigen_path()

define_macros = []
include_dirs = [str(eigen_path)]
library_dirs = []

if sys.platform == "win32":
    define_macros = [("NOMINMAX", "1")]
    extra_compile_args = ["/O2", "/openmp", "/MD", "/EHsc"]
    extra_link_args = []
elif sys.platform == "darwin":
    libomp_prefix = _get_macos_libomp_prefix()
    libomp_include = libomp_prefix / "include"
    libomp_lib = libomp_prefix / "lib"

    include_dirs.append(str(libomp_include))
    library_dirs.append(str(libomp_lib))

    extra_compile_args = ["-O3", "-Xpreprocessor", "-fopenmp"]
    extra_link_args = [
        "-lomp",
        f"-Wl,-rpath,{libomp_lib}"
    ]
else:
    extra_compile_args = ["-O3", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

ext_modules = [
    Pybind11Extension(
        "ensemble._triplet_search",
        ["native/triplet_search.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        cxx_std=17
    )
]

setuptools.setup(
    name="classification_ensemble_optimizer_triplet_search",
    version="0.1",
    packages=["ensemble"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext}
)
