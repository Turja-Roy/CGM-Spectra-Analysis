#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

# Load HPC modules (gcc + MPI + cmake + eigen + fftw)
module load cmake gcc/13.2.0 impi/19.0.9 eigen/3.4.0 fftw3/3.3.10

BUILD_DIR="$SCRIPT_DIR/src/cpp/build"
# Clean old build (to avoid stale CMake cache)

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_FFTW=ON \
    -DPYTHON_EXECUTABLE="$VENV_PYTHON" \
    -DCMAKE_INSTALL_PREFIX="$SCRIPT_DIR/.venv/lib/python3.12/site-packages" \
    -Dpybind11_DIR=$("$VENV_PYTHON" -c "import pybind11; print(pybind11.get_cmake_dir())")

make -j$(nproc)
make install
