#!/bin/bash

# Load TACC modules (if not loaded already)
module load cmake gcc/12.2.0 eigen/3.4.0 fftw/3.3.10

mkdir -p src/cpp/build
cd src/cpp/build

cmake .. \
    -DUSE_FFTW=ON \
    -DCMAKE_BUILD_TYPE=Release \

make -j$(nproc)

make install
