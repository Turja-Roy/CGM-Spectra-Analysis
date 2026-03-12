cmake_minimum_required(VERSION 3.15)
project(cgm_analysis VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use local Eigen3 (header-only, downloaded to third_party/)
set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../../third_party/eigen-3.4.0")
message(STATUS "Using local Eigen3: ${EIGEN3_INCLUDE_DIR}")

# Find pybind11 from venv (before system)
find_package(pybind11 REQUIRED HINTS "${CMAKE_SOURCE_DIR}/../../.venv/lib/python3.14/site-packages/pybind11")
find_package(OpenMP REQUIRED)

# Set include directories for Eigen3
set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

# Create imported target for Eigen3
add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
)

option(USE_FFTW "Use FFTW for FFT (faster)" ON)

# Find FFTW3f (single precision)
find_package(FFTW3f QUIET)
if(NOT FFTW3f_FOUND)
    find_library(FFTW3F_LIBRARY NAMES fftw3f PATHS /usr/lib NO_DEFAULT_PATH)
    if(FFTW3F_LIBRARY)
        set(FFTW3f_FOUND TRUE)
        set(FFTW3f_LIBRARIES ${FFTW3F_LIBRARY})
    endif()
endif()

message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Python executable: ${PYTHON_EXECUTABLE}")
message(STATUS "pybind11 version: ${pybind11_VERSION}")
message(STATUS "Eigen version: 3.4.0 (local)")
message(STATUS "OpenMP: ${OpenMP_CXX_VERSION}")

add_subdirectory(analysis)
add_subdirectory(exploratory)
add_subdirectory(cgm)
add_subdirectory(bindings)

message(STATUS "Build configuration: ${CMAKE_BUILD_TYPE}")

install(TARGETS _analysis_cpp _exploratory_cpp _cgm_cpp
        LIBRARY DESTINATION .
        ARCHIVE DESTINATION .)
