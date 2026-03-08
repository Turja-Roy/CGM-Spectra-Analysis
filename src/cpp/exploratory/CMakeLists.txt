add_library(exploratory STATIC
    spectral_features.cpp
)

target_include_directories(exploratory PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(exploratory PUBLIC
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

target_compile_options(exploratory PRIVATE -O3 -Wall -fPIC)
