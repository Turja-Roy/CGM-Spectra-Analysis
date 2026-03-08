add_library(cgm STATIC
    halos.cpp
)

target_include_directories(cgm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(cgm PUBLIC
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

target_compile_options(cgm PRIVATE -O3 -Wall -fPIC)
