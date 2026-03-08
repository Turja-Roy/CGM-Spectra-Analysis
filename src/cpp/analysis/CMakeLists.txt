add_library(analysis STATIC
    power_spectrum.cpp
    column_density.cpp
    line_width.cpp
    flux_stats.cpp
    temperature_density.cpp
)

target_include_directories(analysis PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(analysis PUBLIC
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

if(FFTW3F_FOUND)
    target_compile_definitions(analysis PUBLIC USE_FFTW)
    target_link_libraries(analysis PUBLIC FFTW3F::FFTW3F)
endif()

target_compile_options(analysis PRIVATE
    -O3
    -Wall
    -Wextra
    -fPIC
)
