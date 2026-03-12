add_library(analysis STATIC
    power_spectrum.cpp
    column_density.cpp
    line_width.cpp
    flux_stats.cpp
    temperature_density.cpp
    voigt.cpp
    Faddeeva.cc
)

target_include_directories(analysis PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/../../third_party
)

target_link_libraries(analysis PUBLIC
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

if(FFTW3f_FOUND)
    target_compile_definitions(analysis PUBLIC USE_FFTW)
    target_link_libraries(analysis PUBLIC ${FFTW3f_LIBRARIES})
endif()

target_compile_options(analysis PRIVATE
    -O3
    -Wall
    -Wextra
    -fPIC
)
