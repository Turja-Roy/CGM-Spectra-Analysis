# Analysis module
pybind11_add_module(_analysis_cpp
    analysis_bindings.cpp
)

target_include_directories(_analysis_cpp PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/analysis
)

target_link_libraries(_analysis_cpp PRIVATE
    analysis
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

# Exploratory module
pybind11_add_module(_exploratory_cpp
    exploratory_bindings.cpp
)

target_include_directories(_exploratory_cpp PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/exploratory
)

target_link_libraries(_exploratory_cpp PRIVATE
    exploratory
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

# CGM module
pybind11_add_module(_cgm_cpp
    cgm_bindings.cpp
)

target_include_directories(_cgm_cpp PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/cgm
)

target_link_libraries(_cgm_cpp PRIVATE
    cgm
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)
