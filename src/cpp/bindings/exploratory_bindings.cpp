#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "exploratory/spectral_features.h"

namespace py = pybind11;

PYBIND11_MODULE(_exploratory_cpp, m) {
    m.doc() = "C++ implementation of CGM exploratory analysis functions";
    
    m.def("extract_spectral_features", 
          [](const Eigen::Ref<const Eigen::ArrayXXf>& tau,
             float velocity_spacing,
             float void_threshold,
             float line_threshold,
             float absorber_threshold,
             int max_sightlines,
             int max_separations,
             float max_separation) {
        auto result = cgm::exploratory::extract_spectral_features(
            tau, velocity_spacing, void_threshold, line_threshold,
            absorber_threshold, max_sightlines, max_separations, max_separation);
        
        py::dict d;
        d["void_sizes"] = result.void_sizes;
        d["line_widths"] = result.line_widths;
        d["absorber_separations"] = result.absorber_separations;
        d["mean_void_size"] = result.mean_void_size;
        d["median_void_size"] = result.median_void_size;
        d["mean_line_width"] = result.mean_line_width;
        d["median_line_width"] = result.median_line_width;
        d["saturation_fraction"] = result.saturation_fraction;
        d["deep_absorption_fraction"] = result.deep_absorption_fraction;
        d["transmission_fraction"] = result.transmission_fraction;
        d["flux_mean"] = result.flux_mean;
        d["flux_variance"] = result.flux_variance;
        d["flux_skewness"] = result.flux_skewness;
        d["flux_kurtosis"] = result.flux_kurtosis;
        d["mean_absorber_separation"] = result.mean_absorber_separation;
        d["n_voids"] = result.n_voids;
        d["n_lines"] = result.n_lines;
        d["n_absorbers"] = result.n_absorbers;
        return d;
    },
          "Extract spectral features from optical depth array",
          py::arg("tau"),
          py::arg("velocity_spacing") = 0.1f,
          py::arg("void_threshold") = 0.9f,
          py::arg("line_threshold") = 0.5f,
          py::arg("absorber_threshold") = 0.5f,
          py::arg("max_sightlines") = 100,
          py::arg("max_separations") = 1000,
          py::arg("max_separation") = 500.0f);
}
