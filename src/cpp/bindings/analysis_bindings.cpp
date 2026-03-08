#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "analysis/power_spectrum.h"
#include "analysis/column_density.h"
#include "analysis/line_width.h"
#include "analysis/flux_stats.h"
#include "analysis/temperature_density.h"

namespace py = pybind11;

PYBIND11_MODULE(_analysis_cpp, m) {
    m.doc() = "C++ implementation of CGM spectroscopy analysis functions";
    
    m.def("compute_flux_statistics", [](const Eigen::ArrayXXf& tau) {
        auto result = cgm::analysis::compute_flux_statistics(tau);
        py::dict d;
        d["mean_flux"] = result.mean_flux;
        d["median_flux"] = result.median_flux;
        d["std_flux"] = result.std_flux;
        d["min_flux"] = result.min_flux;
        d["max_flux"] = result.max_flux;
        d["mean_tau"] = result.mean_tau;
        d["median_tau"] = result.median_tau;
        d["effective_tau"] = result.effective_tau;
        d["deep_absorption_frac"] = result.deep_absorption_frac;
        d["moderate_absorption_frac"] = result.moderate_absorption_frac;
        d["weak_absorption_frac"] = result.weak_absorption_frac;
        return d;
    }, "Compute basic flux statistics from optical depth",
          py::arg("tau"));
    
    m.def("compute_power_spectrum", &cgm::analysis::compute_power_spectrum,
          "Compute power spectrum from flux array",
          py::arg("flux"),
          py::arg("velocity_spacing"),
          py::arg("chunk_size") = 1000);
    
    m.def("compute_column_density_distribution", &cgm::analysis::compute_column_density_distribution,
          "Compute column density distribution function",
          py::arg("tau"),
          py::arg("velocity_spacing"),
          py::arg("threshold") = 0.5f,
          py::arg("colden") = nullptr,
          py::arg("redshift") = std::nan(""),
          py::arg("box_size_ckpc_h") = std::nan(""),
          py::arg("hubble") = 0.6774,
          py::arg("omega_m") = 0.3089);
    
    m.def("compute_line_width_distribution", &cgm::analysis::compute_line_width_distribution,
          "Compute line width (b-parameter) distribution",
          py::arg("tau"),
          py::arg("velocity_spacing"),
          py::arg("threshold") = 0.5f,
          py::arg("colden") = nullptr);
    
    m.def("compute_temperature_density_relation", &cgm::analysis::compute_temperature_density_relation,
          "Compute temperature-density relation",
          py::arg("temperature"),
          py::arg("density"),
          py::arg("tau"),
          py::arg("min_tau") = 0.1f);
}
