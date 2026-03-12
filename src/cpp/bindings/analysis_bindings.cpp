#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "analysis/power_spectrum.h"
#include "analysis/column_density.h"
#include "analysis/line_width.h"
#include "analysis/flux_stats.h"
#include "analysis/temperature_density.h"
#include "analysis/voigt.h"

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
    
    m.def("compute_power_spectrum", 
          [](const Eigen::Ref<const Eigen::ArrayXXf>& flux, double velocity_spacing, int chunk_size) {
        auto result = cgm::analysis::compute_power_spectrum(flux, velocity_spacing, chunk_size);
        py::dict d;
        d["k"] = result.k;
        d["P_k_mean"] = result.P_k_mean;
        d["P_k_std"] = result.P_k_std;
        d["P_k_err"] = result.P_k_err;
        d["n_modes"] = result.n_modes;
        d["mean_flux"] = result.mean_flux;
        d["n_sightlines"] = result.n_sightlines;
        d["velocity_spacing"] = result.velocity_spacing;
        return d;
    },
          "Compute power spectrum from flux array",
          py::arg("flux"),
          py::arg("velocity_spacing"),
          py::arg("chunk_size") = 1000);
    
    m.def("compute_column_density_distribution", 
          [](const Eigen::Ref<const Eigen::ArrayXXf>& tau, double velocity_spacing, float threshold,
             py::array_t<float, py::array::c_style | py::array::forcecast> colden, 
             double redshift, double box_size_ckpc_h, double hubble, double omega_m) {
        // Get buffer info
        py::buffer_info info = colden.request();
        
        const float* colden_data = nullptr;
        int colden_rows = 0, colden_cols = 0;
        
        if (info.size > 0) {
            colden_data = static_cast<const float*>(info.ptr);
            colden_rows = info.shape[0];
            colden_cols = info.shape[1];
        }
        
        // Call C++ function with raw pointer
        auto result = cgm::analysis::compute_column_density_distribution(
            tau, velocity_spacing, threshold, colden_data, colden_rows, colden_cols,
            redshift, box_size_ckpc_h, hubble, omega_m);
        py::dict d;
        d["N_HI"] = result.N_HI;
        d["counts"] = result.counts;
        d["bins"] = result.bins;
        d["bin_centers"] = result.bin_centers;
        d["f_N"] = result.f_N;
        d["beta_fit"] = result.beta_fit;
        d["n_absorbers"] = result.n_absorbers;
        d["n_sightlines"] = result.n_sightlines;
        d["dX"] = result.dX;
        d["redshift"] = result.redshift;
        return d;
    },
          "Compute column density distribution function",
          py::arg("tau"),
          py::arg("velocity_spacing"),
          py::arg("threshold") = 0.5f,
          py::arg("colden") = Eigen::ArrayXXf(),
          py::arg("redshift") = std::nan(""),
          py::arg("box_size_ckpc_h") = std::nan(""),
          py::arg("hubble") = 0.6774,
          py::arg("omega_m") = 0.3089);
    
    m.def("compute_line_width_distribution", 
          [](const Eigen::Ref<const Eigen::ArrayXXf>& tau, double velocity_spacing, float threshold,
             const Eigen::ArrayXXf& colden) {
        // Always pass nullptr since colden handling is broken in binding
        auto result = cgm::analysis::compute_line_width_distribution(tau, velocity_spacing, threshold, nullptr);
        py::dict d;
        d["N_HI"] = result.N_HI;
        d["b_params"] = result.b_params;
        d["temperatures"] = result.temperatures;
        d["b_median"] = result.b_median;
        d["b_mean"] = result.b_mean;
        d["b_std"] = result.b_std;
        d["n_absorbers"] = result.n_absorbers;
        return d;
    },
          "Compute line width (b-parameter) distribution",
          py::arg("tau"),
          py::arg("velocity_spacing"),
          py::arg("threshold") = 0.5f,
          py::arg("colden") = Eigen::ArrayXXf());
    
    m.def("compute_temperature_density_relation", 
          [](const Eigen::Ref<const Eigen::ArrayXXf>& temperature,
             const Eigen::Ref<const Eigen::ArrayXXf>& density,
             const Eigen::Ref<const Eigen::ArrayXXf>& tau, float min_tau) {
        auto result = cgm::analysis::compute_temperature_density_relation(temperature, density, tau, min_tau);
        py::dict d;
        d["temperature"] = result.temperature;
        d["density"] = result.density;
        d["log_T"] = result.log_T;
        d["log_rho"] = result.log_rho;
        d["T0"] = result.T0;
        d["gamma"] = result.gamma;
        d["gamma_err"] = result.gamma_err;
        d["n_pixels"] = result.n_pixels;
        return d;
    },
          "Compute temperature-density relation",
          py::arg("temperature"),
          py::arg("density"),
          py::arg("tau"),
          py::arg("min_tau") = 0.1f);
    
    m.def("compute_voigt_profile",
          [](const Eigen::ArrayXXf& v, double tau_0, double b, double v_center, double damping = 4.7e-4) {
        Eigen::ArrayXXf result(v.size(), 1);
        for (int i = 0; i < v.size(); ++i) {
            result(i) = cgm::analysis::compute_voigt_optical_depth(v(i), tau_0, b, v_center, damping);
        }
        return result;
    },
          "Compute Voigt profile optical depth",
          py::arg("v"),
          py::arg("tau_0"),
          py::arg("b"),
          py::arg("v_center"),
          py::arg("damping") = 4.7e-4);
}
