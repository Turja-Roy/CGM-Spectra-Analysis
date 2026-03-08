#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "cgm/halos.h"

namespace py = pybind11;

PYBIND11_MODULE(_cgm_cpp, m) {
    m.doc() = "C++ implementation of CGM halo analysis functions";
    
    m.def("filter_isolated_halos",
          [](const Eigen::Ref<const Eigen::ArrayXXf>& positions,
             const Eigen::Ref<const Eigen::VectorXf>& masses,
             const Eigen::Ref<const Eigen::VectorXf>& radii,
             float isolation_factor,
             float box_size) {
        auto result = cgm::cgm::filter_isolated_halos(positions, masses, radii, isolation_factor, box_size);
        py::dict d;
        d["isolated_mask"] = result.isolated_mask;
        d["n_isolated"] = result.n_isolated;
        d["n_non_isolated"] = result.n_non_isolated;
        return d;
    },
          "Filter isolated halos based on proximity to other halos",
          py::arg("positions"),
          py::arg("masses"),
          py::arg("radii"),
          py::arg("isolation_factor") = 3.0f,
          py::arg("box_size") = 0.0f);
    
    m.def("compute_impact_parameters", &cgm::cgm::compute_impact_parameters,
          "Compute impact parameters between sightlines and halos",
          py::arg("sightline_origins"),
          py::arg("sightline_dirs"),
          py::arg("halo_positions"),
          py::arg("halo_radii"));
}
