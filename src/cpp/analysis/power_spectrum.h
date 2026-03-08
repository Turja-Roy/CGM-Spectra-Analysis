#ifndef CGM_ANALYSIS_POWER_SPECTRUM_H
#define CGM_ANALYSIS_POWER_SPECTRUM_H

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace cgm {
namespace analysis {

struct PowerSpectrumResult {
    Eigen::VectorXd k;
    Eigen::VectorXd P_k_mean;
    Eigen::VectorXd P_k_std;
    Eigen::VectorXd P_k_err;
    Eigen::VectorXd n_modes;
    double mean_flux;
    int n_sightlines;
    double velocity_spacing;
};

PowerSpectrumResult compute_power_spectrum(
    const Eigen::Ref<const Eigen::ArrayXXf>& flux,
    double velocity_spacing,
    int chunk_size = 1000);

}
}

#endif
