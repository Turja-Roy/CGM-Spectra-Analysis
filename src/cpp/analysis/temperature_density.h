#ifndef CGM_ANALYSIS_TEMPERATURE_DENSITY_H
#define CGM_ANALYSIS_TEMPERATURE_DENSITY_H

#include <Eigen/Dense>

namespace cgm {
namespace analysis {

struct TemperatureDensityResult {
    Eigen::VectorXd temperature;
    Eigen::VectorXd density;
    Eigen::VectorXd log_T;
    Eigen::VectorXd log_rho;
    double T0;
    double gamma;
    double gamma_err;
    double rho_mean;
    int n_pixels;
};

TemperatureDensityResult compute_temperature_density_relation(
    const Eigen::Ref<const Eigen::ArrayXXf>& temperature,
    const Eigen::Ref<const Eigen::ArrayXXf>& density,
    const Eigen::Ref<const Eigen::ArrayXXf>& tau,
    float min_tau = 0.1f);

}
}

#endif
