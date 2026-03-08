#include "temperature_density.h"
#include "constants.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

namespace cgm {
namespace analysis {

TemperatureDensityResult compute_temperature_density_relation(
    const Eigen::Ref<const Eigen::ArrayXXf>& temperature,
    const Eigen::Ref<const Eigen::ArrayXXf>& density,
    const Eigen::Ref<const Eigen::ArrayXXf>& tau,
    float min_tau) {
    
    const float* temp_data = temperature.data();
    const float* dens_data = density.data();
    const float* tau_data = tau.data();
    const int total_size = temperature.size();
    
    std::vector<float> temp_filtered;
    std::vector<float> dens_filtered;
    temp_filtered.reserve(total_size);
    dens_filtered.reserve(total_size);
    
    for (int i = 0; i < total_size; ++i) {
        if (tau_data[i] > min_tau && temp_data[i] > 0 && dens_data[i] > 0 &&
            std::isfinite(temp_data[i]) && std::isfinite(dens_data[i])) {
            temp_filtered.push_back(temp_data[i]);
            dens_filtered.push_back(dens_data[i]);
        }
    }
    
    TemperatureDensityResult result;
    result.n_pixels = temp_filtered.size();
    
    if (result.n_pixels < 100) {
        result.temperature = Eigen::VectorXd(0);
        result.density = Eigen::VectorXd(0);
        result.log_T = Eigen::VectorXd(0);
        result.log_rho = Eigen::VectorXd(0);
        result.T0 = std::nan("");
        result.gamma = std::nan("");
        result.gamma_err = std::nan("");
        return result;
    }
    
    // Convert float to double
    std::vector<double> temp_dbl(temp_filtered.begin(), temp_filtered.end());
    std::vector<double> dens_dbl(dens_filtered.begin(), dens_filtered.end());
    
    result.temperature = Eigen::VectorXd::Map(temp_dbl.data(), temp_dbl.size());
    result.density = Eigen::VectorXd::Map(dens_dbl.data(), dens_dbl.size());
    
    // Compute median manually
    std::vector<double> sorted_density = dens_dbl;
    std::sort(sorted_density.begin(), sorted_density.end());
    float rho_mean = sorted_density[sorted_density.size() / 2];
    Eigen::VectorXd overdensity = result.density / rho_mean;
    
    Eigen::VectorXd log_T(result.n_pixels);
    Eigen::VectorXd log_rho(result.n_pixels);
    for (int i = 0; i < result.n_pixels; ++i) {
        log_T(i) = std::log10(result.temperature(i));
        log_rho(i) = std::log10(overdensity(i));
    }
    result.log_T = log_T;
    result.log_rho = log_rho;
    
    const int n_bins = 30;
    double log_rho_min, log_rho_max;
    log_rho.minCoeff(&log_rho_min);
    log_rho.maxCoeff(&log_rho_max);
    Eigen::VectorXd rho_bins = Eigen::VectorXd::LinSpaced(n_bins, log_rho_min, log_rho_max);
    
    std::vector<double> T_median;
    std::vector<double> rho_centers;
    
    for (int i = 0; i < n_bins - 1; ++i) {
        std::vector<double> T_in_bin;
        for (int j = 0; j < result.n_pixels; ++j) {
            if (log_rho(j) >= rho_bins(i) && log_rho(j) < rho_bins(i + 1)) {
                T_in_bin.push_back(log_T(j));
            }
        }
        if (T_in_bin.size() > 10) {
            std::sort(T_in_bin.begin(), T_in_bin.end());
            T_median.push_back(T_in_bin[T_in_bin.size() / 2]);
            rho_centers.push_back((rho_bins(i) + rho_bins(i + 1)) / 2);
        }
    }
    
    if (rho_centers.size() > 5) {
        int n = rho_centers.size();
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
        for (int i = 0; i < n; ++i) {
            sum_x += rho_centers[i];
            sum_y += T_median[i];
            sum_xy += rho_centers[i] * T_median[i];
            sum_xx += rho_centers[i] * rho_centers[i];
        }
        double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        double intercept = (sum_y - slope * sum_x) / n;
        
        result.gamma = slope + 1.0;
        result.T0 = std::pow(10.0, intercept);
        result.gamma_err = std::nan("");
    } else {
        result.T0 = std::nan("");
        result.gamma = std::nan("");
        result.gamma_err = std::nan("");
    }
    
    return result;
}

}
}
