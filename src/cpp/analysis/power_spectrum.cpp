#include "power_spectrum.h"
#include "constants.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <complex>
#include <omp.h>

namespace cgm {
namespace analysis {

PowerSpectrumResult compute_power_spectrum(
    const Eigen::Ref<const Eigen::ArrayXXf>& flux,
    double velocity_spacing,
    int chunk_size) {
    
    const int n_sightlines = flux.rows();
    const int n_pixels = flux.cols();
    
    float mean_flux = flux.mean();
    if (mean_flux <= 0) {
        throw std::invalid_argument("Mean flux must be positive");
    }
    
    int n_k = n_pixels / 2 + 1;
    Eigen::VectorXd k(n_k);
    for (int i = 0; i < n_k; ++i) {
        k(i) = 2.0 * M_PI * i / (n_pixels * velocity_spacing);
    }
    
    std::vector<double> power_sum(n_k, 0.0);
    std::vector<double> power_sum_sq(n_k, 0.0);
    
    int n_chunks = (n_sightlines + chunk_size - 1) / chunk_size;
    
    #pragma omp parallel
    {
        std::vector<double> local_power_sum(n_k, 0.0);
        std::vector<double> local_power_sum_sq(n_k, 0.0);
        
        #pragma omp for schedule(dynamic)
        for (int c = 0; c < n_chunks; ++c) {
            int start = c * chunk_size;
            int end = std::min((c + 1) * chunk_size, n_sightlines);
            int chunk_n = end - start;
            
            for (int i = 0; i < chunk_n; ++i) {
                Eigen::ArrayXf delta_F = flux.row(start + i) / mean_flux - 1.0f;
                
                for (int j = 0; j < n_k; ++j) {
                    float real = 0.0f, imag = 0.0f;
                    for (int n = 0; n < n_pixels; ++n) {
                        float angle = -2.0f * M_PI * j * n / n_pixels;
                        real += delta_F(n) * cosf(angle);
                        imag += delta_F(n) * sinf(angle);
                    }
                    
                    float power = (real * real + imag * imag) / n_pixels;
                    local_power_sum[j] += power;
                    local_power_sum_sq[j] += power * power;
                }
            }
        }
        
        #pragma omp critical
        {
            for (int j = 0; j < n_k; ++j) {
                power_sum[j] += local_power_sum[j];
                power_sum_sq[j] += local_power_sum_sq[j];
            }
        }
    }
    
    Eigen::VectorXd power_sum_eigen = Eigen::VectorXd::Map(power_sum.data(), n_k);
    Eigen::VectorXd power_sum_sq_eigen = Eigen::VectorXd::Map(power_sum_sq.data(), n_k);
    
    Eigen::VectorXd P_k_mean = (power_sum_eigen / n_sightlines) * velocity_spacing;
    Eigen::VectorXd mean_power = power_sum_eigen / n_sightlines;
    Eigen::VectorXd mean_power_sq = power_sum_sq_eigen / n_sightlines;
    
    Eigen::VectorXd variance(n_k);
    for (int j = 0; j < n_k; ++j) {
        double mp = mean_power(j);
        double mp2 = mean_power_sq(j);
        double v = mp2 - mp * mp;
        variance(j) = (v > 0) ? v : 0;
    }
    
    Eigen::VectorXd P_k_std(n_k);
    for (int j = 0; j < n_k; ++j) {
        P_k_std(j) = std::sqrt(variance(j)) * velocity_spacing;
    }
    
    Eigen::VectorXd P_k_err = P_k_std / std::sqrt(n_sightlines);
    Eigen::VectorXd n_modes = Eigen::VectorXd::Ones(n_k) * n_sightlines;
    
    PowerSpectrumResult result;
    result.k = k;
    result.P_k_mean = P_k_mean;
    result.P_k_std = P_k_std;
    result.P_k_err = P_k_err;
    result.n_modes = n_modes;
    result.mean_flux = mean_flux;
    result.n_sightlines = n_sightlines;
    result.velocity_spacing = velocity_spacing;
    
    return result;
}

}
}
