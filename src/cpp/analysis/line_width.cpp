#include "line_width.h"
#include "constants.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace cgm {
namespace analysis {

LineWidthResult compute_line_width_distribution(
    const Eigen::Ref<const Eigen::ArrayXXf>& tau,
    double velocity_spacing,
    float threshold,
    const Eigen::Ref<const Eigen::ArrayXXf>* colden) {
    
    const int n_sightlines = tau.rows();
    const int n_pixels = tau.cols();
    
    std::vector<double> column_densities;
    std::vector<double> b_params;
    column_densities.reserve(n_sightlines * 10);
    b_params.reserve(n_sightlines * 10);
    
    for (int i = 0; i < n_sightlines; ++i) {
        const auto& tau_line = tau.row(i);
        const float* colden_line = nullptr;
        if (colden) {
            colden_line = colden->row(i).data();
        }
        
        std::vector<int> peaks;
        for (int j = 1; j < n_pixels - 1; ++j) {
            if (tau_line(j) > threshold && 
                tau_line(j) > tau_line(j-1) && 
                tau_line(j) > tau_line(j+1)) {
                peaks.push_back(j);
            }
        }
        
        for (int peak_idx : peaks) {
            int left = peak_idx;
            while (left > 0 && tau_line(left) > threshold * 0.3f) {
                --left;
            }
            
            int right = peak_idx;
            while (right < n_pixels - 1 && tau_line(right) > threshold * 0.3f) {
                ++right;
            }
            
            if (right - left < 3) continue;
            
            float tau_peak = tau_line(peak_idx);
            float fwhm = (right - left) * velocity_spacing;
            float b = fwhm / (2.0f * std::sqrt(std::log(2.0f)));
            b = std::clamp(b, 2.0f, 80.0f);
            
            double N_HI;
            if (colden_line) {
                float max_colden = 0;
                for (int k = left; k <= right; ++k) {
                    max_colden = std::max(max_colden, colden_line[k]);
                }
                N_HI = max_colden;
            } else {
                double tau_sum = 0;
                for (int k = left; k <= right; ++k) {
                    tau_sum += tau_line(k);
                }
                N_HI = constants::TAU_TO_COLDEN_CONSTANT * tau_sum * velocity_spacing;
            }
            
            if (N_HI > constants::COLUMN_DENSITY_MIN && b > 2.0 && b < 80.0) {
                column_densities.push_back(N_HI);
                b_params.push_back(b);
            }
        }
    }
    
    LineWidthResult result;
    if (b_params.empty()) {
        result.N_HI = Eigen::VectorXd(0);
        result.b_params = Eigen::VectorXd(0);
        result.temperatures = Eigen::VectorXd(0);
        result.b_median = std::nan("");
        result.b_mean = std::nan("");
        result.b_std = std::nan("");
        result.n_absorbers = 0;
        return result;
    }
    
    result.N_HI = Eigen::VectorXd::Map(column_densities.data(), column_densities.size());
    result.b_params = Eigen::VectorXd::Map(b_params.data(), b_params.size());
    result.temperatures = result.b_params.array().square() * constants::B_TO_T_FACTOR;
    result.n_absorbers = b_params.size();
    
    Eigen::VectorXd sorted_b = result.b_params;
    std::sort(sorted_b.data(), sorted_b.data() + sorted_b.size());
    result.b_median = sorted_b(sorted_b.size() / 2);
    result.b_mean = result.b_params.mean();
    
    double b_mean = result.b_mean;
    double variance = 0.0;
    for (int i = 0; i < result.b_params.size(); ++i) {
        double diff = result.b_params(i) - b_mean;
        variance += diff * diff;
    }
    variance /= result.b_params.size();
    result.b_std = std::sqrt(variance);
    
    return result;
}

}
}
