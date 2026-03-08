#include "column_density.h"
#include "constants.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

namespace cgm {
namespace analysis {

ColumnDensityResult compute_column_density_distribution(
    const Eigen::Ref<const Eigen::ArrayXXf>& tau,
    double velocity_spacing,
    float threshold,
    const Eigen::Ref<const Eigen::ArrayXXf>* colden,
    double redshift,
    double box_size_ckpc_h,
    double hubble,
    double omega_m) {
    
    const int n_sightlines = tau.rows();
    const int n_pixels = tau.cols();
    
    std::vector<double> column_densities;
    column_densities.reserve(n_sightlines * 10);
    
    for (int i = 0; i < n_sightlines; ++i) {
        const auto& tau_line = tau.row(i);
        const float* colden_line = nullptr;
        if (colden) {
            colden_line = colden->row(i).data();
        }
        
        bool in_feature = false;
        int feature_start = 0;
        
        for (int j = 0; j < n_pixels; ++j) {
            bool absorbing = tau_line(j) > threshold;
            
            if (absorbing && !in_feature) {
                in_feature = true;
                feature_start = j;
            } else if (!absorbing && in_feature) {
                double N_HI;
                if (colden_line) {
                    float max_colden = 0;
                    for (int k = feature_start; k < j; ++k) {
                        max_colden = std::max(max_colden, colden_line[k]);
                    }
                    N_HI = max_colden;
                } else {
                    double tau_sum = 0;
                    for (int k = feature_start; k < j; ++k) {
                        tau_sum += tau_line(k);
                    }
                    N_HI = constants::TAU_TO_COLDEN_CONSTANT * tau_sum * velocity_spacing;
                }
                
                if (N_HI > constants::COLUMN_DENSITY_MIN) {
                    column_densities.push_back(N_HI);
                }
                in_feature = false;
            }
        }
        
        if (in_feature) {
            double N_HI;
            if (colden_line) {
                float max_colden = 0;
                for (int k = feature_start; k < n_pixels; ++k) {
                    max_colden = std::max(max_colden, colden_line[k]);
                }
                N_HI = max_colden;
            } else {
                double tau_sum = 0;
                for (int k = feature_start; k < n_pixels; ++k) {
                    tau_sum += tau_line(k);
                }
                N_HI = constants::TAU_TO_COLDEN_CONSTANT * tau_sum * velocity_spacing;
            }
            if (N_HI > constants::COLUMN_DENSITY_MIN) {
                column_densities.push_back(N_HI);
            }
        }
    }
    
    double dX;
    if (!std::isnan(redshift) && !std::isnan(box_size_ckpc_h)) {
        dX = box_size_ckpc_h / hubble / 1000.0;
    } else {
        dX = 1.0;
    }
    
    ColumnDensityResult result;
    result.n_sightlines = n_sightlines;
    result.dX = dX;
    result.redshift = redshift;
    
    if (column_densities.empty()) {
        result.N_HI = Eigen::VectorXd(0);
        result.counts = Eigen::VectorXi::Zero(49);
        result.bins = Eigen::VectorXd::LinSpaced(50, 1e12, 1e22);
        result.bin_centers = Eigen::VectorXd::Zero(49);
        result.f_N = Eigen::VectorXd::Zero(49);
        result.beta_fit = std::nan("");
        result.n_absorbers = 0;
        return result;
    }
    
    result.N_HI = Eigen::VectorXd::Map(column_densities.data(), column_densities.size());
    result.n_absorbers = column_densities.size();
    
    const int n_bins = 49;
    const double log_N_min = 12.0;
    const double log_N_max = 22.0;
    
    Eigen::VectorXd bins = Eigen::VectorXd::LinSpaced(n_bins + 1, log_N_min, log_N_max);
    for (int i = 0; i <= n_bins; ++i) {
        bins(i) = std::pow(10.0, bins(i));
    }
    
    Eigen::VectorXi counts = Eigen::VectorXi::Zero(n_bins);
    for (double N : column_densities) {
        if (N >= bins(0) && N <= bins(n_bins)) {
            double log_N = std::log10(N);
            int bin_idx = static_cast<int>((log_N - log_N_min) / (log_N_max - log_N_min) * n_bins);
            bin_idx = std::clamp(bin_idx, 0, n_bins - 1);
            counts(bin_idx)++;
        }
    }
    
    Eigen::VectorXd bin_centers = (bins.head(n_bins) + bins.tail(n_bins)) / 2.0;
    Eigen::VectorXd delta_log_N = Eigen::VectorXd::Constant(n_bins, (log_N_max - log_N_min) / n_bins);
    
    Eigen::VectorXd f_N(n_bins);
    double norm_factor = n_sightlines * dX;
    for (int i = 0; i < n_bins; ++i) {
        f_N(i) = static_cast<double>(counts(i)) / (norm_factor * delta_log_N(i));
    }
    
    result.bins = bins;
    result.bin_centers = bin_centers;
    result.counts = counts;
    result.f_N = f_N;
    result.beta_fit = std::nan("");
    
    return result;
}

}
}
