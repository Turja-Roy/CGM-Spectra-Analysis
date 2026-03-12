#include "column_density.h"
#include "constants.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

namespace cgm {
namespace analysis {

// Version with raw pointer - properly handles row-major numpy arrays
ColumnDensityResult compute_column_density_distribution(
    const Eigen::Ref<const Eigen::ArrayXXf>& tau,
    double velocity_spacing,
    float threshold,
    const float* colden_data,
    int colden_rows,
    int colden_cols,
    double redshift,
    double box_size_ckpc_h,
    double hubble,
    double omega_m) {
    
    const int n_sightlines = tau.rows();
    const int n_pixels = tau.cols();
    
    // Check if colden is valid and matches expected dimensions
    const bool has_colden = (colden_data != nullptr && colden_rows == n_sightlines && colden_cols == n_pixels);
    
    std::vector<double> column_densities;
    column_densities.reserve(n_sightlines * 10);
    
    for (int i = 0; i < n_sightlines; ++i) {
        const auto& tau_line = tau.row(i);
        const float* colden_line = nullptr;
        if (has_colden) {
            colden_line = &colden_data[i * n_pixels];  // Row-major access from Python
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
        result.counts = Eigen::VectorXi::Zero(50);
        result.bins = Eigen::VectorXd::LinSpaced(51, 1e12, 1e22);
        result.bin_centers = Eigen::VectorXd::Zero(50);
        result.f_N = Eigen::VectorXd::Zero(50);
        result.beta_fit = std::nan("");
        result.n_absorbers = 0;
        return result;
    }
    
    result.N_HI = Eigen::VectorXd::Map(column_densities.data(), column_densities.size());
    result.n_absorbers = column_densities.size();
    
    const int n_bins = 50;
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
    
    // Fit power law: f(N) = A * N^(-beta) in range 12 < log(N) < 14.5
    double beta_fit = std::nan("");
    
    std::vector<double> log_N_fit;
    std::vector<double> log_f_fit;
    
    for (int i = 0; i < n_bins; ++i) {
        if (bin_centers(i) > 1e12 && bin_centers(i) < 3e14 && counts(i) > 0) {
            double log_N = std::log10(bin_centers(i));
            double f_val = f_N(i);
            if (f_val > 0 && std::isfinite(log_N)) {
                log_N_fit.push_back(log_N);
                log_f_fit.push_back(std::log10(f_val + 1e-10));
            }
        }
    }
    
    if (log_N_fit.size() > 5) {
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
        int n = log_N_fit.size();
        for (int i = 0; i < n; ++i) {
            sum_x += log_N_fit[i];
            sum_y += log_f_fit[i];
            sum_xy += log_N_fit[i] * log_f_fit[i];
            sum_xx += log_N_fit[i] * log_N_fit[i];
        }
        
        double denominator = n * sum_xx - sum_x * sum_x;
        if (std::abs(denominator) > 1e-10) {
            double slope = (n * sum_xy - sum_x * sum_y) / denominator;
            beta_fit = -slope;
        }
    }
    
    result.beta_fit = beta_fit;
    
    return result;
}

}
}
