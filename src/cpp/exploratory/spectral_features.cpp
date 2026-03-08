#include "spectral_features.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace cgm {
namespace exploratory {

SpectralFeaturesResult extract_spectral_features(
    const Eigen::Ref<const Eigen::ArrayXXf>& tau,
    float velocity_spacing,
    float void_threshold,
    float line_threshold,
    float absorber_threshold,
    int max_sightlines,
    int max_separations,
    float max_separation) {
    
    const int n_sightlines = tau.rows();
    const int n_pixels = tau.cols();
    
    Eigen::ArrayXXf flux = (-tau).array().exp();
    
    std::vector<double> void_sizes;
    std::vector<double> line_widths;
    
    void_sizes.reserve(n_sightlines * 10);
    line_widths.reserve(n_sightlines * 10);
    
    for (int i = 0; i < n_sightlines; ++i) {
        const auto& flux_row = flux.row(i);
        
        bool in_void = false;
        int void_size = 0;
        for (int j = 0; j < n_pixels; ++j) {
            if (flux_row(j) > void_threshold) {
                if (!in_void) {
                    in_void = true;
                    void_size = 1;
                } else {
                    ++void_size;
                }
            } else {
                if (in_void) {
                    void_sizes.push_back(void_size * velocity_spacing);
                    in_void = false;
                }
            }
        }
        if (in_void) {
            void_sizes.push_back(void_size * velocity_spacing);
        }
        
        bool in_line = false;
        int line_width = 0;
        for (int j = 0; j < n_pixels; ++j) {
            if (flux_row(j) < line_threshold) {
                if (!in_line) {
                    in_line = true;
                    line_width = 1;
                } else {
                    ++line_width;
                }
            } else {
                if (in_line) {
                    line_widths.push_back(line_width * velocity_spacing);
                    in_line = false;
                }
            }
        }
        if (in_line) {
            line_widths.push_back(line_width * velocity_spacing);
        }
    }
    
    auto total_pixels = static_cast<float>(flux.size());
    float sat_frac = (flux < 0.1f).cast<float>().sum() / total_pixels;
    float deep_frac = ((flux >= 0.1f) && (flux < 0.5f)).cast<float>().sum() / total_pixels;
    float trans_frac = (flux >= 0.5f).cast<float>().sum() / total_pixels;
    
    float flux_mean = flux.mean();
    float flux_var = (flux - flux_mean).square().mean();
    float flux_std = std::sqrt(flux_var);
    
    float flux_skew = 0.0f, flux_kurt = 0.0f;
    if (flux_std > 0) {
        Eigen::ArrayXXf flux_norm = (flux - flux_mean) / flux_std;
        flux_skew = flux_norm.cube().mean();
        flux_kurt = (flux_norm.pow(4)).mean() - 3.0f;
    }
    
    std::vector<double> absorber_positions;
    absorber_positions.reserve(max_sightlines * 10);
    
    int n_sightlines_to_process = std::min(max_sightlines, n_sightlines);
    for (int i = 0; i < n_sightlines_to_process; ++i) {
        const auto& tau_row = tau.row(i);
        for (int j = 1; j < n_pixels - 1; ++j) {
            if (tau_row(j) > absorber_threshold && 
                tau_row(j) > tau_row(j-1) && 
                tau_row(j) > tau_row(j+1)) {
                absorber_positions.push_back(j * velocity_spacing);
            }
        }
    }
    
    std::vector<double> separations;
    if (absorber_positions.size() > 10) {
        std::sort(absorber_positions.begin(), absorber_positions.end());
        
        for (size_t i = 0; i < absorber_positions.size() && separations.size() < static_cast<size_t>(max_separations); ++i) {
            auto it = std::lower_bound(absorber_positions.begin() + i + 1,
                                      absorber_positions.end(),
                                      absorber_positions[i] + 1.0);
            while (it != absorber_positions.end() && 
                   (*it - absorber_positions[i]) < max_separation &&
                   separations.size() < static_cast<size_t>(max_separations)) {
                separations.push_back(*it - absorber_positions[i]);
                ++it;
            }
        }
    }
    
    SpectralFeaturesResult result;
    
    if (!void_sizes.empty()) {
        result.void_sizes = Eigen::VectorXd::Map(void_sizes.data(), void_sizes.size());
        result.mean_void_size = result.void_sizes.mean();
        Eigen::VectorXd sorted_void = result.void_sizes;
        std::sort(sorted_void.data(), sorted_void.data() + sorted_void.size());
        result.median_void_size = sorted_void(sorted_void.size() / 2);
        result.n_voids = void_sizes.size();
    } else {
        result.void_sizes = Eigen::VectorXd(0);
        result.mean_void_size = 0.0;
        result.median_void_size = 0.0;
        result.n_voids = 0;
    }
    
    if (!line_widths.empty()) {
        result.line_widths = Eigen::VectorXd::Map(line_widths.data(), line_widths.size());
        result.mean_line_width = result.line_widths.mean();
        Eigen::VectorXd sorted_lines = result.line_widths;
        std::sort(sorted_lines.data(), sorted_lines.data() + sorted_lines.size());
        result.median_line_width = sorted_lines(sorted_lines.size() / 2);
        result.n_lines = line_widths.size();
    } else {
        result.line_widths = Eigen::VectorXd(0);
        result.mean_line_width = 0.0;
        result.median_line_width = 0.0;
        result.n_lines = 0;
    }
    
    if (!separations.empty()) {
        result.absorber_separations = Eigen::VectorXd::Map(separations.data(), separations.size());
        result.mean_absorber_separation = result.absorber_separations.mean();
        result.n_absorbers = absorber_positions.size();
    } else {
        result.absorber_separations = Eigen::VectorXd(0);
        result.mean_absorber_separation = 0.0;
        result.n_absorbers = 0;
    }
    
    result.saturation_fraction = sat_frac;
    result.deep_absorption_fraction = deep_frac;
    result.transmission_fraction = trans_frac;
    result.flux_mean = flux_mean;
    result.flux_variance = flux_var;
    result.flux_skewness = flux_skew;
    result.flux_kurtosis = flux_kurt;
    
    return result;
}

}
}
