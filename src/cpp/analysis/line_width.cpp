#include "line_width.h"
#include "constants.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace cgm {
namespace analysis {

// ============================================================================
// Line Width Distribution (Fallback Implementation)
// ============================================================================
//
// This C++ implementation provides a fast fallback for line width estimation
// when fake_spectra.voigtfit is not available. It uses:
//
// 1. Peak detection in optical depth profiles
// 2. FWHM-based Doppler parameter estimation (Gaussian approximation)
//
// For accurate Voigt profile fitting, use fake_spectra.voigtfit in Python,
// which provides exact Faddeeva function evaluation via scipy.special.wofz().
//
// The relationship between FWHM and Doppler parameter b for a Gaussian is:
//   FWHM = 2 * sqrt(ln(2)) * b ≈ 1.665 * b
//   => b ≈ FWHM / 1.665
//
// For Voigt profiles, the FWHM depends on both Doppler and damping widths,
// but for typical IGM conditions (damping parameter a ~ 4.7e-4 << 1),
// the Gaussian approximation is reasonable for initial estimates.
// ============================================================================

// Estimate b-parameter from FWHM using Gaussian approximation
// FWHM = 2 * sqrt(ln(2)) * b => b = FWHM / (2 * sqrt(ln(2)))
static float estimate_b_from_fwhm(float fwhm) {
    // 2 * sqrt(ln(2)) ≈ 1.6651
    constexpr float FWHM_TO_B = 1.0f / 1.6651f;
    return fwhm * FWHM_TO_B;
}

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
        // Use Eigen's row access directly (handles storage order correctly)
        auto tau_row = tau.row(i);
        
        // Find peaks in optical depth (simple local maximum detection)
        std::vector<int> peaks;
        for (int j = 1; j < n_pixels - 1; ++j) {
            if (tau_row(j) > threshold && 
                tau_row(j) > tau_row(j-1) && 
                tau_row(j) > tau_row(j+1)) {
                peaks.push_back(j);
            }
        }
        
        for (int peak_idx : peaks) {
            float tau_peak = tau_row(peak_idx);
            
            // Define feature extent (where tau drops below 30% of threshold)
            int left = peak_idx;
            while (left > 0 && tau_row(left) > threshold * 0.3f) {
                --left;
            }
            
            int right = peak_idx;
            while (right < n_pixels - 1 && tau_row(right) > threshold * 0.3f) {
                ++right;
            }
            
            // Need at least 3 pixels for meaningful measurement
            if (right - left < 3) continue;
            
            // Compute FWHM by finding half-maximum crossings
            float half_max = tau_peak * 0.5f;
            
            // Find left half-max crossing
            float v_left = 0.0f;
            for (int k = peak_idx; k >= 1; --k) {
                if (tau_row(k) >= half_max && tau_row(k-1) < half_max) {
                    float t = (half_max - tau_row(k-1)) / (tau_row(k) - tau_row(k-1));
                    v_left = (k - 1 + t) * velocity_spacing;
                    break;
                }
            }
            
            // Find right half-max crossing
            float v_right = (n_pixels - 1) * velocity_spacing;
            for (int k = peak_idx; k < n_pixels - 1; ++k) {
                if (tau_row(k) >= half_max && tau_row(k+1) < half_max) {
                    float t = (half_max - tau_row(k)) / (tau_row(k+1) - tau_row(k));
                    v_right = (k + t) * velocity_spacing;
                    break;
                }
            }
            
            float fwhm = v_right - v_left;
            if (fwhm <= 0.0f) {
                fwhm = velocity_spacing * 3.0f;  // Minimum 3 pixels
            }
            
            float b = estimate_b_from_fwhm(fwhm);
            
            // Clamp to physical range (2-80 km/s typical for IGM)
            b = std::clamp(b, 2.0f, 80.0f);
            
            // Compute column density
            double N_HI;
            if (colden && colden->rows() > 0) {
                // Use provided column density (max over feature)
                auto colden_row = colden->row(i);
                float max_colden = 0;
                for (int k = left; k <= right; ++k) {
                    max_colden = std::max(max_colden, colden_row(k));
                }
                N_HI = max_colden;
            } else {
                // Estimate from integrated optical depth
                // N_HI ≈ (m_e c / π e² f λ) ∫ τ dv
                double tau_sum = 0;
                for (int k = left; k <= right; ++k) {
                    tau_sum += tau_row(k);
                }
                N_HI = constants::TAU_TO_COLDEN_CONSTANT * tau_sum * velocity_spacing;
            }
            
            // Filter to physical range
            if (N_HI > constants::COLUMN_DENSITY_MIN && b > 2.0 && b < 80.0) {
                column_densities.push_back(N_HI);
                b_params.push_back(b);
            }
        }
    }
    
    // Build result structure
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
    
    // T = (m_H / 2 k_B) * b² where b is in km/s
    // Factor ≈ 60.57 K/(km/s)²
    result.temperatures = result.b_params.array().square() * constants::B_TO_T_FACTOR;
    result.n_absorbers = b_params.size();
    
    // Compute statistics
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
