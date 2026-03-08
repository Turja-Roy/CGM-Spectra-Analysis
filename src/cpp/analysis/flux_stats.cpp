#include "flux_stats.h"
#include "constants.h"
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <numeric>

namespace cgm {
namespace analysis {

FluxStatsResult compute_flux_statistics(const Eigen::ArrayXXf& tau) {
    FluxStatsResult result;
    
    Eigen::ArrayXXf flux = (-tau).array().exp();
    
    result.mean_flux = flux.mean();
    
    // Compute median manually
    std::vector<float> flux_flat(flux.data(), flux.data() + flux.size());
    std::sort(flux_flat.begin(), flux_flat.end());
    size_t n = flux_flat.size();
    if (n % 2 == 0) {
        result.median_flux = (flux_flat[n/2 - 1] + flux_flat[n/2]) / 2.0;
    } else {
        result.median_flux = flux_flat[n/2];
    }
    
    // Compute std manually
    float mean = result.mean_flux;
    float variance = 0.0f;
    for (float f : flux_flat) {
        variance += (f - mean) * (f - mean);
    }
    variance /= flux_flat.size();
    result.std_flux = std::sqrt(variance);
    
    result.min_flux = flux.minCoeff();
    result.max_flux = flux.maxCoeff();
    result.mean_tau = tau.mean();
    
    // Compute median for tau
    std::vector<float> tau_flat(tau.data(), tau.data() + tau.size());
    std::sort(tau_flat.begin(), tau_flat.end());
    if (tau_flat.size() % 2 == 0) {
        result.median_tau = (tau_flat[tau_flat.size()/2 - 1] + tau_flat[tau_flat.size()/2]) / 2.0;
    } else {
        result.median_tau = tau_flat[tau_flat.size()/2];
    }
    
    float mean_flux_val = static_cast<float>(result.mean_flux);
    result.effective_tau = (mean_flux_val > 0) ? -std::log(mean_flux_val) : std::numeric_limits<double>::infinity();
    
    auto total_pixels = static_cast<float>(flux.size());
    result.deep_absorption_frac = (flux < 0.1f).cast<float>().sum() / total_pixels;
    result.moderate_absorption_frac = ((flux >= 0.1f) && (flux < 0.5f)).cast<float>().sum() / total_pixels;
    result.weak_absorption_frac = (flux >= 0.5f).cast<float>().sum() / total_pixels;
    
    return result;
}

} // namespace analysis
} // namespace cgm
