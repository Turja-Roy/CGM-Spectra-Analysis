#ifndef CGM_EXPLORATORY_SPECTRAL_FEATURES_H
#define CGM_EXPLORATORY_SPECTRAL_FEATURES_H

#include <Eigen/Dense>

namespace cgm {
namespace exploratory {

struct SpectralFeaturesResult {
    Eigen::VectorXd void_sizes;
    Eigen::VectorXd line_widths;
    Eigen::VectorXd absorber_separations;
    double mean_void_size;
    double median_void_size;
    double mean_line_width;
    double median_line_width;
    double saturation_fraction;
    double deep_absorption_fraction;
    double transmission_fraction;
    double flux_mean;
    double flux_variance;
    double flux_skewness;
    double flux_kurtosis;
    double mean_absorber_separation;
    int n_voids;
    int n_lines;
    int n_absorbers;
};

SpectralFeaturesResult extract_spectral_features(
    const Eigen::Ref<const Eigen::ArrayXXf>& tau,
    float velocity_spacing = 0.1f,
    float void_threshold = 0.9f,
    float line_threshold = 0.5f,
    float absorber_threshold = 0.5f,
    int max_sightlines = 100,
    int max_separations = 1000,
    float max_separation = 500.0f);

}
}

#endif
