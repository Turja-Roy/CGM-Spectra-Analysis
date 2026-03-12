#include "voigt.h"

namespace cgm {
namespace analysis {

double compute_voigt_optical_depth(double v, double tau_0, double b,
                                   double v_center, double damping) {

    if (b <= 0.0)
        return 0.0;

    double u = (v - v_center) / b;
    double a = damping * 1215.67 / b;

    double a_scaled = a * 2.0 * SQRT_LN2;
    double voigt_profile = cgm::analysis::internal::voigt(u, a_scaled);

    double tau = tau_0 * voigt_profile * (2.0 * SQRT_LN2) / b;

    return tau;
}

namespace internal {

double voigt_residual(const double *params, const double *tau_data,
                      int n_pixels, double velocity_spacing, double damping) {

    double tau_0 = params[0];
    double b = params[1];
    double v_center = params[2];

    if (b <= 0.1 || b > 200.0 || tau_0 <= 0) {
        return 1e10;
    }

    double residual = 0.0;
    for (int i = 0; i < n_pixels; ++i) {
        double v = i * velocity_spacing;
        double tau_model =
            compute_voigt_optical_depth(v, tau_0, b, v_center, damping);
        double diff = tau_data[i] - tau_model;
        residual += diff * diff;
    }

    return residual;
}

} // namespace internal

} // namespace analysis

} // namespace cgm
