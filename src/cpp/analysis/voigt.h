#ifndef CGM_ANALYSIS_VOIGT_H
#define CGM_ANALYSIS_VOIGT_H

#include <cmath>
#include <complex>
#include <limits>
#include "Faddeeva.hh"

namespace cgm {
namespace analysis {

constexpr double VOIGT_DAMPING = 4.7e-4;
constexpr double SQRT_PI_INV = 0.5641895835477563;
constexpr double SQRT_LN2 = 0.8325546111576977;

struct VoigtResult {
    double tau_0;
    double b_param;
    double v_center;
    double residual;
};

namespace internal {

inline double exp_safe(double x) {
    if (x > 700.0) return std::numeric_limits<double>::infinity();
    if (x < -700.0) return 0.0;
    return std::exp(x);
}

inline double voigt(double x, double a) {
    std::complex<double> z(x, a);
    std::complex<double> w = Faddeeva::w(z);
    return w.real() * SQRT_PI_INV;
}

}

double compute_voigt_optical_depth(
    double v,
    double tau_0,
    double b,
    double v_center,
    double damping = VOIGT_DAMPING);

}
}

#endif
