#ifndef CGM_ANALYSIS_CONSTANTS_H
#define CGM_ANALYSIS_CONSTANTS_H

#include <cmath>

namespace cgm {
namespace constants {

constexpr double TAU_TO_COLDEN_CONSTANT = 8.51e11;
constexpr double LYMAN_ALPHA_WAVELENGTH = 1215.67e-10;
constexpr double DAMPING_PARAMETER = 4.7e-4;
constexpr double B_TO_T_FACTOR = 60.57;  // K / (km/s)^2: T = 60.57 * b^2
constexpr double DEFAULT_HUBBLE = 0.6774;
constexpr double DEFAULT_OMEGA_M = 0.3089;
constexpr double DEFAULT_OMEGA_LAMBDA = 0.6911;
constexpr double DEFAULT_TAU_THRESHOLD = 0.5;
constexpr double COLUMN_DENSITY_MIN = 1e12;

}
}

#endif
