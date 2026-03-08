#ifndef CGM_ANALYSIS_FLUX_STATS_H
#define CGM_ANALYSIS_FLUX_STATS_H

#include <Eigen/Dense>

namespace cgm {
namespace analysis {

struct FluxStatsResult {
    double mean_flux;
    double median_flux;
    double std_flux;
    double min_flux;
    double max_flux;
    double mean_tau;
    double median_tau;
    double effective_tau;
    double deep_absorption_frac;
    double moderate_absorption_frac;
    double weak_absorption_frac;
};

FluxStatsResult compute_flux_statistics(const Eigen::ArrayXXf& tau);

}
}

#endif
