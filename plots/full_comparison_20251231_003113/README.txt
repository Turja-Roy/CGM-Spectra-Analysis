CAMEL Full Comparison Analysis Results
Generated: Wed Dec 31 00:31:13 CST 2025

Directory Structure:
====================

cross_sim_igm/
    - Cross-simulation comparison at z~2 (IGM, random sightlines)
    - 6 simulations: LH_80, LH_100, LH_200, LH_300, LH_400, LH_500
    - Mode: FULL (all exploratory analyses)
    - Key files:
        * comparison_enhanced.png - 10-panel enhanced comparison
        * feature_comparison.png - Spectral features
        * spectra_clustering.png - PCA/t-SNE clustering
        * statistical_tests.txt - Significance results
        * pairwise_ks_matrix.png - NxN KS test heatmap

cross_sim_cgm/
    - Cross-simulation comparison at z~2 (CGM, targeted sightlines)
    - Same structure as cross_sim_igm/

evolution_igm/
    - Time evolution for LH_80 (IGM)
    - 8 snapshots from z~3.5 to z~0.8
    - Mode: DETAILED
    - Tracks: τ_eff(z), <F>(z), N_absorbers(z), T-ρ evolution

evolution_cgm/
    - Time evolution for LH_80 (CGM)
    - Same structure as evolution_igm/

Quick Start:
============
1. Look at comparison_enhanced.png for overview
2. Check statistical_tests.txt for significance
3. Examine feature_comparison.png for interpretable differences
4. Look at spectra_clustering.png to see if sims separate

Pattern Discovery Checklist:
=============================
□ Do box plots show different variances?
□ Do power spectrum ratios deviate from 1?
□ Are there significant differences in statistical_tests.txt?
□ Do PCA/t-SNE plots show separated clusters?
□ Are feature z-scores large (>1) in feature matrix?
□ Do distributions differ in CDFs/QQ-plots?

Logs:
=====
See logs/ directory for detailed output from each analysis.

Commands Run:
=============
Pane 0: Cross-simulation IGM (full mode)
Pane 1: Cross-simulation CGM (full mode)
Pane 2: Evolution IGM (detailed mode)
Pane 3: Evolution CGM (detailed mode)

