# CAMEL Lyman-Alpha Forest Analysis Pipeline

Comprehensive Python toolkit for analyzing the Lyman-alpha forest from cosmological simulations (CAMEL project). Features multi-line spectral analysis, CGM-targeted observations, and memory-efficient comparison tools for large-scale parameter studies.

## Quick Start

### Installation
```bash
# Clone repository
cd /path/to/CGM

# Install dependencies
pip install h5py numpy matplotlib scipy scikit-learn

# For spectra generation (optional)
pip install fake_spectra
```

### Basic Usage
```bash
# List available data
python analyze_spectra.py list

# Generate IGM spectra
python analyze_spectra.py generate data/snap_080.hdf5 -n 10000

# Analyze spectra
python analyze_spectra.py analyze camel_lya_spectra_snap_080_n10000.hdf5

# Compare multiple simulations (memory-efficient!)
python analyze_spectra.py compare --mode detailed \
    spectra/LH_80/camel_lya_spectra_snap_082_n10000.hdf5 \
    spectra/LH_100/camel_lya_spectra_snap_082_n10000.hdf5 \
    spectra/LH_200/camel_lya_spectra_snap_082_n10000.hdf5 \
    -l 'LH_80,LH_100,LH_200'
```

---

## Commands Reference

### `list`
List all available simulation snapshots and spectra files
```bash
python analyze_spectra.py list
```

### `explore`
Explore HDF5 file structure (snapshots or spectra)
```bash
python analyze_spectra.py explore data/snap_080.hdf5 -d 2
```

### `generate`
Generate synthetic spectra from simulation snapshot

**Options:**
- `-n, --sightlines`: Number of random sightlines (default: 100)
- `-r, --res`: Velocity resolution in km/s (default: 0.1)
- `--line`: Spectral lines (default: lya)
  - Single: `--line lya`
  - Multiple: `--line lya,civ,ovi`
- `-o, --output`: Output file path

**Examples:**
```bash
# Basic IGM spectra
python analyze_spectra.py generate data/snap_080.hdf5 -n 10000

# Multi-line analysis
python analyze_spectra.py generate data/snap_080.hdf5 -n 10000 --line lya,civ,ovi

# Custom resolution
python analyze_spectra.py generate data/snap_080.hdf5 -n 5000 --res 0.05
```

### `analyze`
Comprehensive analysis of spectra file

**Performs:**
- Flux statistics and sample spectra
- Effective optical depth τ_eff
- Flux power spectrum P_F(k)
- Column density distribution f(N_HI)
- Line width distribution b(N_HI) and temperatures (if T/ρ available)
- Temperature-density relation T(ρ) (if available)
- Metal line statistics (if multi-line data)

**Example:**
```bash
python analyze_spectra.py analyze spectra_file.hdf5
```

**Output:** 8-12 plots in `plots/` directory

### `compare`
Compare multiple simulations with three analysis modes

**Options:**
- `-l, --labels`: Comma-separated labels
- `-o, --output`: Output directory
- `--mode {quick,detailed,full}`: Analysis depth

**Modes:**

**Quick** (~30s): Basic 6-panel comparison
```bash
python analyze_spectra.py compare --mode quick \
    file1.hdf5 file2.hdf5 file3.hdf5
```

**Detailed** (~2 min):

```bash
python analyze_spectra.py compare --mode detailed \
    --output plots/my_comparison \
    file1.hdf5 file2.hdf5 file3.hdf5 file4.hdf5 file5.hdf5 file6.hdf5 \
    -l 'LH_80,LH_100,LH_200,LH_300,LH_400,LH_500'
```

**Output files (detailed mode):**
- `comparison_basic.png` - Original 6-panel plot
- `comparison_enhanced.png` - 10-panel with box plots, ratios, sample spectra
- `flux_distributions.png` - CDFs, QQ-plots, histograms, box plots
- `power_spectrum_ratios.png` - Scale-dependent differences
- `correlation_matrices.png` - Observable correlations per simulation
- `statistical_tests.txt` - Comprehensive test results with p-values

**Full** (~5-10 min): Complete exploratory analysis

```bash
python analyze_spectra.py compare --mode full \
    file1.hdf5 file2.hdf5 file3.hdf5 \
    -l 'Sim1,Sim2,Sim3'
```

**Additional output files (full mode):**
- `feature_comparison.png` - 9-panel feature analysis
- `physics_regimes.png` - Analysis by absorption strength
- `spectra_clustering.png` - PCA and t-SNE projections
- `pairwise_ks_matrix.png` - N×N significance heatmap

**Example: Cross-Simulation Parameter Study**
```bash
python analyze_spectra.py compare --mode full \
    spectra/IllustrisTNG/LH/LH_80/camel_lya_spectra_snap_082_n10000.hdf5 \
    spectra/IllustrisTNG/LH/LH_100/camel_lya_spectra_snap_082_n10000.hdf5 \
    spectra/IllustrisTNG/LH/LH_200/camel_lya_spectra_snap_082_n10000.hdf5 \
    spectra/IllustrisTNG/LH/LH_300/camel_lya_spectra_snap_082_n10000.hdf5 \
    spectra/IllustrisTNG/LH/LH_400/camel_lya_spectra_snap_082_n10000.hdf5 \
    spectra/IllustrisTNG/LH/LH_500/camel_lya_spectra_snap_082_n10000.hdf5 \
    -l 'LH_80,LH_100,LH_200,LH_300,LH_400,LH_500' \
    -o plots/cross_simulation_study
```

### `evolve`
Track redshift evolution of observables

**Options:**
- `-l, --labels`: Comma-separated redshift labels
- `-o, --output`: Output plot path
- `--mode {quick,detailed,full}`: Analysis depth (same as compare)

**Example:**
```bash
python analyze_spectra.py evolve --mode detailed \
    spectra_snap_076.hdf5 \
    spectra_snap_078.hdf5 \
    spectra_snap_080.hdf5 \
    spectra_snap_082.hdf5 \
    spectra_snap_084.hdf5 \
    -l "z=3.5,z=3.0,z=2.5,z=2.0,z=1.5" \
    -o plots/thermal_evolution
```

**Output:** Evolution plots showing τ_eff(z), <F>(z), T₀(z), γ(z)

### `diagnose`
Deep diagnostic analysis of a single spectra file

**Options:**
- `--features`: Extract spectral features (void sizes, line widths, clustering)
- `--distribution`: Detailed flux distribution analysis

**Example:**
```bash
python analyze_spectra.py diagnose spectra.hdf5 \
    --features \
    --distribution \
    --output plots/diagnostics
```

**Output:**
- Feature extraction plots
- Flux distribution analysis
- Detailed statistics

### `pipeline`
Full pipeline: generate + analyze in one command

**Example:**
```bash
python analyze_spectra.py pipeline data/snap_080.hdf5 -n 10000 --res 0.1
```

### `halo`
Analyze individual galaxy halos and their CGM properties

**Options:**
- `--mass-range MIN MAX`: Halo mass range in log10(M_sun) (default: 11.0 12.5)
- `--halo-id ID`: Analyze specific halo by ID
- `--n-halos N`: Number of top massive halos to analyze (default: 10)
- `--isolated-only`: Only analyze isolated halos
- `--output-dir DIR`: Output directory for plots
- `--slice-thickness`: Projection slice thickness in ckpc/h (default: 1000)
- `--plot-type`: Type of plot (projection, temperature, radial, summary)

**Example:**
```bash
python analyze_spectra.py halo data/snap_080.hdf5 \
    --mass-range 11.0 12.5 \
    --n-halos 10 \
    --plot-type summary
```

### `cgm`
Generate halo-targeted CGM spectra around massive halos

**Options:**
- `--mass-range MIN MAX`: Halo mass range in log10(M_sun) (default: 11.0 12.5)
- `--n-halos N`: Number of top massive halos to use (default: all in mass range)
- `--impact-params`: Impact parameters in Rvir, comma-separated (default: 0.25,0.5,0.75,1.0,1.25)
- `--n-per-bin N`: Number of sightlines per impact parameter bin (default: 100)
- `--azimuthal N`: Number of azimuthal angle samples (default: 8)
- `--line`: Spectral lines (comma-separated, default: lya)
- `--output`: Output spectra filename (default: auto-generated)
- `--isolated-only`: Only use isolated halos

**Example:**
```bash
python analyze_spectra.py cgm data/snap_080.hdf5 \
    --mass-range 11.0 12.0 \
    --impact-params 0.5,1.0,1.5,2.0 \
    --n-per-bin 100 \
    --line lya,civ,ovi
```

---

## Output Files

### Spectra Files (HDF5)
Generated spectra stored in hierarchical format:
```
spectra/
├── IllustrisTNG/
│   └── LH/
│       ├── LH_80/
│       │   ├── camel_lya_spectra_snap_082_n10000.hdf5
│       │   └── cgm/
│       │       └── cgm_lya_spectra_snap_082_n300.hdf5
│       └── LH_100/
│           └── ...

File structure (HDF5):
camel_lya_spectra_snap_XXX_nXXXX.hdf5
├── Header/
│   ├── redshift
│   ├── hubble
│   └── box_size
├── tau/                      # Multi-line support
│   ├── H/1/1215              # HI Lyα
│   ├── C/4/1548              # CIV (if requested)
│   └── O/6/1031              # OVI (if requested)
├── temperature/H/1/          # Temperature field
├── density_weight_density/H/1/  # Density field
├── flux                      # Transmitted flux
└── wavelength                # Wavelength array
```

### Plot Outputs

**Structure**: Plots are automatically organized by simulation structure:
```
plots/
├── IllustrisTNG/
│   └── LH/
│       ├── LH_80/
│       │   ├── camel_*.png
│       │   ├── cgm/
│       │   │   └── camel_*.png
│       │   └── halos/
│       │       └── halo_*_summary_*.png
│       └── LH_100/
│           └── ...
└── comparisons/
    └── comparison_*.png
```

**Basic analysis:**
- `sample_spectra_*.png`
- `flux_statistics_*.png`
- `flux_power_spectrum_*.png`
- `column_density_distribution_*.png`

**Advanced analysis:**
- `line_width_distribution_*.png`
- `temperature_density_relation_*.png`
- `multi_line_comparison_*.png`

**CGM/Halo analysis:**
- `halo_*_summary_*.png` - Individual halo properties
- CGM spectra plots in `cgm/` subdirectory

**Comparison (detailed mode):**
- `comparison_basic.png` (6 panels)
- `comparison_enhanced.png` (10 panels)
- `flux_distributions.png` (4 panels)
- `power_spectrum_ratios.png`
- `correlation_matrices.png`
- `statistical_tests.txt`

**Comparison (full mode):**
- All detailed mode files plus:
- `feature_comparison.png` (9 panels)
- `physics_regimes.png`
- `spectra_clustering.png`
- `pairwise_ks_matrix.png`

---

## Common Workflows

### 1. Quick Test (10 min)
```bash
# Generate small dataset
python analyze_spectra.py generate data/snap_080.hdf5 -n 100

# Analyze
python analyze_spectra.py analyze camel_lya_spectra_snap_080_n100.hdf5
```

### 2. Production Analysis
```bash
# Generate 10k sightlines with multiple lines
python analyze_spectra.py generate data/snap_080.hdf5 \
    -n 10000 \
    --line lya,civ,ovi

# Full analysis
python analyze_spectra.py analyze camel_lya_spectra_snap_080_n10000.hdf5
```

### 3. Parameter Study with Enhanced Comparison
```bash
# Generate spectra for 6 parameter variations
for lh in 80 100 200 300 400 500; do
    python analyze_spectra.py generate \
        data/IllustrisTNG/LH/LH_${lh}/snap_082.hdf5 \
        -n 10000 \
        -o spectra_LH_${lh}.hdf5
done

# Compare with full exploratory analysis
python analyze_spectra.py compare --mode full \
    spectra_LH_*.hdf5 \
    -l "LH_80,LH_100,LH_200,LH_300,LH_400,LH_500" \
    -o plots/parameter_study

# Review outputs
cat plots/parameter_study/statistical_tests.txt
open plots/parameter_study/comparison_enhanced.png
open plots/parameter_study/feature_comparison.png
```

### 4. Redshift Evolution Study
```bash
# Generate at multiple redshifts
for snap in 076 078 080 082 084 086 088 090; do
    python analyze_spectra.py generate \
        data/LH_80/snap_${snap}.hdf5 \
        -n 10000 \
        -o spectra_snap_${snap}.hdf5
done

# Track evolution with detailed analysis
python analyze_spectra.py evolve --mode detailed \
    spectra_snap_*.hdf5 \
    -l "z=3.5,z=3.2,z=2.9,z=2.5,z=2.2,z=1.9,z=1.6,z=1.3" \
    -o plots/thermal_history
```

### 5. CGM-Targeted Study
```bash
# First, analyze halos to identify interesting targets
python analyze_spectra.py halo data/snap_082.hdf5 \
    --mass-range 11.0 12.0 \
    --n-halos 10

# Generate IGM spectra (random sightlines)
python analyze_spectra.py generate data/snap_082.hdf5 -n 10000

# Generate CGM spectra around massive halos
python analyze_spectra.py cgm data/snap_082.hdf5 \
    --mass-range 11.0 12.0 \
    --impact-params 0.5,1.0,1.5 \
    --n-per-bin 100

# Compare IGM vs CGM
python analyze_spectra.py compare --mode detailed \
    camel_lya_spectra_snap_082_n10000.hdf5 \
    cgm_lya_spectra_snap_082_n300.hdf5 \
    -l "IGM,CGM" \
    -o plots/igm_vs_cgm
```

---

## Configuration

Edit `scripts/config.py` to customize:

### Paths
```python
DATA_DIR = Path(__file__).parent / 'data'
PLOTS_DIR = Path(__file__).parent / 'plots'
OUTPUT_DIR = Path(__file__).parent / 'output'
```

### Default Parameters
```python
DEFAULT_SIGHTLINES = 100
DEFAULT_RESOLUTION = 0.1  # km/s
DEFAULT_BOX_SIZE = 25000  # ckpc/h
```

### Spectral Lines
```python
SPECTRAL_LINES = {
    'lya': {'wavelength': 1215.67, 'element': 'H', 'ion': 1},
    'civ': {'wavelength': 1548.19, 'element': 'C', 'ion': 4},
    'ovi': {'wavelength': 1031.93, 'element': 'O', 'ion': 6},
    'mgii': {'wavelength': 2796.35, 'element': 'Mg', 'ion': 2},
    'siiv': {'wavelength': 1393.76, 'element': 'Si', 'ion': 4},
    # ... more lines available
}
```

---

## Code Organization

```
CGM/
├── analyze_spectra.py          # Main CLI entry point (214 lines)
├── batch_process.py            # Batch processing utility (363 lines)
├── downloader.py               # CAMEL data downloader (234 lines)
├── scripts/
│   ├── config.py              # Configuration & spectral line database (408 lines)
│   ├── utils.py               # Analysis functions (1700+ lines)
│   ├── analysis.py            # Core analysis routines
│   ├── plotting.py            # Visualization functions
│   ├── comparison.py          # Memory-efficient comparison (1000+ lines)
│   ├── statistical_tests.py   # Statistical framework (200+ lines)
│   ├── exploratory.py         # Feature extraction & clustering (500+ lines)
│   ├── hdf5_io.py            # HDF5 I/O utilities
│   ├── fake_spectra_fix.py   # Python 3.13 compatibility patches
│   ├── cgm/
│   │   ├── halos.py          # Halo analysis
│   │   ├── targeted_spectra.py  # CGM spectra generation
│   │   └── visualization.py   # CGM-specific plots
│   └── commands/
│       ├── analyze.py        # Analyze command implementation
│       ├── compare_evolve.py # Compare, evolve & diagnose commands
│       ├── generate.py       # Generate command implementation
│       ├── cgm.py           # CGM command implementation
│       ├── halo.py          # Halo command implementation
│       ├── list_explore.py  # List & explore commands
│       └── pipeline.py      # Pipeline command implementation
├── slurm_templates/           # HPC SLURM job templates
│   ├── generate_spectra.sbatch
│   ├── analyze_spectra.sbatch
│   └── batch_pipeline.sbatch
├── plots/                     # Output plots (organized by suite/set/sim)
├── spectra/                   # Generated spectra files
├── output/                    # Other output files
└── data/                      # Simulation data (snapshots & group catalogs)
```

**Total**: ~8,000 lines of Python code

---

## Additional Tools

### Batch Processing
Process multiple snapshots in parallel:

```bash
# Generate spectra for all snapshots in a directory
python batch_process.py generate data/IllustrisTNG/LH/LH_80/ -n 10000 --workers 4

# Analyze all spectra files
python batch_process.py analyze data/IllustrisTNG/LH/LH_80/ --workers 2

# Full pipeline
python batch_process.py pipeline data/IllustrisTNG/LH/LH_80/ -n 10000
```

### Data Downloader
Download CAMEL simulation data:

```bash
# Download snapshot
python downloader.py --suite IllustrisTNG --set LH --sim 80 --snapshot 82

# Download group catalog for existing snapshot
python downloader.py --groups data/IllustrisTNG/LH/LH_80/snap_082.hdf5

# Download with custom output path
python downloader.py --suite IllustrisTNG --set LH --sim 80 --snapshot 82 \
    --output my_data/snap_082.hdf5
```

---

## Documentation

- **README.md**: This file (overview and usage)
- **AGENTS.md**: Code guidelines and conventions for developers
- **Meeting-minutes.md**: Project meeting notes

Additional documentation may be available in the project directory.

---

## Memory Optimization Details

**Problem**: Out of memory when comparing 6 simulations × 10,000 spectra
- Original requirement: ~10 GB RAM
- Failed on HPC login nodes

**Solution**: Lazy loading + chunked processing
- New requirement: ~200 MB RAM
- **98% memory reduction** (50× less)

**Method**:
- Load data on-demand in chunks of 1,000 spectra
- Sample 100,000 pixels for statistical tests (still highly robust)
- Process features incrementally without approximation
- Keep files on disk, not in RAM

**Preserved**:
- ✅ All visualizations (same quality)
- ✅ Statistical significance (p-values still < 1e-70)
- ✅ Feature accuracy (processes all data, just chunked)
- ✅ Backward compatibility

**See**: `MEMORY_OPTIMIZATION.md` for full technical details

---

## Citation

- **CAMEL Project**: Villaescusa-Navarro et al. (2021)
- **fake_spectra**: Bird et al. (2015)

---

## Dependencies

**Required**:
- Python 3.10+ (tested on 3.13)
- numpy
- h5py
- matplotlib
- scipy
- scikit-learn (clustering analysis in full mode)
- fake_spectra (spectra generation - includes Python 3.13 compatibility patches)
