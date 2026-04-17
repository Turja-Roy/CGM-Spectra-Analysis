# CAMEL Lyman-Alpha Forest Analysis Pipeline

Comprehensive Python toolkit for analyzing the Lyman-alpha forest from cosmological simulations (CAMEL project). Features multi-line spectral analysis, CGM-targeted observations, and memory-efficient comparison tools for large-scale parameter studies.

## Quick Start

### Installation
```bash
# Clone repository
cd /path/to/CGM

# Install dependencies
pip install h5py numpy matplotlib scipy scikit-learn pandas fake_spectra

# For MPI parallelization (optional, for generate command on HPC)
pip install mpi4py
```

### Basic Usage
```bash
# List available data
python analyze_spectra.py list

# Generate IGM spectra
python analyze_spectra.py generate data/snap_080.hdf5 -n 10000

# Analyze spectra
python analyze_spectra.py analyze spectra/snap_080_spectra.hdf5

# Compare parameter variations with overlay plots
python analyze_spectra.py compare 'spectra/1P/1P_p1_*/snap_080_spectra.hdf5' \
    --param Omega_m --fiducial 1P_0 --name omega_scan
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

### `generate-sightlines`
Generate master sightlines for consistent parameter scans
```bash
python analyze_spectra.py generate-sightlines <name> -n 10000 --seed 42
```
**Output**: `output/sightlines/<name>.hdf5` with positions, axes, and metadata

### `generate`
Generate synthetic spectra from simulation snapshot

**Options:**
- `-n, --sightlines`: Number of random sightlines (default: 100)
- `-r, --res`: Velocity resolution in km/s (default: 0.1)
- `--line`: Spectral lines (default: lya)
  - Single: `--line lya`
  - Multiple: `--line lya,civ,ovi`
- `--sightlines-from`: Use pre-generated sightlines from file
- `-o, --output`: Output file path
- `--mpi`: Use MPI parallelization via fake_spectra (requires mpi4py, run with mpirun)

**Examples:**
```bash
# Basic IGM spectra
python analyze_spectra.py generate data/snap_080.hdf5 -n 10000

# Use consistent sightlines across simulations
python analyze_spectra.py generate 'data/1P/1P_p1_*/snap_080.hdf5' \
  --sightlines-from output/sightlines/snap80_omega.hdf5 --line lya

# Multi-line analysis
python analyze_spectra.py generate data/snap_080.hdf5 -n 10000 --line lya,civ,ovi

# MPI parallel generation (on HPC)
mpirun -n 48 python analyze_spectra.py generate data/snap_080.hdf5 -n 10000 --mpi
```

### `analyze`
Comprehensive analysis of spectra file

**Performs:**
- Flux statistics and sample spectra
- Effective optical depth τ_eff
- Flux power spectrum P_F(k)
- Column density distribution f(N_HI) using accurate pre-computed values
- Line width distribution b(N_HI) and temperatures (if T/ρ available)
- Temperature-density relation T(ρ) (if available)
- Metal line statistics (if multi-line data)

**Options:**
- `--line`: Spectral line to analyze (auto-detect if not specified)
- `--cd-method`: Column density method (simple or vpfit)
- `--max-sightlines`: Maximum sightlines to analyze (for memory-limited systems)
- `--workers`: Number of parallel workers (default: 1, use 4-8 for parallel)

**Example:**
```bash
# Sequential analysis
python analyze_spectra.py analyze spectra_file.hdf5

# Parallel analysis with 8 workers (for large datasets)
python analyze_spectra.py analyze spectra_file.hdf5 --workers 8
```

**Output**: 
- Plots in `plots/<suite>/<sim_set>/<sim_name>/snap-<N>/`
- Data export in `output/analysis/<suite>/<sim_set>/<sim_name>/snap-<N>/`
  - `analysis_results.json` (full results)
  - CSV files: `power_spectrum.csv`, `cddf.csv`, `flux_stats.csv`, etc.

### `compare`
Compare multiple simulations with overlay plots and fiducial ratios

**Options:**
- `<pattern>`: Glob pattern for spectra files (use quotes!)
- `--param`: Parameter name for auto-labeling (e.g., Omega_m, sigma_8)
- `--fiducial`: Reference simulation name for ratio plots
- `--name`: Output name for comparison plots
- `--precomputed`: Use pre-computed analysis data (faster)

**Example:**
```bash
# Compare Ωₘ variations with auto-labeling
python analyze_spectra.py compare 'spectra/1P/1P_p1_*/snap_080_spectra.hdf5' \
  --param Omega_m --fiducial 1P_0 --name omega_scan_z0
```

**Output**: `plots/comparison/<name>/`
- `power_spectrum_overlay.png` - P(k) curves with ratio panel
- `cddf_overlay.png` - f(N_HI) distributions
- `flux_stats_comparison.png` - Bar chart comparison
- `tau_eff_comparison.png` - Effective optical depths
- `sample_spectra_comparison.png` - Side-by-side spectra

**Features**:
- Auto-labeling from CAMEL parameter CSV files
- Fiducial ratios show deviations from reference
- Loads pre-computed data from `output/analysis/` (if available)
- Falls back to on-the-fly computation if needed

### `evolve`
Track redshift evolution of observables across multiple snapshots

**Example:**
```bash
python analyze_spectra.py evolve \
    spectra_snap_076.hdf5 spectra_snap_080.hdf5 spectra_snap_084.hdf5 \
    -l "z=3.5,z=2.5,z=1.5" -o plots/evolution
```

### `diagnose`
Deep diagnostic analysis of a single spectra file

**Example:**
```bash
python analyze_spectra.py diagnose spectra.hdf5 --output plots/diagnostics
```

### `pipeline`
Full pipeline: generate + analyze in one command

**Example:**
```bash
python analyze_spectra.py pipeline data/snap_080.hdf5 -n 10000
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
├── colden/                   # Accurate column densities
│   ├── H/1                   # HI column density [cm^-2]
│   ├── C/4                   # CIV (if requested)
│   └── O/6                   # OVI (if requested)
├── temperature/H/1/          # Temperature field
├── density_weight_density/H/1/  # Density field
└── Sightlines/               # Sightline metadata
    ├── positions             # 3D positions [ckpc/h]
    ├── axes                  # Axis indices (1=x, 2=y, 3=z)
    └── seed                  # Random seed used
```

### Plot Outputs

**Structure**: Plots are automatically organized by simulation structure:
```
plots/
├── <suite>/
│   └── <sim_set>/
│       └── <sim_name>/
│           └── snap-XXX/
│               ├── sample_spectra_*.png
│               ├── flux_statistics_*.png
│               ├── flux_power_spectrum_*.png
│               ├── column_density_distribution_*.png
│               ├── line_width_distribution_*.png
│               ├── temperature_density_relation_*.png
│               └── multi_line_comparison_*.png
└── comparison/
    └── <name>/
        ├── power_spectrum_overlay.png
        ├── cddf_overlay.png
        ├── flux_stats_comparison.png
        ├── tau_eff_comparison.png
        └── sample_spectra_comparison.png

output/
├── sightlines/
│   └── <name>.hdf5                # Master sightlines
└── analysis/
    └── <suite>/<sim_set>/<sim_name>/snap-XXX/
        ├── analysis_results.json   # Full results
        ├── power_spectrum.csv
        ├── cddf.csv
        ├── flux_stats.csv
        ├── line_widths.csv
        ├── temp_density.csv
        └── metal_lines.csv
```

---

## Common Workflows

### 1. Single Simulation Analysis
```bash
# Generate and analyze
python analyze_spectra.py generate data/snap_080.hdf5 -n 10000 --line lya
python analyze_spectra.py analyze spectra/snap_080_spectra.hdf5
```

### 2. Parameter Scan with Identical Sightlines
```bash
# Step 1: Generate master sightlines
python analyze_spectra.py generate-sightlines snap80_omega -n 10000 --seed 42

# Step 2: Generate spectra for all Ωₘ variations (using same sightlines)
python analyze_spectra.py generate 'data/1P/1P_p1_*/snap_080.hdf5' \
  --sightlines-from output/sightlines/snap80_omega.hdf5 --line lya

# Step 3: Analyze all simulations
python analyze_spectra.py analyze 'spectra/1P/1P_p1_*/snap_080_spectra.hdf5'

# Step 4: Compare with overlay plots and fiducial ratios
python analyze_spectra.py compare 'spectra/1P/1P_p1_*/snap_080_spectra.hdf5' \
  --param Omega_m --fiducial 1P_0 --name omega_scan_z0
```

### 3. Multi-Line CGM Study
```bash
# Generate spectra with multiple ions
python analyze_spectra.py generate data/snap_080.hdf5 \
    -n 10000 --line lya,civ,ovi

# Full analysis (includes metal line statistics)
python analyze_spectra.py analyze spectra/snap_080_spectra.hdf5
```

### 4. CGM-Targeted Study
```bash
# Generate IGM spectra (random sightlines)
python analyze_spectra.py generate data/snap_082.hdf5 -n 10000

# Generate CGM spectra around massive halos
python analyze_spectra.py cgm data/snap_082.hdf5 \
    --mass-range 11.0 12.0 \
    --impact-params 0.5,1.0,1.5 \
    --n-per-bin 100

# Compare IGM vs CGM with overlay plots
python analyze_spectra.py compare \
    'spectra/snap_082_spectra.hdf5' \
    'spectra/cgm/cgm_snap_082_spectra.hdf5' \
    --name igm_vs_cgm
```

---

## Configuration

Edit `scripts/config.py` to customize:

### Paths
```python
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "plots"
OUTPUT_DIR = PROJECT_ROOT / "output"
SPECTRA_DIR = PROJECT_ROOT / "spectra"
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
    'lya':  ('H',  1, 1215, 'Lyman-alpha'),
    'lyb':  ('H',  1, 1025, 'Lyman-beta'),
    'heii': ('He', 2, 303,  'HeII-303'),
    'civ':  ('C',  4, 1548, 'CIV-1548'),
    'ovi':  ('O',  6, 1031, 'OVI-1031'),
    'mgii': ('Mg', 2, 2796, 'MgII-2796'),
    'siiv': ('Si', 4, 1393, 'SiIV-1393'),
}
```

## Build (C++ Extensions)

Optional: Build pybind11 C++ extensions for accelerated analysis:
```bash
./compile.sh
```
Requires HPC modules: cmake, gcc, impi, eigen, fftw3

---

## Code Organization

```
CGM/
├── analyze_spectra.py          # Main CLI entry point
├── batch_process.py            # Batch processing utility
├── downloader.py               # CAMEL data downloader
├── scripts/
│   ├── config.py              # Configuration & spectral line database
│   ├── utils.py               # Core analysis functions
│   ├── analysis.py            # Analysis routines (uses accurate colden)
│   ├── plotting.py            # Visualization (including overlay plots)
│   ├── sightline_manager.py  # Sightline generation & management
│   ├── label_generator.py    # Auto-labeling from CAMEL parameters
│   ├── data_export.py         # Export results (JSON + CSV)
│   ├── comparison.py          # Statistical comparison framework
│   ├── hdf5_io.py            # HDF5 I/O utilities
│   ├── fake_spectra_fix.py   # Patches for fake_spectra bugs (auto-imported)
│   ├── cgm/
│   │   ├── halos.py          # Halo analysis
│   │   ├── targeted_spectra.py  # CGM spectra generation
│   │   └── visualization.py   # CGM-specific plots
│   └── commands/
│       ├── generate_sightlines.py  # Sightline generation command
│       ├── analyze.py        # Analyze command
│       ├── compare.py        # Compare command (overlay plots)
│       ├── generate.py       # Generate command (with colden)
│       ├── cgm.py           # CGM command
│       ├── halo.py          # Halo command
│       └── pipeline.py      # Pipeline command
├── plots/                     # Output plots
├── spectra/                   # Generated spectra files
├── output/                    # Analysis data & sightlines
└── data/                      # Simulation snapshots
```

---

## Additional Tools

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

## Key Features

### Identical Sightlines for Parameter Scans
- Generate master sightlines once, reuse for all simulations
- Ensures fair comparison across parameter variations
- Eliminates sample variance

### Accurate Column Densities
- Uses fake_spectra's pre-computed `colden` datasets
- More accurate than tau-based estimation
- Automatically saved during generation and used in analysis

### Auto-Labeling
- Reads CAMEL parameter CSV files
- Generates formatted labels (e.g., "Ωₘ = 0.3")
- Detects which parameter varies across simulations

### Overlay Plots with Fiducial Ratios
- Compare multiple simulations on single plots
- Optional ratio panels show deviations from reference
- Professional styling with color cycling

### Data Export
- JSON: Full analysis results with metadata
- CSV: Individual data tables for each observable
- Organized by suite/sim_set/sim_name/snapshot hierarchy

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
- scikit-learn
- fake_spectra (spectra generation)

**Optional**:
- mpi4py (MPI parallelization)
