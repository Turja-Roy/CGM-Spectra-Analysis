#!/bin/bash
#SBATCH -J camel_analysis          # Job name
#SBATCH -o logs/job_%A_%a.out      # Output file (%A=job ID, %a=array task ID)
#SBATCH -e logs/job_%A_%a.err      # Error file
#SBATCH -p normal                  # Partition/queue (normal for Lonestar6)
#SBATCH -N 1                       # Number of nodes
#SBATCH -n 48                      # Number of tasks (cores) - for MPI or workers
#SBATCH -t 48:00:00                # Time limit (48 hours for high-z)
#SBATCH --mem=64                   # Memory per node (GB)
#SBATCH --array=1-25               # Array job with 25 tasks (one per simulation)
#SBATCH --mail-type=END,FAIL       # Email notifications (optional)
#SBATCH --mail-user=$turja.roy@uta.edu  # Email address (update if needed)

# ====================================================================
# SLURM Array Job for CAMEL Spectra Generation and Analysis
# ====================================================================
# This script processes 25 simulations in parallel as array tasks.
# Supports both MPI (generate) and multiprocessing (analyze).
#
# Usage:
#   With MPI for generate:    sbatch --constraint=haswell run_analysis.sbatch
#   Sequential generate:       sbatch run_analysis.sbatch
#
# For generate: Uses fake_spectra MPI when --mpi flag is passed
# For analyze: Uses --workers flag for parallel analysis
# ====================================================================

# Configuration
USE_MPI=true           # Set to true for MPI parallelized generation
NUM_WORKERS=12           # Number of parallel workers for analyze
N_SIGHTLINES=10000      # Number of sightlines to generate

# Snapshot configuration
SNAP_NUM=014
SIGHTLINES_FILE="output/sightlines/analysis_sightlines_2025-01-28.hdf5"  # Set to "" to generate random sightlines

# Print job info
echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Cores: $SLURM_NTASKS"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo ""
echo "MPI enabled: $USE_MPI"
echo "Analyze workers: $NUM_WORKERS"
echo "Snapshot: snap_$SNAP_NUM"
echo "Sightlines: ${SIGHTLINES_FILE:-random}"
echo ""

# Load modules (uncomment if needed on Lonestar6)
# module load python3
# module load hdf5

# Get simulation name from simulations.txt based on array task ID
SIMULATION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" simulations.txt)

if [ -z "$SIMULATION" ]; then
    echo "ERROR: Could not read simulation name for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Processing simulation: $SIMULATION"
echo "========================================"
echo ""

# Define paths
SNAPSHOT_PATH="data/IllustrisTNG/1P/${SIMULATION}/snap_${SNAP_NUM}.hdf5"

# Handle sightlines file
if [ -n "$SIGHTLINES_FILE" ] && [ -f "$SIGHTLINES_FILE" ]; then
    SIGHTLINES_ARG="--sightlines-from $SIGHTLINES_FILE"
else
    SIGHTLINES_ARG=""
    if [ -n "$SIGHTLINES_FILE" ]; then
        echo "Warning: Sightlines file not found: $SIGHTLINES_FILE"
        echo "Generating random sightlines instead"
    fi
fi

SPECTRA_PATH="spectra/IllustrisTNG/1P/${SIMULATION}/camel_lya_spectra_snap_${SNAP_NUM}_n${N_SIGHTLINES}.hdf5"

# Check if snapshot exists
if [ ! -f "$SNAPSHOT_PATH" ]; then
    echo "ERROR: Snapshot file not found: $SNAPSHOT_PATH"
    exit 1
fi

# Activate virtual environment
echo "[1/2] Activating virtual environment..."
source .venv/bin/activate

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

echo "Virtual environment activated"
echo ""

# Step 1: Generate spectra
echo "[2/2] Generating spectra..."

if [ "$USE_MPI" = true ]; then
    # MPI parallelized generation (use with mpirun)
    # Note: On SLURM, use srun instead of mpirun for better integration
    echo "Using MPI with $SLURM_NTASKS cores"
    echo "Command: srun -n $SLURM_NTASKS python analyze_spectra.py generate '$SNAPSHOT_PATH' $SIGHTLINES_ARG --mpi -n $N_SIGHTLINES"
    
    srun -n $SLURM_NTASKS python analyze_spectra.py generate "$SNAPSHOT_PATH" \
        $SIGHTLINES_ARG \
        --mpi \
        -n $N_SIGHTLINES
else
    # Sequential generation
    echo "Command: python analyze_spectra.py generate '$SNAPSHOT_PATH' $SIGHTLINES_ARG -n $N_SIGHTLINES"
    python analyze_spectra.py generate "$SNAPSHOT_PATH" \
        $SIGHTLINES_ARG \
        -n $N_SIGHTLINES
fi

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Spectra generation failed for $SIMULATION"
    exit 1
fi

echo ""
echo "Spectra generation completed successfully"
echo ""

# Step 2: Analyze spectra with parallel workers
echo "[3/2] Analyzing spectra with $NUM_WORKERS workers..."
echo "Command: python analyze_spectra.py analyze '$SPECTRA_PATH' --workers $NUM_WORKERS"
echo ""

python analyze_spectra.py analyze "$SPECTRA_PATH" --workers $NUM_WORKERS

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Spectra analysis failed for $SIMULATION"
    exit 1
fi

echo ""
echo "========================================"
echo "Simulation $SIMULATION completed successfully!"
echo "Finished at: $(date)"
echo "========================================"

exit 0
