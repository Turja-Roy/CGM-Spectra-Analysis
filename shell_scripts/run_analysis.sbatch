#!/bin/bash
#SBATCH -J camel_analysis          # Job name
#SBATCH -o logs/job_%A_%a.out      # Output file (%A=job ID, %a=array task ID)
#SBATCH -e logs/job_%A_%a.err      # Error file
#SBATCH -p normal                  # Partition/queue (normal for Lonestar6)
#SBATCH -N 1                       # Number of nodes
#SBATCH -n 1                       # Number of tasks (cores)
#SBATCH -t 02:30:00                # Time limit (4 hours)
#SBATCH --mem=8                    # Memory per node
#SBATCH --array=1-25               # Array job with 25 tasks (one per simulation)
#SBATCH --mail-type=END,FAIL       # Email notifications (optional)
#SBATCH --mail-user=$turja.roy@uta.edu  # Email address (update if needed)

# ====================================================================
# SLURM Array Job for CAMEL Spectra Generation and Analysis
# ====================================================================
# This script processes 25 simulations in parallel as array tasks.
# Each task reads its simulation name from simulations.txt and runs:
#   1. python analyze_spectra.py generate (create spectra)
#   2. python analyze_spectra.py analyze (analyze spectra)
#
# Usage: sbatch run_analysis.sbatch
# Monitor: squeue -u $USER
# Cancel: scancel <JOBID>
# ====================================================================

# Print job info
echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
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
SNAPSHOT_PATH="data/IllustrisTNG/1P/${SIMULATION}/snap_080.hdf5"
SIGHTLINES_FILE="output/sightlines/analysis_sightlines_2025-01-28.hdf5"
SPECTRA_PATH="spectra/IllustrisTNG/1P/${SIMULATION}/camel_lya_spectra_snap_080_n10000.hdf5"

# Check if snapshot exists
if [ ! -f "$SNAPSHOT_PATH" ]; then
    echo "ERROR: Snapshot file not found: $SNAPSHOT_PATH"
    exit 1
fi

# Check if sightlines file exists
if [ ! -f "$SIGHTLINES_FILE" ]; then
    echo "ERROR: Sightlines file not found: $SIGHTLINES_FILE"
    exit 1
fi

# Activate virtual environment
echo "[1/2] Activating virtual environment..."
source .venv/bin/activate

# Check activation
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

echo "Virtual environment activated"
echo ""

# Step 1: Generate spectra
echo "[2/2] Generating spectra..."
echo "Command: python analyze_spectra.py generate '$SNAPSHOT_PATH' --sightlines-from '$SIGHTLINES_FILE'"
echo ""

python analyze_spectra.py generate "$SNAPSHOT_PATH" \
    --sightlines-from "$SIGHTLINES_FILE"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Spectra generation failed for $SIMULATION"
    exit 1
fi

echo ""
echo "Spectra generation completed successfully"
echo ""

# Step 2: Analyze spectra
echo "[3/2] Analyzing spectra..."
echo "Command: python analyze_spectra.py analyze '$SPECTRA_PATH'"
echo ""

python analyze_spectra.py analyze "$SPECTRA_PATH"

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
