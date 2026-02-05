#!/bin/bash
# ====================================================================
# Submit All CAMEL Analysis Jobs to SLURM
# ====================================================================
# This script submits the SLURM array job that processes all 25
# simulations in parallel on compute nodes.
#
# Usage: ./submit_all.sh
# ====================================================================

echo "========================================"
echo "CAMEL Spectra Analysis - Job Submission"
echo "========================================"
echo ""

# Create logs directory if it doesn't exist
if [ ! -d "logs" ]; then
    echo "Creating logs directory..."
    mkdir -p logs
    echo "Logs directory created"
else
    echo "Logs directory already exists"
fi
echo ""

# Check if simulations.txt exists
if [ ! -f "simulations.txt" ]; then
    echo "ERROR: simulations.txt not found!"
    echo "This file should list all 25 simulation names (one per line)"
    exit 1
fi

# Count simulations
NUM_SIMS=$(wc -l < simulations.txt)
echo "Found $NUM_SIMS simulations in simulations.txt"
echo ""

# Check if run_analysis.sbatch exists
if [ ! -f "run_analysis.sbatch" ]; then
    echo "ERROR: run_analysis.sbatch not found!"
    exit 1
fi

# Submit the array job
echo "Submitting SLURM array job..."
echo "Command: sbatch run_analysis.sbatch"
echo ""

JOBID=$(sbatch run_analysis.sbatch 2>&1)

if [ $? -eq 0 ]; then
    echo "SUCCESS!"
    echo "$JOBID"
    
    # Extract job ID number
    JOB_NUM=$(echo "$JOBID" | grep -oP '\d+')
    
    echo ""
    echo "========================================"
    echo "Monitoring Commands"
    echo "========================================"
    echo "Check job status:     squeue -u \$USER"
    echo "Check specific job:   squeue -j $JOB_NUM"
    echo "Cancel all tasks:     scancel $JOB_NUM"
    echo "Cancel one task:      scancel ${JOB_NUM}_<TASKID>"
    echo ""
    echo "Logs will be saved to:"
    echo "  logs/job_${JOB_NUM}_<TASKID>.out"
    echo "  logs/job_${JOB_NUM}_<TASKID>.err"
    echo "========================================"
else
    echo "ERROR: Job submission failed!"
    echo "$JOBID"
    exit 1
fi

exit 0
