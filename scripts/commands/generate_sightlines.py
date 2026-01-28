"""
Command to generate and save sightlines for consistent cross-simulation comparison.
"""

import os
import numpy as np
from pathlib import Path

import scripts.config as config
from scripts.sightline_manager import (
    generate_random_sightlines, 
    save_sightlines_hdf5,
    get_sightline_summary
)


def cmd_generate_sightlines(args):
    """Generate and save sightlines to HDF5 file."""
    name = args.name
    n_sightlines = args.sightlines
    seed = args.seed
    box_size = args.box_size
    
    print("=" * 70)
    print("GENERATING SIGHTLINES")
    print("=" * 70)
    print(f"Name:         {name}")
    print(f"Sightlines:   {n_sightlines:,}")
    print(f"Random seed:  {seed}")
    print(f"Box size:     {box_size:.1f} comoving kpc/h")
    print(f"Distribution: uniform")
    print(f"Axes mode:    random")
    
    # Create output directory
    output_dir = config.OUTPUT_DIR / 'sightlines'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output path
    output_path = output_dir / f"{name}.hdf5"
    
    # Check if file already exists
    if output_path.exists():
        print(f"\nWarning: File already exists: {output_path}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("Aborted.")
            return 1
    
    # Generate sightlines
    print("\nGenerating random sightlines...")
    sightlines = generate_random_sightlines(
        n_sightlines=n_sightlines,
        box_size=box_size,
        seed=seed,
        axes_mode='random'
    )
    
    # Prepare metadata
    metadata = {
        'seed': seed,
        'box_size': box_size,
        'distribution': 'uniform',
        'axes_mode': 'random',
        'n_sightlines': n_sightlines
    }
    
    # Save to HDF5
    print(f"\nSaving to: {output_path}")
    save_sightlines_hdf5(sightlines, output_path, metadata=metadata)
    
    # Print summary
    print("\n" + "-" * 70)
    summary = get_sightline_summary({'positions': sightlines['positions'], 
                                      'axes': sightlines['axes'],
                                      'metadata': metadata})
    print(summary)
    print("-" * 70)
    
    print(f"\n{'=' * 70}")
    print("Sightlines saved successfully!")
    print(f"Location: {output_path}")
    print(f"\nUsage example:")
    print(f"  python analyze_spectra.py generate data/snap_080.hdf5 \\")
    print(f"    --sightlines-from {output_path} --line lya")
    print(f"{'=' * 70}")
    
    return 0
