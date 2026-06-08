#!/bin/bash
# ====================================================================
# Download the forest-core snapshots for the Omega0--sigma8 study.
# Adds z~4 (snap 024), z~3 (snap 032), z~2 (snap 044) for all 10
# p1/p2 variants. (z6=014, z5=018, z0.27=080 already present.)
# Snapshot->z verified from group-catalog headers:
#   024 -> z=3.996, 032 -> z=2.999, 044 -> z=2.002
# Run from repo root: bash shell_scripts/download_snapshots.sh
# ====================================================================
set -u

NEW_SNAPS=(24 32 44)                      # downloader takes int; writes snap_0NN.hdf5
SIMS=(p1_n2 p1_n1 p1_0 p1_1 p1_2 p2_n2 p2_n1 p2_0 p2_1 p2_2)

for sim in "${SIMS[@]}"; do
  for s in "${NEW_SNAPS[@]}"; do
    dest=$(printf "data/IllustrisTNG/1P/1P_%s/snap_%03d.hdf5" "$sim" "$s")
    if [ -f "$dest" ]; then
      echo "SKIP (exists): $dest"
      continue
    fi
    echo ">>> 1P_${sim} snap ${s}"
    python downloader.py --suite IllustrisTNG --set 1P --sim "$sim" --snapshot "$s"
  done
done

echo "Done. Verify: find data/IllustrisTNG/1P -name 'snap_024.hdf5' -o -name 'snap_032.hdf5' -o -name 'snap_044.hdf5' | wc -l   # expect 30"
