#!/usr/bin/env bash
# Reproduce every number and figure (about 1 min on a laptop). Run from the repository root.
set -e
export PYTHONPATH=code
mkdir -p results figures
for s in s01_prepare s02_match s03_discrepancy s04_stage_planes; do python3 code/$s.py; done
python3 code/s05_clusters.py hdb
python3 code/s05_clusters.py bgmm
for s in s06_stress s07_overlap s08_facts make_figures; do python3 code/$s.py; done
