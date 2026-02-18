#!/bin/bash
# Gemma 3 scaling series: 270M → 1B → 4B
# Perturb at layer 2, measure at 2nd-to-last layer, cov-matched
set -e
cd /workspace/steering-plateaus

echo "=========================================="
echo "Gemma 3 270M (18 layers, measure at layer 16)"
echo "=========================================="
python -u scripts/run_perturbation_sweep_spherical.py \
    --model gemma-3-270m --layer 2 --measure_layer -2 \
    --cov_matched --n_fit 3000

echo ""
echo "=========================================="
echo "Gemma 3 1B (26 layers, measure at layer 24)"
echo "=========================================="
python -u scripts/run_perturbation_sweep_spherical.py \
    --model gemma-3-1b --layer 2 --measure_layer -2 \
    --cov_matched --n_fit 3000

echo ""
echo "=========================================="
echo "Gemma 3 4B (34 layers, measure at layer 32)"
echo "=========================================="
python -u scripts/run_perturbation_sweep_spherical.py \
    --model gemma-3-4b --layer 2 --measure_layer -2 \
    --cov_matched --n_fit 3000

echo ""
echo "All done!"
