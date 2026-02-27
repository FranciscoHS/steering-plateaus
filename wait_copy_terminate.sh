#!/bin/bash
# Wait for v3b optimization to finish on RunPod, copy results, terminate pod
set -e

echo "$(date): Waiting for optimization to finish..."

while true; do
    # Check if the python process is still running
    RUNNING=$(ssh runpod "pgrep -f run_delta_optimization_v3 | wc -l" 2>/dev/null || echo "0")
    if [ "$RUNNING" = "0" ]; then
        echo "$(date): Optimization finished!"
        break
    fi
    echo "$(date): Still running..."
    sleep 120
done

# Copy results
echo "$(date): Copying results..."
mkdir -p /home/francisco/Research/steering-plateaus/results/delta_opt_v3b
scp runpod:/workspace/steering-plateaus/results/delta_opt_v3b/eval_r*.png /home/francisco/Research/steering-plateaus/results/delta_opt_v3b/ 2>/dev/null || true
scp runpod:/workspace/steering-plateaus/results/delta_opt_v3b/results_r*.pkl /home/francisco/Research/steering-plateaus/results/delta_opt_v3b/ 2>/dev/null || true
scp runpod:/workspace/steering-plateaus/results/delta_opt_v3b/run.log /home/francisco/Research/steering-plateaus/results/delta_opt_v3b/ 2>/dev/null || true

echo "$(date): Results copied. Files:"
ls -la /home/francisco/Research/steering-plateaus/results/delta_opt_v3b/

# Terminate pod
echo "$(date): Terminating pod..."
rpc terminate

echo "$(date): All done!"
