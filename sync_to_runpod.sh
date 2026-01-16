#!/bin/bash
# Sync code to RunPod (excludes results/)
#
# Usage: ./sync_to_runpod.sh [pod_name]
#
# If pod_name not provided, uses the first active pod.

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_DIR="/workspace/steering-plateaus"

# Get pod name
if [ -n "$1" ]; then
    POD_NAME="$1"
else
    # Get first active pod from rpc list
    POD_NAME=$(rpc list 2>/dev/null | grep -oP 'pod-\w+' | head -1)
    if [ -z "$POD_NAME" ]; then
        echo "No active pods found. Create one with: rpc create"
        exit 1
    fi
fi

echo "Syncing to pod: $POD_NAME"
echo "Local:  $PROJECT_DIR"
echo "Remote: $REMOTE_DIR"
echo ""

rsync -avz --progress \
    --exclude 'results/' \
    --exclude '*.pyc' \
    --exclude '__pycache__' \
    --exclude '.ipynb_checkpoints' \
    --exclude '*.egg-info' \
    --exclude '.git' \
    "$PROJECT_DIR/" \
    "$POD_NAME:$REMOTE_DIR/"

echo ""
echo "Done! SSH with: ssh $POD_NAME"
