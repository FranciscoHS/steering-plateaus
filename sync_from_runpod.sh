#!/bin/bash
# Sync results from RunPod to local
#
# Usage: ./sync_from_runpod.sh [pod_name]
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
        echo "No active pods found."
        exit 1
    fi
fi

echo "Syncing from pod: $POD_NAME"
echo "Remote: $REMOTE_DIR/results/"
echo "Local:  $PROJECT_DIR/results/"
echo ""

rsync -avz --progress \
    "$POD_NAME:$REMOTE_DIR/results/" \
    "$PROJECT_DIR/results/"

echo ""
echo "Done!"
