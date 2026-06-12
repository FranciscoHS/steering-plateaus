#!/usr/bin/env python3
"""Print SSH host:port for every running RunPod pod (from the CLAUDE.md
troubleshooting recipe). Useful when rpc create raced and the SSH config
alias points at the wrong pod."""
import os

import runpod

with open(os.path.expanduser('~/.config/runpod_cli/.env')) as f:
    for line in f:
        if line.startswith('RUNPOD_API_KEY='):
            runpod.api_key = line.strip().split('=', 1)[1]

for pod in runpod.get_pods():
    rt = pod.get('runtime') or {}
    ssh = [f"{p['ip']}:{p['publicPort']}"
           for p in (rt.get('ports') or []) if p['privatePort'] == 22]
    gpus = pod.get('machine', {}).get('gpuDisplayName', '?')
    print(f"{pod['name']:<12} {pod['id']}  {gpus:<22} "
          f"ssh={ssh[0] if ssh else 'NONE'}")
