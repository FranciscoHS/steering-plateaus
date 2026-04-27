#!/usr/bin/env python3
"""Serial download of Gemma-2-9B to avoid xet parallel-staging disk spikes."""
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
from huggingface_hub import snapshot_download
path = snapshot_download(
    'google/gemma-2-9b',
    max_workers=1,
    cache_dir=os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')) + '/hub',
)
print('done:', path)
