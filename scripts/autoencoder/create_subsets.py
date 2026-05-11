#!/usr/bin/env python3
"""Create subset memmap caches from a large memmap activation cache.

Each subset gets its own memmap file + .meta.json sidecar. Copy is done
in chunks so RAM stays tiny.
"""

import argparse
import json
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='Path to the full cache (memmap format)')
    parser.add_argument('--sizes', nargs='+', type=int, required=True,
                        help='Subset sizes to create (e.g. 4000000 6000000 8000000)')
    parser.add_argument('--output_dir', required=True, help='Directory for subset caches')
    parser.add_argument('--chunk', type=int, default=100000,
                        help='Rows per chunk when copying (keeps RAM tiny)')
    args = parser.parse_args()

    meta_path = args.source + '.meta.json'
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta sidecar not found: {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    src_shape = tuple(meta['shape'])
    np_dtype = np.dtype(meta['dtype'])
    n_total, d_model = src_shape

    print(f"Source: {args.source} shape={src_shape} dtype={np_dtype}")
    src = np.memmap(args.source, dtype=np_dtype, mode='r', shape=src_shape)

    for size in args.sizes:
        if size > n_total:
            print(f"  SKIP {size}: larger than source ({n_total})")
            continue

        size_label = f"{size // 1000000}M"
        out_dir = os.path.join(args.output_dir, f"l10_cache_{size_label}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "cached_activations.pt")

        dst_shape = (size, d_model)
        nbytes = size * d_model * np.dtype(np_dtype).itemsize
        print(f"  Creating {size_label} subset ({nbytes / 1e9:.1f} GB) -> {out_path}")
        dst = np.memmap(out_path, dtype=np_dtype, mode='w+', shape=dst_shape)

        for i in range(0, size, args.chunk):
            j = min(i + args.chunk, size)
            dst[i:j] = src[i:j]
        dst.flush()
        del dst

        with open(out_path + '.meta.json', 'w') as f:
            json.dump({'shape': list(dst_shape),
                       'dtype': str(np_dtype),
                       'format': 'memmap_raw'}, f)

    print("Done.")

if __name__ == '__main__':
    main()
