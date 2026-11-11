"""Add the ground-removal record to mask files saved before save_masks stored it, so they can be replayed elsewhere.
Run on the machine that made them, where ground removal gives the same points:
    python scripts/add_keep_to_masks.py configs/experiment/diagnose_val.yaml"""
import glob
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, "src")
from treeseg.engine.benchmark import prepare_tile

cfg = yaml.safe_load(open(sys.argv[1]))
for name in cfg["tiles"]:
    tile = prepare_tile(cfg["tile_dir"], name, cfg["csf"], cfg["min_tree_points"])
    for path in sorted(glob.glob(os.path.join(cfg["out_dir"], f"*_{name}_*_masks.npz"))):
        saved = dict(np.load(path))
        highest = int(saved["points"].max()) if len(saved["points"]) else -1
        if highest >= tile["keep"].sum():
            raise SystemExit(f"{path}: masks reach point {highest} but ground removal here kept {tile['keep'].sum()}")
        np.savez_compressed(path, **{**saved, "keep": np.packbits(tile["keep"]), "tile_points": len(tile["keep"])})
        print("added keep to", path, flush=True)
    np.save(os.path.join(cfg["out_dir"], f"{name}_keep.npy"), tile["keep"])
