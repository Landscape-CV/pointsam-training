"""Build tiles and crops for every strip in a data config. No GPU needed."""
import json
import os
import sys
import time

import yaml

sys.path.insert(0, "src")

from treeseg.data.prepare import load_strip, voxelize, write_crops, write_tiles


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/data/isprs_wytham.yaml"
    cfg = yaml.safe_load(open(config_path))
    summary = {"config": cfg, "strips": {}}

    for split, filename in cfg["strips"].items():
        start = time.time()
        xyz, ids = load_strip(os.path.join(cfg["source_dir"], filename))
        vox_xyz, vox_ids = voxelize(xyz, ids, cfg["voxel_m"])
        n_raw = len(xyz)
        del xyz, ids

        tile_dir = os.path.join(cfg["out_dir"], "tiles")
        crop_dir = os.path.join(cfg["out_dir"], "crops", split)
        tiles = write_tiles(vox_xyz, vox_ids, tile_dir, f"isprs_{split}", cfg["tile_m"])
        crops = write_crops(vox_xyz, vox_ids, crop_dir, f"isprs_{split}", cfg["crop_m"], cfg["stride_m"],
                            cfg["min_crop_points"], cfg["max_crop_points"], cfg["min_crop_trees"], cfg["seed"])

        tree_frac = float((vox_ids > 0).mean())
        summary["strips"][split] = {"raw_points": n_raw, "voxels": len(vox_xyz), "tree_fraction": round(tree_frac, 3),
                                    "tiles": tiles, "crops": crops}
        print(f"{split}: {n_raw:,} points -> {len(vox_xyz):,} voxels ({tree_frac:.0%} tree), "
              f"{len(tiles)} tiles, {len(crops)} crops, {time.time() - start:.0f}s", flush=True)

    with open(os.path.join(cfg["out_dir"], "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
