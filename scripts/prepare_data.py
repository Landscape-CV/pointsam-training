"""Build tiles and crops for every strip in a data config. No GPU needed.

Tiles keep every point, because the benchmark runs the cloth filter on them itself. Crops have the ground
removed by the same filter, so training and validation see what the model sees in real use.

  python scripts/prepare_data.py [config] [--crops-only]
"""
import json
import os
import shutil
import sys
import time

import yaml

sys.path.insert(0, "src")

from treeseg.data.prepare import ground_by_tile, load_strip, voxelize, write_crops, write_tiles


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    crops_only = "--crops-only" in sys.argv
    config_path = args[0] if args else "configs/data/isprs_wytham.yaml"
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
        tiles = [] if crops_only else write_tiles(vox_xyz, vox_ids, tile_dir, f"isprs_{split}", cfg["tile_m"])

        ground = ground_by_tile(vox_xyz, cfg["tile_m"], cfg["csf"])
        keep_xyz, keep_ids = vox_xyz[~ground], vox_ids[~ground]
        shutil.rmtree(crop_dir, ignore_errors=True)  # a rebuild can make fewer crops, so old ones must not linger
        crops = write_crops(keep_xyz, keep_ids, crop_dir, f"isprs_{split}", cfg["crop_m"], cfg["stride_m"],
                            cfg["min_crop_points"], cfg["max_crop_points"], cfg["min_crop_trees"], cfg["seed"])

        tree_frac = float((keep_ids > 0).mean())
        summary["strips"][split] = {"raw_points": n_raw, "voxels": len(vox_xyz), "removed_as_ground": float(ground.mean()),
                                    "kept_tree_fraction": round(tree_frac, 3), "tree_points_lost": float(ground[vox_ids > 0].mean()),
                                    "crops": crops}
        print(f"{split}: {len(vox_xyz):,} voxels, {ground.mean():.1%} removed as ground, "
              f"{ground[vox_ids > 0].mean():.2%} of tree points lost, {tree_frac:.0%} of the rest is tree, "
              f"{len(tiles)} tiles written, {len(crops)} crops, {time.time() - start:.0f}s", flush=True)

    with open(os.path.join(cfg["out_dir"], "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
