"""Check CSF ground removal on tiles from the command line. notebooks/01_auto_prompt.ipynb has the same step.

Usage: python scripts/inspect_ground.py <tile_dir> <out_dir> <tile> [<tile> ...]
"""
import json
import os
import sys

sys.path.insert(0, "src")

from treeseg.data.ground import classify, csf_ground_mask, load_tile, plot_slice, summary, write_for_viewers


def main():
    tile_dir, out_dir = sys.argv[1], sys.argv[2]
    for tile in sys.argv[3:]:
        xyz, ids = load_tile(tile_dir, tile)
        classes = classify(csf_ground_mask(xyz), ids)
        prefix = os.path.join(out_dir, "csf_" + tile)
        write_for_viewers(prefix, xyz, classes)
        plot_slice(xyz, classes, tile, path=prefix + "_slice.png")
        print(tile, json.dumps(summary(xyz, ids, classes), indent=2))


if __name__ == "__main__":
    main()
