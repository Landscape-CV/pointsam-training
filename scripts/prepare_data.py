"""Build tiles and crops for every strip in a data config. No GPU needed.

notebooks/00_dataset.ipynb does the same thing with pictures and checks; this is for running it unattended.
The original scan must be in the config's source_dir. Get it with:

  hf download haag-lidar/wytham-isprs-tiles --repo-type dataset --local-dir data/wytham_isprs --include "raw/*"

  python scripts/prepare_data.py [config] [--crops-only]
"""
import sys

import yaml

sys.path.insert(0, "src")

from treeseg.data.prepare import build_dataset


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cfg = yaml.safe_load(open(args[0] if args else "configs/data/isprs_wytham.yaml"))
    build_dataset(cfg, crops_only="--crops-only" in sys.argv)


if __name__ == "__main__":
    main()
