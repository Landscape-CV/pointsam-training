"""
download.py - get the data from our Hugging Face dataset, and compare a local build with it.

The dataset holds the authors' original scan (raw/) and the tiles and crops cut from it. It is public,
so nothing here needs a login.
"""
import hashlib
import json
import os

REPO = "haag-lidar/wytham-isprs-tiles"
PARTS = {"raw": ["raw/*"], "built": ["tiles/*", "crops/*", "summary.json"]}


def fetch(part: str, out_dir: str, repo: str = REPO) -> str:
    """Download one part into out_dir, keeping the dataset's folder names. Files already there and unchanged are skipped."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()  # hundreds of small files would otherwise fill the notebook with progress lines
    return snapshot_download(repo, repo_type="dataset", local_dir=out_dir, allow_patterns=PARTS[part])


def published_summary(repo: str = REPO) -> dict:
    from huggingface_hub import hf_hub_download

    return json.load(open(hf_hub_download(repo, "summary.json", repo_type="dataset")))


def published_checksums(repo: str = REPO) -> dict[str, str]:
    """SHA-256 of every large file in the dataset, by its path. Read from the file listing, so nothing is downloaded."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_tree(repo, repo_type="dataset", recursive=True)
    return {f.path: f.lfs.sha256 for f in files if getattr(f, "lfs", None) is not None}


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_with_published(out_dir: str, repo: str = REPO) -> list[dict]:
    """Is a local build the same as the published copy? One row per thing compared.

    Tile labels are compared byte for byte. Tile point files and crops carry a date inside the file, so they
    can't be: tiles are covered by their labels, and crops by their window, point count and tree count."""
    local = json.load(open(os.path.join(out_dir, "summary.json")))
    published, sums = published_summary(repo), published_checksums(repo)
    rows = []
    labels = sorted(p for p in sums if p.startswith("tiles/") and p.endswith("_instances.npy"))
    same = sum(os.path.exists(os.path.join(out_dir, p)) and sha256(os.path.join(out_dir, p)) == sums[p] for p in labels)
    rows.append({"what": "tile label files identical", "local": f"{same} of {len(labels)}", "published": "", "same": same == len(labels)})
    for split, theirs in published["strips"].items():
        ours = local["strips"].get(split, {})
        rows.append({"what": f"{split}: points after thinning", "local": f"{ours.get('voxels', 0):,}", "published": f"{theirs['voxels']:,}",
                     "same": ours.get("voxels") == theirs["voxels"]})
        rows.append({"what": f"{split}: share removed as ground", "local": f"{ours.get('removed_as_ground', float('nan')):.4%}",
                     "published": f"{theirs['removed_as_ground']:.4%}", "same": ours.get("removed_as_ground") == theirs["removed_as_ground"]})
        mine = {c["crop"]: c for c in ours.get("crops", [])}
        matching = sum(mine.get(c["crop"]) == c for c in theirs["crops"])
        rows.append({"what": f"{split}: crops with the same window, points and trees", "local": f"{matching} of {len(mine)}",
                     "published": f"{len(theirs['crops'])}", "same": matching == len(theirs["crops"]) == len(mine)})
    return rows
