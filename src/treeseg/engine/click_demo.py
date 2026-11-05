"""
click_demo.py - show what one click does: where it lands, and the 3 masks the model returns for it.
"""
import numpy as np
import torch

from .trainer import first_clicks


def click_positions(item: dict, mask_index: int = 0, seed: int = 0) -> dict:
    """Where each kind of training click lands on one target tree, in the crop's normalised coordinates."""
    coords, gt = item["coords"], item["gt_masks"][mask_index]
    stem, upper = item["stem_idx"][mask_index].tolist()
    rng = np.random.default_rng(seed)
    return {"stem": coords[stem].numpy(), "upper": coords[upper].numpy(),
            "random": coords[int(rng.choice(np.flatnonzero(gt.numpy())))].numpy()}


@torch.no_grad()
def predict_one_click(model, item: dict, mask_index: int = 0, kind: str = "stem1", device: str = "cuda") -> dict:
    """Run one click on one tree. Returns the 3 candidate masks, the model's score for each and their true IoU."""
    coords = item["coords"].unsqueeze(0).to(device)
    features = item["features"].unsqueeze(0).to(device)
    gt = item["gt_masks"][mask_index:mask_index + 1].unsqueeze(0).to(device)
    stem_idx = item["stem_idx"][mask_index:mask_index + 1].unsqueeze(0).to(device)
    points, labels = first_clicks(coords, gt, stem_idx, kind)
    model.eval()
    masks, scores = model.predict_masks(coords, features, points, labels, prompt_masks=None, multimask_output=True)
    masks = (masks[0] > 0).float().cpu().numpy() > 0.5
    scores = scores[0].float().cpu().numpy()
    truth = item["gt_masks"][mask_index].numpy()
    ious = np.array([(m & truth).sum() / max((m | truth).sum(), 1) for m in masks])
    return {"masks": masks, "scores": scores, "ious": ious, "picked": int(scores.argmax()), "best": int(ious.argmax()),
            "click": points[0].cpu().numpy(), "truth": truth, "coords": item["coords"].numpy()}
