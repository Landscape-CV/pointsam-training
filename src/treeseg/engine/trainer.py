"""
trainer.py - fine-tune Point-SAM on forest crops.

The recipe is Kevin Hu's v3 (Ecomodel jianwei/spcg, pointsam_training/train.py): first click sampled from
a mix of types, first round weighted highest, checkpoint picked by the mask the model itself would choose
after one stem click (val_iou_pick). Removed: the decoder-only and full fine-tune modes and the VRAM guards.
Added: every logged step goes to history.csv straight away, so curves survive a dropped session.

Validation clicks come from the true labels, so these curves measure mask quality given a good click.
They do not measure automatic clicks. The benchmark does that.
"""
import csv
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.point_sam import save_checkpoint

HISTORY_FIELDS = ["step", "epoch", "split", "loss", "iou_first", "val_loss", "val_iou", "val_iou_first",
                  "val_iou_pick", "lr", "minutes"]


def lr_factor(step: int, total_steps: int, warmup: int, min_frac: float) -> float:
    """Linear warmup, then cosine decay down to min_frac of the base rate."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * progress))


def first_clicks(coords: torch.Tensor, gt: torch.Tensor, stem_idx: torch.Tensor, kind: str):
    """First click per mask as (points, labels), or (None, None) to use Point-SAM's own sampler."""
    b, m, _ = gt.shape
    if kind == "interior":
        return None, None
    all_coords = coords.unsqueeze(1).expand(b, m, -1, -1)
    if kind in ("stem", "stem1"):
        k = 2 if kind == "stem" else 1
        idx = stem_idx[:, :, :k]
        points = torch.gather(all_coords, 2, idx.unsqueeze(-1).expand(-1, -1, -1, 3)).reshape(b * m, k, 3)
    elif kind == "random":
        pick = torch.multinomial(gt.reshape(b * m, -1).float() + 1e-9, 1).squeeze(1)
        points = all_coords.reshape(b * m, -1, 3)[torch.arange(b * m, device=coords.device), pick].unsqueeze(1)
    else:
        raise ValueError(f"unknown click kind: {kind}")
    return points, torch.ones(points.shape[:2], dtype=torch.bool, device=coords.device)


def forward_rounds(model, coords, features, gt, points=None, labels=None, is_eval=False) -> list[dict]:
    """Point-SAM's forward pass, but the first click can be fixed. Later clicks correct the last mask."""
    from pc_sam.model.common import repeat_interleave, sample_prompts_adapter
    from pc_sam.model.mask_decoder import AuxInputs
    from torkit3d.nn.functional import batch_index_select

    b, m = gt.shape[:2]
    embedding, patches = model.pc_encoder(coords, features)
    centers, knn_idx = patches["centers"], patches["knn_idx"]
    aux = AuxInputs(coords=coords, features=features, centers=centers)
    position = model.point_encoder.pe_layer(centers)

    #some rounds refine the mask without adding a click
    refine_rounds = []
    if model.enable_mask_refinement_iterations and model.training:
        refine_rounds = [model.prompt_iters - 1]
        if model.prompt_iters > 1:
            refine_rounds.append(int(torch.randint(1, model.prompt_iters, (1,)).item()))

    clicks = coords.new_empty((b * m, 0, 3))
    click_labels = gt.new_empty((b * m, 0))
    prompt_masks = None
    outputs = []
    for i in range(model.prompt_iters):
        if i == 0 and points is not None:
            clicks = torch.cat([clicks, points.to(clicks.dtype)], 1)
            click_labels = torch.cat([click_labels, labels], 1)
        elif i == 0 or i not in refine_rounds:
            new_clicks, new_labels = sample_prompts_adapter(coords, gt, prompt_masks, is_eval=is_eval)
            clicks, click_labels = torch.cat([clicks, new_clicks], 1), torch.cat([click_labels, new_labels], 1)
        sparse = model.point_encoder(clicks, click_labels)
        dense = model.mask_encoder(prompt_masks, coords, centers, knn_idx, center_idx=patches.get("fps_idx"))
        dense = repeat_interleave(dense, sparse.shape[0] // dense.shape[0], 0)
        masks, iou_preds = model.mask_decoder(embedding, position, sparse, dense, aux_inputs=aux,
                                              multimask_output=(i == 0))
        if i == 0:
            chosen = torch.argmax(iou_preds, dim=1)
            prompt_masks = batch_index_select(masks, chosen, dim=1)
        else:
            chosen = 0
            prompt_masks = masks[:, 0]
        outputs.append({"prompt_coords": clicks, "prompt_labels": click_labels, "masks": masks, "iou_preds": iou_preds,
                        "max_iou_pred_ind": chosen, "prompt_masks": prompt_masks})
    return outputs


def weighted_loss(criterion, outputs, gt_flat, first_weight: float):
    """Average the rounds, with extra weight on the first because inference only uses one click."""
    losses, aux = [], []
    for output in outputs:
        loss, details = criterion([output], gt_flat)
        losses.append(loss)
        aux += details
    weights = torch.tensor([first_weight] + [1.0] * (len(losses) - 1), device=losses[0].device)
    return (torch.stack(losses) * weights).sum() / weights.sum(), aux


def collate(batch: list[dict]) -> dict:
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


@torch.no_grad()
def validate(model, loader, criterion, device, click_kind: str = "stem") -> dict:
    """val_iou: after all corrections. val_iou_first: best of the 3 first masks, given the true answer.
    val_iou_pick: the mask the model itself picks after one click, which is what inference uses."""
    model.eval()
    losses, ious, firsts, picks = [], [], [], []
    for batch in loader:
        coords, features, gt = (batch[k].to(device) for k in ("coords", "features", "gt_masks"))
        points, labels = first_clicks(coords, gt, batch["stem_idx"].to(device), click_kind)
        with torch.autocast("cuda"):
            outputs = forward_rounds(model, coords, features, gt, points, labels, is_eval=True)
            gt_flat = gt.flatten(0, 1)
            loss, aux = criterion(outputs, gt_flat)
        losses.append(loss.item())
        ious.append(aux[-1]["iou"].mean().item())
        firsts.append(aux[0]["iou"].mean().item())
        chosen = torch.argmax(outputs[0]["iou_preds"], dim=1)
        pred = outputs[0]["masks"][torch.arange(len(chosen), device=chosen.device), chosen] > 0
        inter = (pred & gt_flat).sum(-1).float()
        union = (pred | gt_flat).sum(-1).float().clamp(min=1)
        picks.append((inter / union).mean().item())
    model.train()
    return {"val_loss": float(np.mean(losses)), "val_iou": float(np.mean(ious)),
            "val_iou_first": float(np.mean(firsts)), "val_iou_pick": float(np.mean(picks))}


def train(model, train_set, val_set, cfg: dict, out_dir: str, device: str = "cuda") -> str:
    """Run the recipe in cfg. Writes history.csv, best.safetensors and last.safetensors to out_dir."""
    from pc_sam.model.loss import Criterion

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    model.to(device).train()
    loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                        num_workers=cfg.get("num_workers", 0), collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate)

    encoder = [p for p in model.pc_encoder.parameters() if p.requires_grad]
    encoder_ids = {id(p) for p in encoder}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": cfg["encoder_lr"]}, {"params": rest, "lr": cfg["lr"]}],
                                  weight_decay=cfg["weight_decay"])
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda")
    criterion = Criterion(use_soft_iou=False)

    click_kinds, click_probs = list(cfg["click_mix"]), np.array(list(cfg["click_mix"].values()), dtype=float)
    click_probs /= click_probs.sum()
    click_rng = np.random.default_rng(cfg["seed"] + 3)
    crops_per_step = cfg["batch_size"] * cfg["grad_accum"]
    params = [p for p in model.parameters() if p.requires_grad]

    history_path = os.path.join(out_dir, "history.csv")
    with open(history_path, "w", newline="") as f:
        csv.DictWriter(f, HISTORY_FIELDS).writeheader()

    def log(row: dict):
        row = {"step": step, "epoch": round(step * crops_per_step / len(train_set), 3),
               "lr": optimizer.param_groups[-1]["lr"], "minutes": round((time.time() - start) / 60, 2), **row}
        with open(history_path, "a", newline="") as f:
            csv.DictWriter(f, HISTORY_FIELDS).writerow(row)

    start = time.time()
    step, micro, epoch, best, bad_rounds, stop = 0, 0, 0, -1.0, 0, False
    optimizer.zero_grad(set_to_none=True)
    while step < cfg["max_steps"] and not stop:
        epoch += 1
        train_set.epoch = epoch
        for batch in loader:
            coords, features, gt = (batch[k].to(device) for k in ("coords", "features", "gt_masks"))
            kind = str(click_rng.choice(click_kinds, p=click_probs))
            points, labels = first_clicks(coords, gt, batch["stem_idx"].to(device), kind)
            with torch.autocast("cuda"):
                outputs = forward_rounds(model, coords, features, gt, points, labels)
                loss, aux = weighted_loss(criterion, outputs, gt.flatten(0, 1), cfg["first_round_weight"])
            scaler.scale(loss / cfg["grad_accum"]).backward()
            micro += 1
            if micro % cfg["grad_accum"]:
                continue

            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * lr_factor(step, cfg["max_steps"], cfg["warmup_steps"], cfg["min_lr_frac"])
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % cfg["log_every"] == 0:
                iou_first = aux[0]["iou"].mean().item()
                log({"split": "train", "loss": round(loss.item(), 5), "iou_first": round(iou_first, 5)})
                print(f"step {step}/{cfg['max_steps']}  loss={loss.item():.4f}  iou_first={iou_first:.3f}", flush=True)

            if step % cfg["val_every"] == 0 or step == cfg["max_steps"]:
                metrics = validate(model, val_loader, criterion, device, cfg["val_click"])
                log({"split": "val", **{k: round(v, 5) for k, v in metrics.items()}})
                print(f"val step {step}  loss={metrics['val_loss']:.4f}  iou={metrics['val_iou']:.3f}  "
                      f"pick={metrics['val_iou_pick']:.3f}", flush=True)
                save_checkpoint(model, os.path.join(out_dir, "last.safetensors"))
                if metrics[cfg["select_metric"]] > best:
                    best, bad_rounds = metrics[cfg["select_metric"]], 0
                    save_checkpoint(model, os.path.join(out_dir, "best.safetensors"))
                    print(f"  new best {cfg['select_metric']}={best:.3f}", flush=True)
                else:
                    bad_rounds += 1
                    if bad_rounds >= cfg["early_stop_patience"]:
                        print(f"early stop at step {step}, best {cfg['select_metric']}={best:.3f}", flush=True)
                        stop = True
            if stop or step >= cfg["max_steps"]:
                break
    print(f"done in {(time.time() - start) / 60:.1f} min, best {cfg['select_metric']}={best:.3f}", flush=True)
    return history_path
