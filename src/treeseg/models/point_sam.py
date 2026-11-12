"""
point_sam.py - load upstream Point-SAM and choose what trains.

Point-SAM itself is not in this repo: clone https://github.com/zyc00/Point-SAM into thirdparty/Point-SAM
and put the Hugging Face checkpoint (yuchen0187/Point-SAM) at the path in the config. Its torkit3d
dependency is CUDA-only, so without a CUDA GPU the plain-PyTorch versions in torkit3d_fallback.py stand in
for it and the model runs on cpu or Apple mps (see docs/no_cuda.md).
"""
import os
import sys

from .lora import add_lora, merged_lora
from .torkit3d_fallback import install_torkit3d


def add_point_sam_to_path(root: str = "thirdparty/Point-SAM") -> str:
    root = os.path.abspath(root)
    if not os.path.isdir(os.path.join(root, "pc_sam")):
        raise FileNotFoundError(f"Point-SAM not found at {root}")
    if root not in sys.path:
        sys.path.insert(0, root)
    #must come before the first pc_sam import, which is always after this function
    install_torkit3d()
    return root


def build_point_sam(checkpoint: str, group_number: int = 512, group_size: int = 64, root: str = "thirdparty/Point-SAM"):
    """Stock Point-SAM. Patch groups are set for 10,000-point crops, not the whole-object default."""
    root = add_point_sam_to_path(root)
    import hydra
    from omegaconf import OmegaConf
    from pc_sam.utils.torch_utils import replace_with_fused_layernorm
    from safetensors.torch import load_model

    with hydra.initialize_config_dir(config_dir=os.path.join(root, "configs"), version_base=None):
        cfg = hydra.compose(config_name="large")
        OmegaConf.resolve(cfg)
    model = hydra.utils.instantiate(cfg.model)
    #the fused LayerNorm needs NVIDIA apex; plain LayerNorm gives the same results, only slower
    try:
        model.apply(replace_with_fused_layernorm)
    except Exception as exc:
        print(f"fused LayerNorm unavailable ({exc}), using nn.LayerNorm")
    load_model(model, checkpoint)
    grouper = model.pc_encoder.patch_embed.grouper
    grouper.num_groups, grouper.group_size = group_number, group_size
    return model


def prepare_for_lora(model, rank: int = 8, alpha: float = 16.0, targets=("q_proj", "v_proj")) -> None:
    """Freeze everything, then train LoRA adapters, patch embedding and the prompt and mask decoders."""
    for p in model.parameters():
        p.requires_grad = False
    wrapped = add_lora(model.pc_encoder.transformer, targets, rank, alpha)
    for part in (model.pc_encoder.patch_embed, model.point_encoder, model.mask_encoder, model.mask_decoder):
        for p in part.parameters():
            p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA adapters: {wrapped}, trainable {trainable / 1e6:.1f}M of {total / 1e6:.1f}M parameters")


def save_checkpoint(model, path: str) -> None:
    from safetensors.torch import save_model

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with merged_lora(model):
        save_model(model, path)
