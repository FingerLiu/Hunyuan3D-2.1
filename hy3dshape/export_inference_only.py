import os
import torch
import logging
from omegaconf import OmegaConf

from hy3dshape.utils.misc import instantiate_from_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: WIP 还不能用
# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def disable_moe_in_config(cfg):
    """
    Best-effort disable MoE switches in config to force Dense DiT.
    This does NOT modify training checkpoints, only the export-time model.
    """
    candidates = [
        ("model", "use_moe"),
        ("model", "denoiser", "use_moe"),
        ("model", "denoiser", "use_moe_attention"),
        ("model", "denoiser", "moe"),
        ("model", "denoiser", "moe_cfg", "enable"),
    ]

    for path in candidates:
        cur = cfg
        ok = True
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            elif hasattr(cur, p):
                cur = getattr(cur, p)
            else:
                ok = False
                break

        if ok:
            try:
                if isinstance(cur, dict):
                    cur["enable"] = False
                else:
                    setattr(cur, "enable", False)
                logger.info(f"Disabled MoE via config path: {'.'.join(path)}")
            except Exception:
                pass


def find_dense_dit(module, max_depth=6, prefix=""):
    """
    Locate Dense HunYuanDiTPlain by checking for Dense attention keys.
    (MoE models will NOT have these keys.)
    """
    try:
        sd = module.state_dict()
        if (
            "x_embedder.weight" in sd
            and "blocks.0.attn1.to_q.weight" in sd
        ):
            logger.info(f"Found Dense HunYuanDiTPlain at: {prefix or '<root>'}")
            return module
    except Exception:
        pass

    if max_depth <= 0:
        return None

    for name, child in module.named_children():
        found = find_dense_dit(
            child,
            max_depth=max_depth - 1,
            prefix=f"{prefix}.{name}" if prefix else name,
        )
        if found is not None:
            return found

    return None


# ----------------------------------------------------------------------
# Main export function
# ----------------------------------------------------------------------
def export_inference_only(
    ckpt_path: str,
    config_path: str,
    output_path: str,
    device: str = "cpu",
):
    """
    Export inference-only *Dense* HunYuanDiTPlain weights.

    Output format (STRICT, loader-compatible):
        {
            "model": <Dense HunYuanDiTPlain.state_dict()>
        }
    """

    # --------------------------------------------------------------
    # 1) Load config
    # --------------------------------------------------------------
    logger.info(f"Loading training config: {config_path}")
    config = OmegaConf.load(config_path)

    # --------------------------------------------------------------
    # 2) Force Dense (disable MoE)
    # --------------------------------------------------------------
    logger.info("Forcing Dense DiT (disabling MoE for export)...")
    disable_moe_in_config(config)

    # --------------------------------------------------------------
    # 3) Instantiate Diffuser (LightningModule)
    # --------------------------------------------------------------
    logger.info("Instantiating Diffuser from config...")
    diffuser = instantiate_from_config(config.model)
    diffuser.to(device)

    # --------------------------------------------------------------
    # 4) Load checkpoint
    # --------------------------------------------------------------
    logger.info(f"Loading checkpoint: {ckpt_path}")

    if os.path.isdir(ckpt_path):
        model_state_file = os.path.join(
            ckpt_path, "checkpoint", "mp_rank_00_model_states.pt"
        )
        if not os.path.exists(model_state_file):
            raise FileNotFoundError(
                f"DeepSpeed model state not found: {model_state_file}"
            )

        ckpt = torch.load(model_state_file, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("module", ckpt)
        state_dict = {
            k.replace("_forward_module.", ""): v
            for k, v in state_dict.items()
        }
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

    missing, unexpected = diffuser.load_state_dict(state_dict, strict=False)
    logger.info(
        f"Checkpoint loaded into Diffuser "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )

    # --------------------------------------------------------------
    # 5) Apply EMA (same semantics as inference)
    # --------------------------------------------------------------
    try:
        ema_cfg = getattr(diffuser, "ema_config", None)
        ema_inference = False
        if ema_cfg is not None:
            ema_inference = bool(
                getattr(ema_cfg, "ema_inference", False)
                or ema_cfg.get("ema_inference", False)
            )

        if ema_inference and hasattr(diffuser, "model_ema"):
            logger.info("Applying EMA weights...")
            diffuser.model_ema.copy_to(diffuser.model)
        else:
            logger.info("EMA not enabled; using last-step weights.")
    except Exception as e:
        logger.warning(f"EMA application failed: {e}")

    # --------------------------------------------------------------
    # 6) Locate Dense HunYuanDiTPlain
    # --------------------------------------------------------------
    logger.info("Locating Dense HunYuanDiTPlain inside pipeline.model...")
    pipeline = diffuser.pipeline
    root_model = pipeline.model

    dit_model = find_dense_dit(root_model)
    if dit_model is None:
        raise RuntimeError(
            "Failed to locate Dense HunYuanDiTPlain. "
            "MoE may still be enabled or config mismatch."
        )

    dit_state = dit_model.state_dict()
    logger.info(f"Dense DiT extracted ({len(dit_state)} parameters).")

    # --------------------------------------------------------------
    # 7) Save (from_single_file-compatible)
    # --------------------------------------------------------------
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(
        {"model": dit_state},
        output_path
    )

    logger.info(f"Inference-only Dense checkpoint saved to: {output_path}")
    logger.info("✅ Export completed successfully.")


# ----------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------
if __name__ == "__main__":
    CKPT_PATH = "output_folder/dit/finetuning_4090_sem_v2/ckpt/ckpt-step=00050000.ckpt"
    CONFIG_PATH = "output_folder/dit/finetuning_4090_sem_v2/hunyuandit-finetuning-4090-24gb-sem.yaml"
    OUTPUT_PATH = "hunyuan3d_dit_sem_dense_inference.pt"

    export_inference_only(
        ckpt_path=CKPT_PATH,
        config_path=CONFIG_PATH,
        output_path=OUTPUT_PATH,
        device="cpu",
    )
