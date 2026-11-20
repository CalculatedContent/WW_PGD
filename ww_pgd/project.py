from __future__ import annotations

import math
from typing import Callable, Optional, List, Dict

import torch
import torch.nn as nn
import pandas as pd
import weightwatcher as ww

from .config import WWTailConfig

LayerSelector = Callable[[nn.Module, str, pd.Series], Optional[nn.Module]]

def default_layer_selector(model: nn.Module, lname: str, row: pd.Series) -> Optional[nn.Module]:
    if not lname:
        return model if hasattr(model, "weight") else None
    cur: nn.Module = model
    for part in lname.split("."):
        if hasattr(cur, part):
            cur = getattr(cur, part)
        else:
            return None
    return cur if hasattr(cur, "weight") else None

def _cayley_update_lambdas(
    lam_current: torch.Tensor,
    lam_target: torch.Tensor,
    eta: float,
    eps: float = 1e-8,
    min_ratio: float = 0.1,
    max_ratio: float = 10.0,
) -> torch.Tensor:
    if eta <= 0.0:
        return lam_target
    log_curr = torch.log(lam_current + eps)
    log_tgt = torch.log(lam_target + eps)
    g = log_curr - log_tgt
    ratio = (1.0 - eta * g) / (1.0 + eta * g)
    ratio = torch.clamp(ratio, min_ratio, max_ratio)
    return lam_current * ratio

def _shape_single_layer_tail_homotopy(
    weight: torch.Tensor,
    *,
    xmin: float,
    detx_num: Optional[int],
    cfg: WWTailConfig,
    hardness: float,
    eps: float = 1e-8,
    rank_rtol: Optional[float] = None,
    template_dtype: torch.dtype = torch.float32,
) -> None:
    with torch.no_grad():
        W = weight.data
        W2 = W.reshape(W.size(0), -1)

        U, S, Vh = torch.linalg.svd(W2, full_matrices=False)
        S = S.clamp_min(eps)
        lam = S**2
        n = lam.numel()
        if n == 0:
            return

        # Update mode must not turn roundoff-sized singular values into a tail.
        # This shares the projection SVD rather than adding another decomposition.
        resolved = torch.ones_like(S, dtype=torch.bool)
        if rank_rtol is not None:
            resolved = S > S.max() * rank_rtol
        resolved_count = int(resolved.sum().item())
        if resolved_count < cfg.min_tail:
            return

        if not (math.isfinite(xmin) and xmin > 0.0):
            return

        # Tail threshold: midpoint of xmin-tail size and detX tail size (if available)
        lam_thr = float(xmin)

        if detx_num is not None and detx_num > 0:
            pl_tail_size = int(((lam >= float(xmin)) & resolved).sum().item())
            if pl_tail_size > 0:
                k_pl = max(1, min(n, pl_tail_size))
                k_detx = max(1, min(resolved_count, detx_num))
                k_star = max(1, int(0.5 * (k_pl + k_detx)))
                lam_thr = max(lam_thr, lam[k_star - 1].item())

        tail_mask = (lam >= lam_thr) & resolved
        tail_size = int(tail_mask.sum().item())
        if tail_size < cfg.min_tail:
            return

        lam_tail = lam[tail_mask]

        # Fixed q template (q≈1 -> alpha≈2)
        r = torch.arange(1, tail_size + 1, device=W.device, dtype=template_dtype)
        mu = r.pow(-cfg.q)

        # Match tail TraceLog
        T_target = torch.log(lam_tail).sum()
        sum_log_mu = torch.log(mu).sum()
        A = torch.exp((T_target - sum_log_mu) / tail_size)
        lam_tail_target = A * mu

        # Softened Cayley and blending
        cayley_eta_eff = float(hardness) * cfg.cayley_eta
        lam_tail_new = _cayley_update_lambdas(lam_tail, lam_tail_target, cayley_eta_eff, eps=eps)

        # Retract to preserve tail TraceLog
        log_tail_new = torch.log(lam_tail_new + eps).sum()
        shift = (T_target - log_tail_new) / tail_size
        lam_tail_new = lam_tail_new * torch.exp(shift)

        # Reconstruct
        S_new = S.clone()
        S_new[tail_mask] = torch.sqrt(lam_tail_new.clamp_min(eps))
        W2_shaped = (U * S_new.unsqueeze(0)) @ Vh

        blend_eta_eff = float(hardness) * cfg.blend_eta
        W_new = (1.0 - blend_eta_eff) * W2 + blend_eta_eff * W2_shaped
        weight.data.copy_(W_new.reshape_as(W))

def _projection_hardness(cfg: WWTailConfig, epoch: int) -> float:
    if epoch < cfg.warmup_epochs:
        return 0.0
    if epoch >= cfg.warmup_epochs + cfg.ramp_epochs:
        return 1.0
    return max(0.0, min(1.0, (epoch - cfg.warmup_epochs + 1) / max(cfg.ramp_epochs, 1)))

def ww_pgd_project(
    model: nn.Module,
    cfg: WWTailConfig,
    *,
    epoch: int,
    num_epochs: int,
    global_step: Optional[int] = None,
    ww_logs: Optional[List[pd.DataFrame]] = None,
    layer_selector: Optional[LayerSelector] = None,
) -> None:
    """
    Run WeightWatcher analyze(detX=True) and apply WW-PGD tail projection to layers.
    """
    if layer_selector is None:
        layer_selector = default_layer_selector

    # hardness schedule
    hardness = _projection_hardness(cfg, epoch)

    watcher = ww.WeightWatcher(model=model)
    details: pd.DataFrame = watcher.analyze(detX=True, randomize=False, plot=False)

    # add metadata
    details = details.copy()
    details["epoch_id"] = epoch
    details["num_epochs"] = num_epochs
    if global_step is not None:
        details["global_step"] = global_step
        details["step"] = global_step
    if ww_logs is not None:
        ww_logs.append(details.copy())

    key = "longname" if "longname" in details.columns else "name"
    has_detx_num = "detX_num" in details.columns

    for _, row in details.iterrows():
        lname = str(row[key])
        xmin = float(row.get("xmin", float("nan")))
        mod = layer_selector(model, lname, row)
        if mod is None:
            continue

        detx_num = None
        if cfg.use_detx and has_detx_num:
            v = row.get("detX_num", float("nan"))
            if pd.notna(v) and float(v) > 0:
                detx_num = int(v)

        if cfg.enable_tail_pgd and hardness > 0.0:
            _shape_single_layer_tail_homotopy(
                mod.weight,
                xmin=xmin,
                detx_num=detx_num,
                cfg=cfg,
                hardness=hardness,
            )
