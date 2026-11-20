"""WW-PGD on detached optimizer updates, without analyzing model weights."""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import weightwatcher as ww

from .config import WWTailConfig
from .project import LayerSelector, _shape_single_layer_tail_homotopy


@torch.no_grad()
def _project_updates(
    model: nn.Module,
    updates: Dict[str, Tuple[nn.Parameter, torch.Tensor]],
    cfg: WWTailConfig,
    *,
    hardness: float,
    epoch: int,
    num_epochs: int,
    global_step: int,
    layer_selector: LayerSelector,
    ww_logs: Optional[List[pd.DataFrame]],
) -> Dict[nn.Parameter, torch.Tensor]:
    """Return projected updates only for layers with a usable spectral tail.

    Each update is scaled by its maximum absolute entry for analysis and shaping,
    then restored to its original scale. This keeps fine-tuning deltas away from
    absolute numerical floors. WW and the projector see the same flattened matrix;
    convolutional updates are not analyzed as a different collection of slices.
    """
    matrices = OrderedDict()
    entries = {}
    for name, (parameter, delta) in updates.items():
        if delta.numel() == 0:
            continue
        matrix = delta.reshape(delta.shape[0], -1)
        if min(matrix.shape) < max(2, cfg.min_tail):
            continue
        if not torch.isfinite(matrix).all():
            raise RuntimeError(f"Non-finite optimizer update for layer {name!r}.")
        scale = matrix.abs().max()
        if scale == 0:
            continue
        matrix = matrix / scale
        # Synthetic names also support a root Linear and tied/shared parameters.
        key = f"update_{len(entries)}"
        entries[key] = (name, parameter, matrix, scale)
        matrices[f"{key}.weight"] = matrix.cpu().clone()

    if not matrices:
        return {}

    watcher = ww.WeightWatcher(model=matrices)
    details = watcher.analyze(
        detX=cfg.use_detx, randomize=False, plot=False, savefig=False,
        min_evals=max(2, cfg.min_tail), svd_method="accurate",
    ).copy()
    if details.empty:
        return {}

    key_column = "longname" if "longname" in details.columns else "name"
    projected = {}
    for index, row in details.iterrows():
        entry = entries.get(str(row[key_column]))
        if entry is None:
            continue
        name, parameter, matrix, scale = entry
        row = row.copy()
        row["longname"] = row["name"] = name
        details.loc[index, "longname"] = name
        details.loc[index, "name"] = name
        details.loc[index, "update_scale"] = float(scale)
        module = layer_selector(model, name, row)
        if module is None or getattr(module, "weight", None) is not parameter:
            continue

        if row.get("status", "success") != "success":
            continue
        xmin = float(row.get("xmin", float("nan")))
        if not (math.isfinite(xmin) and xmin > 0):
            continue
        detx_num = row.get("detX_num", float("nan"))
        detx_num = int(detx_num) if cfg.use_detx and pd.notna(detx_num) and detx_num > 0 else None
        shaped = matrix.clone()
        _shape_single_layer_tail_homotopy(
            shaped, xmin=xmin, detx_num=detx_num, cfg=cfg,
            hardness=hardness, eps=0.0,
            rank_rtol=max(matrix.shape) * torch.finfo(matrix.dtype).eps,
            template_dtype=matrix.dtype,
        )
        if not torch.isfinite(shaped).all():
            raise RuntimeError(f"Non-finite projected update for layer {name!r}.")
        # Do not rewrite an unchanged proposal (including a skipped short tail).
        if not torch.equal(shaped, matrix):
            projected[parameter] = (shaped * scale).reshape_as(parameter)

    details["projection_target"] = "updates"
    details["epoch_id"] = epoch
    details["num_epochs"] = num_epochs
    details["global_step"] = details["step"] = global_step
    if ww_logs is not None:
        ww_logs.append(details)
    return projected
