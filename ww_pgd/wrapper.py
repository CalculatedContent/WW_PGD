from __future__ import annotations

from typing import Optional, List

import torch
import torch.nn as nn
import pandas as pd

from .config import WWTailConfig
from .project import ww_pgd_project, LayerSelector, default_layer_selector

class WWPGDWrapper:
    """
    Wrap any torch optimizer (SGD/Adam/AdamW/Muon/etc.) and apply WW-PGD projection
    at epoch boundaries.

    Typical usage:

        base_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        cfg = WWTailConfig(warmup_epochs=0, ramp_epochs=5)
        ww_opt = WWPGDWrapper(model, base_opt, cfg)

        for epoch in range(num_epochs):
            for xb, yb in loader:
                loss = ...
                loss.backward()
                ww_opt.step()
                ww_opt.zero_grad()
            ww_opt.apply_tail_projection(epoch=epoch, num_epochs=num_epochs)
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        tail_config: WWTailConfig,
        *,
        apply_every_epochs: int = 1,
        layer_selector: Optional[LayerSelector] = None,
        ww_logs: Optional[List[pd.DataFrame]] = None,
    ) -> None:
        self.model = model
        self.base_optimizer = base_optimizer
        self.tail_config = tail_config
        self.apply_every_epochs = max(1, int(apply_every_epochs))
        self.layer_selector = layer_selector or default_layer_selector
        self.ww_logs = ww_logs
        self.global_step = 0

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self.global_step += 1
        return loss

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def apply_tail_projection(self, *, epoch: int, num_epochs: int) -> None:
        if (epoch + 1) % self.apply_every_epochs != 0:
            return
        ww_pgd_project(
            self.model,
            self.tail_config,
            epoch=epoch,
            num_epochs=num_epochs,
            global_step=self.global_step,
            ww_logs=self.ww_logs,
            layer_selector=self.layer_selector,
        )
