from __future__ import annotations

from typing import Optional, List

import torch
import torch.nn as nn
import pandas as pd

from .config import WWTailConfig
from .project import ww_pgd_project, LayerSelector, default_layer_selector, _projection_hardness
from .updates import _project_updates

class WWPGDWrapper:
    """
    Wrap any torch optimizer (SGD/Adam/AdamW/Muon/etc.) and apply WW-PGD projection
    at epoch boundaries (``projection_target="weights"``, the default), or to
    individual optimizer updates (``projection_target="updates"``).

    Update mode computes W_next = W_before + P(W_proposed - W_before).
    Pass ``epoch`` and ``num_epochs`` to ``step`` to use the epoch homotopy.
    ``apply_tail_projection`` is a no-op in this mode, so it cannot project the
    full weights accidentally. Base optimizer state is not projected.

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
        projection_target: str = "weights",
        apply_every_steps: int = 1,
        layer_selector: Optional[LayerSelector] = None,
        ww_logs: Optional[List[pd.DataFrame]] = None,
    ) -> None:
        if projection_target not in ("weights", "updates"):
            raise ValueError("projection_target must be 'weights' or 'updates'")
        if not isinstance(apply_every_steps, int) or isinstance(apply_every_steps, bool) or apply_every_steps < 1:
            raise ValueError("apply_every_steps must be a positive integer")
        if projection_target == "updates" and not 0.0 <= tail_config.blend_eta <= 1.0:
            raise ValueError("Update projection requires 0 <= blend_eta <= 1")
        if projection_target == "updates" and any(
            group.get("differentiable", False) for group in base_optimizer.param_groups
        ):
            raise ValueError("Update projection does not support differentiable optimizers")
        self.model = model
        self.base_optimizer = base_optimizer
        self.tail_config = tail_config
        self.apply_every_epochs = max(1, int(apply_every_epochs))
        self.projection_target = projection_target
        self.apply_every_steps = apply_every_steps
        self.layer_selector = layer_selector or default_layer_selector
        self.ww_logs = ww_logs
        self.global_step = 0

    def step(self, closure=None, *, epoch: int = 0, num_epochs: int = 1):
        """Take a base step, optionally shaping its actual matrix displacement.

        The displacement includes momentum, adaptive preconditioning, learning
        rate, and weight decay. Only trainable module weights owned by the base
        optimizer are eligible; biases and other vectors follow the base step.
        """
        hardness = _projection_hardness(self.tail_config, epoch)
        project_update = (
            self.projection_target == "updates"
            and self.tail_config.enable_tail_pgd
            and hardness > 0
            and self.tail_config.blend_eta > 0
            and (epoch + 1) % self.apply_every_epochs == 0
            and (self.global_step + 1) % self.apply_every_steps == 0
        )
        before = {}
        if project_update:
            optimizer_ids = {id(p) for group in self.base_optimizer.param_groups for p in group["params"]}
            seen = set()
            for name, module in self.model.named_modules():
                p = getattr(module, "weight", None)
                if (
                    isinstance(p, nn.Parameter) and p.requires_grad and p.ndim >= 2
                    and p.is_floating_point() and id(p) in optimizer_ids and id(p) not in seen
                ):
                    seen.add(id(p))
                    # Snapshot before the closure: it may create gradients itself.
                    before[name] = (p, p.detach().clone())
        loss = self.base_optimizer.step(closure)
        self.global_step += 1
        if before:
            with torch.no_grad():
                updates = {}
                originals = {}
                for name, (p, old) in before.items():
                    dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
                    original = old.to(dtype=dtype)
                    originals[p] = original
                    updates[name] = (p, p.detach().to(dtype=dtype) - original)
                projected = _project_updates(
                    self.model, updates, self.tail_config, hardness=hardness,
                    epoch=epoch, num_epochs=num_epochs, global_step=self.global_step,
                    layer_selector=self.layer_selector, ww_logs=self.ww_logs,
                )
                candidates = {p: (originals[p] + delta).to(dtype=p.dtype) for p, delta in projected.items()}
                if any(not torch.isfinite(candidate).all() for candidate in candidates.values()):
                    raise RuntimeError("Projected update produced non-finite weights")
                for p, candidate in candidates.items():
                    p.copy_(candidate)
        return loss

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def apply_tail_projection(self, *, epoch: int, num_epochs: int) -> None:
        if self.projection_target == "updates":
            return
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
