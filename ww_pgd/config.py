from __future__ import annotations
from dataclasses import dataclass

@dataclass
class WWTailConfig:
    """
    WW-PGD projection configuration.

    This package implements WW-PGD as an add-on to any base optimizer:
      - base_optimizer.step() runs normally (SGD/Adam/AdamW/Muon/etc.)
      - ww_pgd_project() runs at epoch boundaries (or every N epochs)

    This version is the “epoch-homotopy” PGD that:
      - uses fixed q (default 1.0 -> alpha≈2) in the tail template
      - uses WeightWatcher only for tail selection (xmin + detX_num)
      - ramps projection strength from 0 -> 1 over the first ramp_epochs epochs
    """

    enable_tail_pgd: bool = True
    min_tail: int = 5

    # fixed alpha≈2 template via q=1 (q=1/(alpha-1))
    q: float = 1.0

    # PGD strength at full hardness
    blend_eta: float = 0.5
    cayley_eta: float = 0.25
    use_detx: bool = True

    # epoch-based homotopy schedule
    warmup_epochs: int = 0
    ramp_epochs: int = 5

    # trap PGD placeholders (optional; off by default)
    enable_trap_pgd: bool = False
    trap_blend_eta: float = 0.5
    trap_tw_k: float = 2.0
    trap_min_spikes: int = 1

    verbose: bool = False
