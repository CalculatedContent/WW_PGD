# ww_pgd

**ww_pgd** (WeightWatcher Projected Gradient Descent) is a small PyTorch add-on that applies a **spectral tail projection**
(“WW-PGD”) at epoch boundaries using the **WeightWatcher** library.

It is designed to wrap any PyTorch optimizer:
- SGD / SGD+momentum
- Adam / AdamW
- Muon (or any custom optimizer with a `step()` method)

## Install

```bash
pip install ww_pgd
```


## Quickstart

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import ww_pgd

model = nn.Linear(10, 10)

base_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
cfg = ww_pgd.WWTailConfig(warmup_epochs=0, ramp_epochs=5)

opt = ww_pgd.WWPGDWrapper(model, base_opt, cfg)

for epoch in range(num_epochs):
    for xb, yb in loader:
        loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # epoch-boundary spectral projection
    opt.apply_tail_projection(epoch=epoch, num_epochs=num_epochs)
