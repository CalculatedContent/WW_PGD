# ww_pgd

**ww_pgd** (WeightWatcher Projected Gradient Descent) is a small PyTorch add-on that applies a **spectral tail projection**
(“WW-PGD”) at epoch boundaries using the **WeightWatcher** library.

It is designed to wrap any PyTorch optimizer:
- SGD / SGD+momentum
- Adam / AdamW
- Muon (or any custom optimizer with a `step()` method)

Warning: This very experimental, optimized for understanding if out-of-sample performance increases,
but is not optimized yet for numerical performance.

## Install

```bash
pip install ww_pgd
```


## Quickstart:
Trivial exmaple, single layer trained on FashionMNIST
This will drive the layer alpha to 2.0 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import ww_pgd

device = "cuda" if torch.cuda.is_available() else "cpu"

# Data loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(train_ds, batch_size=128, shuffle=True)

# Model
model = nn.Linear(28 * 28, 10).to(device)

# Optimizers
base_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
cfg = ww_pgd.WWTailConfig(warmup_epochs=0, ramp_epochs=5)
opt = ww_pgd.WWPGDWrapper(model, base_opt, cfg)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        xb = xb.view(xb.size(0), -1)  # flatten

        loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # epoch-boundary spectral projection
    opt.apply_tail_projection(epoch=epoch, num_epochs=num_epochs)
    print(f"epoch {epoch+1}/{num_epochs} done")
```

### Evaluate
final results with weightwatcher

```python
import weightwatcher as ww
watcher = ww.WeightWatcher(model=model)
details = watcher.analyze(detX=True, randomize=False, plot=True)
details
```

### Realistic Example
Train a 3-layer MLP on FashionMNIST using AdamW+Ww_PGD

You should find that the layer alphas converge to 2.0 and
that the test accuracies match that of the baseline (AdamW).

See the Notebook WW-PGD-QuickStart.ipynb


<hr>

## Contributors

[Charles H Martin, PhD](https://www.linkedin.com/in/charlesmartin14)
[Hari Kishan Prakash](https://www.linkedin.com/in/hari-kishan-prakash-2b786967/)

<hr>

#### Calculation Consulting Practice
Need help with AI ?  Talk to Chuck

[Calculation Consulting homepage](https://calculationconsulting.com)

[Calculated Content Blog](https://calculatedcontent.com)