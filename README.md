# ww_pgd

**ww_pgd** (WeightWatcher Projected Gradient Descent) is a small PyTorch add-on that applies a **spectral tail projection**
(“WW-PGD”) using the **WeightWatcher** library. It can project full weights at epoch
boundaries, or project only optimizer updates for fine-tuning.

It is designed to wrap any PyTorch optimizer:
- SGD / SGD+momentum
- Adam / AdamW
- Muon (not tested yet)

Warning: This very experimental, optimized for understanding if out-of-sample performance increases,
but is not optimized yet for numerical performance.

## Install

```bash
pip install git+https://github.com/CalculatedContent/WW_PGD.git
```

To try update projection before this branch is merged, install and test the
branch locally (no separate compilation step is needed):

```bash
git clone --branch finetune-update-projection https://github.com/CalculatedContent/WW_PGD.git
cd WW_PGD
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Fine-tuning: project only the weight update

Use `projection_target="updates"` when adapting a pretrained model. For each
selected step and layer, the wrapper computes:

```text
W_proposed = base_optimizer.step(W_before)
delta_W = W_proposed - W_before
W_next = W_before + WW_PGD(delta_W)
```

WeightWatcher selects the tail from **delta_W**, and the spectral projection shapes
**delta_W**. The pretrained weight matrix is never analyzed or projected in this
mode. `delta_W` is the actual displacement produced by the base optimizer,
including its learning rate, momentum/adaptive scaling, and weight decay; it is
neither the raw gradient nor the accumulated change since the pretrained checkpoint.
Adding the projected update can still change the final weight spectrum: the
invariant is that the spectral operation is applied only to the update.

The `q=1` template is an experimental spectral regularizer of the step. This
mode does not establish an HTSR/SETOL critical-manifold condition for the weights,
does not impose `detX=1`, and does not preserve the update's Frobenius norm. The
tail TraceLog retraction matches the selected update tail's existing TraceLog
before the final blend; a partial blend need not preserve that TraceLog exactly.

```python
import torch
import ww_pgd

# Load your pretrained checkpoint into model before starting fine-tuning.
base_opt = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=1e-5,
    weight_decay=0.01,
)
cfg = ww_pgd.WWTailConfig(warmup_epochs=0, ramp_epochs=5)
opt = ww_pgd.WWPGDWrapper(
    model, base_opt, cfg,
    projection_target="updates",
    apply_every_steps=1,
)

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        opt.step(epoch=epoch, num_epochs=num_epochs)
```

Pass the current zero-based `epoch` to `step` so `warmup_epochs` and `ramp_epochs`
advance. Without these keywords, `step` uses `epoch=0, num_epochs=1`; for a
step-based loop with immediate full strength, set `ramp_epochs=0`.
`apply_tail_projection(...)` is a no-op in update mode, so leaving an old
epoch-boundary call in the training loop cannot project full weights.

Behavior and limits:

- Projection applies to trainable `module.weight` parameters with two or more
  dimensions that belong to the base optimizer. Biases and normalization vectors
  follow the base optimizer. Frozen and excluded weights are not projected, and
  tied weights are processed once.
- `apply_every_steps=N` projects every Nth proposed step. Intervening steps use
  the base optimizer unchanged; updates are not accumulated between projections.
  `apply_every_epochs=N` additionally restricts projection to every Nth epoch.
- A zero update, an insufficient tail, a disabled projection, or zero homotopy
  strength leaves the base optimizer's proposal unchanged.
- Update shaping excludes numerically unresolved singular values using
  `S > S_max * max(matrix.shape) * eps`, where `eps` is machine epsilon for the
  SVD working dtype. If fewer than `min_tail` resolved modes remain, shaping is
  skipped even if WeightWatcher reports a usable cutoff. This is a relative
  numerical-rank test, not an entropy-based effective rank or an absolute norm
  cutoff. A uniformly tiny but well-resolved update can still be projected.
- Each nonzero update is divided by its largest absolute entry before analysis
  and projection, then restored to its original scale. This avoids imposing an
  absolute spectral floor on small fine-tuning updates. SVD runs in float32 for
  float16/bfloat16/float32 parameters and float64 for float64 parameters. Convolution
  kernels are flattened to `[out_channels, -1]` for both analysis and projection.
- No positive floor is imposed on the restored update: that could inflate tiny
  steps or create artificial modes. Update mode requires `0 <= blend_eta <= 1`.
  In exact arithmetic, the projection keeps the update's singular directions
  with nonnegative coefficients. This does not guarantee coordinate-wise sign
  preservation, a bound on the step norm, or a decrease in loss.
- Optimizer momentum and Adam moment buffers retain their ordinary base-optimizer
  values. This is a projection of the resulting parameter step, not of optimizer
  state. Differentiable optimizers are not supported in update mode.
- The displacement is measured after the base step, so updates already rounded
  away by low-precision parameter storage cannot be recovered. Float32 master
  weights are preferable for small fine-tuning steps.
- This experimental path performs WeightWatcher analysis and SVD on selected
  steps and keeps temporary weight/update copies. It is not optimized for large
  models; `apply_every_steps` can reduce its cost. It does not establish an
  accuracy improvement or guarantee alpha=2 for the full adapted weights.

When collecting `ww_logs`, rows are labeled `projection_target="updates"` and use
the original layer names. Spectral values such as `xmin` describe the normalized
update; multiply eigenvalue quantities by `update_scale**2` to recover the original
update units. See [examples/minimal_finetune.py](examples/minimal_finetune.py) for a
reusable classification loop.

For an executed, matched fine-tuning test, see
[the Fashion-MNIST translation experiment](experiments/finetune_updates/README.md)
and [its recorded results](experiments/finetune_updates/results/fashion_shift_20261115/RESULTS.md).
Across three seeds, shifted-test accuracy was 70.11% for AdamW and 69.44% for
update-only WW-PGD with the fixed pilot configuration. The experiment measures
alpha of full weights, cumulative checkpoint differences, and proposed/applied
steps; it includes an exactly matching disabled-projection control. This pilot
checks implementation behavior and does not demonstrate an accuracy benefit.

The default `projection_target="weights"` retains the epoch-boundary interface below.


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

See the notebook [WW_PGD_QuickStart.ipynb](examples/WW_PGD_QuickStart.ipynb)

<hr>

## Contributors

[Charles H Martin, PhD](https://www.linkedin.com/in/charlesmartin14)
[Hari Kishan Prakash](https://www.linkedin.com/in/hari-kishan-prakash-2b786967/)

<hr>

#### Calculation Consulting Practice
Need help with AI ?  Talk to Chuck

[Calculation Consulting homepage](https://calculationconsulting.com)

[Calculated Content Blog](https://calculatedcontent.com)
