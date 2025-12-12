import torch
import torch.nn as nn
import ww_pgd

def test_wrapper_step_runs():
    model = nn.Linear(4, 3)
    base = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = ww_pgd.WWTailConfig(warmup_epochs=9999, ramp_epochs=1)  # disable projection
    opt = ww_pgd.WWPGDWrapper(model, base, cfg)

    x = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    opt.step()  # should not raise
    opt.zero_grad()
