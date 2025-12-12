import torch
import torch.nn as nn
import ww_pgd

def test_apply_tail_projection_signature():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 3))
    base = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = ww_pgd.WWTailConfig(warmup_epochs=0, ramp_epochs=1, verbose=False)
    opt = ww_pgd.WWPGDWrapper(model, base, cfg)

    # This test only checks the call path; it will import weightwatcher.
    opt.apply_tail_projection(epoch=0, num_epochs=1)
