"""Fine-tune a loaded classifier with WW-PGD on each optimizer update.

Call finetune(pretrained_model, train_loader) after loading your own checkpoint.
The loader should yield (inputs, class_indices); the model should return logits.
This example does not download data or claim a measured accuracy improvement.
"""
import torch
import torch.nn.functional as F

import ww_pgd


def finetune(model, train_loader, *, num_epochs=3, lr=1e-5, apply_every_steps=1):
    device = next(model.parameters()).device
    base_opt = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
        weight_decay=0.01,
    )
    cfg = ww_pgd.WWTailConfig(warmup_epochs=0, ramp_epochs=num_epochs)
    opt = ww_pgd.WWPGDWrapper(
        model, base_opt, cfg,
        projection_target="updates",
        apply_every_steps=apply_every_steps,
    )

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inputs), targets)
            loss.backward()
            opt.step(epoch=epoch, num_epochs=num_epochs)

    return opt
