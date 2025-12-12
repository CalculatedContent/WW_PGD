import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import ww_pgd

device = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        return self.net(x)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

model = MLP().to(device)

base_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
cfg = ww_pgd.WWTailConfig(warmup_epochs=0, ramp_epochs=5, verbose=False)

opt = ww_pgd.WWPGDWrapper(model, base_opt, cfg)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    opt.apply_tail_projection(epoch=epoch, num_epochs=num_epochs)
    print("epoch", epoch, "done")
