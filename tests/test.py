from src.models.losses import ArcFace, AngularPenaltySMLoss
import torch
import numpy as np
from tqdm import tqdm

x = torch.from_numpy(np.array([[2., 1., 1.1], [1., 2., 0.5], [0.1, 0.4, 3.]])).float()
target = torch.from_numpy(np.array([1, 0, 2])).long()

# arcface_loss = ArcFace(s=64.0, margin=0.5)
arcface_loss = AngularPenaltySMLoss(in_features=3, out_features=3, loss_type='arcface', s=64.0, m=0.5)
sphereface_loss = AngularPenaltySMLoss(in_features=3, out_features=3, loss_type='sphereface')
cosface_loss = AngularPenaltySMLoss(in_features=3, out_features=3, loss_type='cosface')

losses = [arcface_loss, sphereface_loss, cosface_loss]
optims = [torch.optim.Adam(loss.parameters(), lr=0.001) for loss in losses]

epochs = 10000
for loss_fn, optim in zip(losses, optims):
    pbar = tqdm(range(epochs), desc=f"Loss: {loss_fn.loss_type}")
    for epoch in pbar:
        optim.zero_grad()
        loss = loss_fn(x, target)
        loss.backward()

        optim.step()
        pbar.set_postfix({'Loss': loss.item()})
