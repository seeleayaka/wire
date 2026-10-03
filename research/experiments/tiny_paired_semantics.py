"""Small fixed nonlinear paired semantics, no metadata or annotation inputs."""
import torch
from torch import nn
from torch.nn import functional as F


class TinyPairedHead(nn.Sequential):
    def __init__(self,dimensions=6144):
        super().__init__(nn.Linear(dimensions,16),nn.ReLU(),nn.Dropout(.1),nn.Linear(16,3))


def fit(features,labels):
    if features.ndim!=2 or labels.ndim!=1 or len(features)!=len(labels) or not len(labels):
        raise ValueError('Nonlinear training shapes mismatch')
    if not torch.isfinite(features).all() or labels.dtype!=torch.long or (labels<0).any() or (labels>2).any():
        raise ValueError('Invalid nonlinear training data')
    counts=torch.bincount(labels,minlength=3).float()
    if (counts==0).any():raise ValueError('Training fold lacks a class')
    torch.manual_seed(0)
    head=TinyPairedHead(features.shape[1])
    nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
    weights=counts.sum()/(3*counts)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.001)
    head.train()
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(head(features),labels,weight=weights)
        if not torch.isfinite(loss):raise ValueError('Nonfinite nonlinear training loss')
        loss.backward();optimizer.step()
    return head.eval().requires_grad_(False)
