"""Relative bounded box geometry, independent of photo names and GT at inference."""
import math


def geometry(box):
    if len(box)!=4 or not all(math.isfinite(float(v)) for v in box):raise ValueError('Invalid box coordinates')
    l,t,r,b=map(float,box);w,h=r-l,b-t
    if w<=0 or h<=0:raise ValueError('Nonpositive box size')
    return (l+r)/2,(t+b)/2,w,h


def encode(box,target):
    x,y,w,h=geometry(box);X,Y,W,H=geometry(target)
    return [(X-x)/w,(Y-y)/h,math.log(W/w),math.log(H/h)]


def bounded(delta):
    if len(delta)!=4 or not all(math.isfinite(float(v)) for v in delta):raise ValueError('Invalid relative box prediction')
    return [max(-.15,min(.15,float(delta[0]))),max(-.15,min(.15,float(delta[1]))),
        max(math.log(.8),min(math.log(1.2),float(delta[2]))),max(math.log(.8),min(math.log(1.2),float(delta[3])))]


def decode(box,delta):
    x,y,w,h=geometry(box);dx,dy,dw,dh=bounded(delta)
    cx,cy=x+dx*w,y+dy*h;W,H=w*math.exp(dw),h*math.exp(dh)
    return [cx-W/2,cy-H/2,cx+W/2,cy+H/2]


def fit(features,targets):
    import torch
    from torch.nn import functional as F
    if features.ndim!=2 or targets.shape!=(len(features),4) or not len(features):raise ValueError('Box-regression shapes mismatch')
    if not torch.isfinite(features).all() or not torch.isfinite(targets).all():raise ValueError('Nonfinite regression features/targets')
    torch.manual_seed(0);head=torch.nn.Linear(features.shape[1],4)
    torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.001)
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True);loss=F.smooth_l1_loss(head(features),targets)
        if not torch.isfinite(loss):raise ValueError('Nonfinite regression loss')
        loss.backward();optimizer.step()
    return head.eval().requires_grad_(False)
