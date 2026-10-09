"""Exact pixel permutation, not new evidence, interpolation or mask fusion."""
import numpy as np


def forward(rgb):
    a=np.asarray(rgb)
    if a.ndim!=3 or a.shape[2]!=3 or a.dtype!=np.uint8:raise ValueError('uint8 RGB required')
    return np.rot90(a,1).copy()


def restore(mask,original_size):
    a=np.asarray(mask);w,h=original_size
    if a.ndim!=2 or a.shape!=(w,h):raise ValueError('mask must match rotated source dimensions')
    return np.rot90(a,-1).copy()


def prompt(box):
    if len(box)!=4 or any(type(x) not in [float,int] or not np.isfinite(x) for x in box):
        raise ValueError('finite normalized center/width/height required')
    x,y,w,h=box
    if not (w>0 and h>0 and 0<=x-w/2<x+w/2<=1 and 0<=y-h/2<y+h/2<=1):
        raise ValueError('prompt outside image')
    return [y,1-x,h,w]
