"""Reference-derived unwarped local context; never expand exact endpoints."""
import math
import numpy as np


def context_box(anchor_box, inspection_to_reference, size):
    l,t,r,b = [float(v) for v in anchor_box]
    if not l < r or not t < b: raise ValueError('invalid anchor')
    pad=max(r-l,b-t)
    corners=np.array([[l-pad,t-pad,1],[r+pad,t-pad,1],
                      [r+pad,b+pad,1],[l-pad,b+pad,1]])
    q=corners@np.linalg.inv(np.asarray(inspection_to_reference,float)).T
    if not np.isfinite(q).all() or (abs(q[:,2])<1e-9).any() or not (np.all(q[:,2]>0) or np.all(q[:,2]<0)):
        raise ValueError('invalid projective context')
    xy=q[:,:2]/q[:,2:]
    box=[math.floor(xy[:,0].min()),math.floor(xy[:,1].min()),math.ceil(xy[:,0].max()),math.ceil(xy[:,1].max())]
    if not 0 <= box[0] < box[2] <= size[0] or not 0 <= box[1] < box[3] <= size[1]:
        raise ValueError('context outside image; no clipping')
    return box


def native_coverage(rgb, region, raw, score):
    import cv2
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    bins=hsv[:,:,0].astype(int)//10
    valid=(hsv[:,:,1]>=64)&(hsv[:,:,2]>=32)
    groups=[[0,1,2,3,16,17],[9,10,11]]
    hits=[int((valid & np.isin(bins,g) & raw & region).sum()) for g in groups]
    total=[int((valid & np.isin(bins,g) & raw).sum()) for g in groups]
    return dict(score=float(score),exact_CPU_color_hits=hits,context_color_hits=total,
                local_two_color_coverage_candidate=bool(score>=.75 and min(hits)>=8),
                physical_identity_confirmed=False,electrical_continuity='not_assessed')
