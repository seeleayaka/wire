"""Generic reference-only local socket spatial cue, not mask-derived."""
import math
import numpy as np


def positive_box(anchor_box,inspection_to_reference,crop):
    l,t,r,b=anchor_box;pad=max(r-l,b-t)*.5
    inverse=np.linalg.inv(np.asarray(inspection_to_reference,float));points=[]
    for x,y in [(l-pad,t-pad),(r+pad,t-pad),(r+pad,b+pad),(l-pad,b+pad)]:
        q=inverse@np.array([x,y,1.])
        if not np.isfinite(q).all() or q[2]<=1e-9:raise ValueError('invalid box mapping')
        points.append(q[:2]/q[2])
    box=[math.floor(min(p[0] for p in points)),math.floor(min(p[1] for p in points)),
         math.ceil(max(p[0] for p in points)),math.ceil(max(p[1] for p in points))]
    if not crop[0]<=box[0]<box[2]<=crop[2] or not crop[1]<=box[1]<box[3]<=crop[3]:
        raise ValueError('positive box outside frozen context; no clipping')
    w,h=crop[2]-crop[0],crop[3]-crop[1]
    return box,[(box[0]+box[2]-2*crop[0])/(2*w),(box[1]+box[3]-2*crop[1])/(2*h),(box[2]-box[0])/w,(box[3]-box[1])/h]
