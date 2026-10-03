"""Translation/scale-invariant box shape, no IDs or absolute position."""
import math
import torch


def crop_signature(box,scale):
    l,t,r,b=map(float,box);cx,cy=(l+r)/2,(t+b)/2
    side=max(24.,max(r-l,b-t)*scale)
    return (math.floor(cx-side/2),math.floor(cy-side/2),math.ceil(cx+side/2),math.ceil(cy+side/2))


def shape_vector(box):
    l,t,r,b=map(float,box);w,h=r-l,b-t
    if not all(math.isfinite(v) for v in (l,t,r,b)) or min(w,h)<=0:raise ValueError('invalid box shape')
    side=max(w,h)
    return [w/side,h/side,w*h/(side*side)]


def shaped_features(features,boxes):
    if features.shape!=(len(boxes),6144) or not torch.isfinite(features).all():raise ValueError('invalid paired shape input')
    values=torch.tensor([shape_vector(box) for box in boxes],dtype=features.dtype).reshape(len(boxes),3)
    return torch.cat((features,values),dim=1)


def aspect_examples(box):
    l,t,r,b=map(float,box);w,h=r-l,b-t;cx,cy=(l+r)/2,(t+b)/2
    if min(w,h)<=0:raise ValueError('invalid training box')
    variants=[]
    for fraction in (.25,.5):
        if w>=h:variants.append([l,cy-h*fraction/2,r,cy+h*fraction/2])
        else:variants.append([cx-w*fraction/2,t,cx+w*fraction/2,b])
    if w>=h:variants.append([l,cy-w/2,r,cy+w/2])
    else:variants.append([cx-h/2,t,cx+h/2,b])
    return variants
