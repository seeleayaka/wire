"""Train-only frozen DINO crop semantics; no image IDs/coordinates as features."""
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F

CONTEXT_SCALES=(1.5,3.0)
PROBABILITY_GATE=.98
MIN_PRECISION=.98
MIN_RECALL=.25

def context_box(box,scale):
    l,t,r,b=map(float,box);cx,cy=(l+r)/2,(t+b)/2
    side=max(24.,max(r-l,b-t)*scale)
    return [cx-side/2,cy-side/2,cx+side/2,cy+side/2]

def coverage(a,b):
    left,top=max(a[0],b[0]),max(a[1],b[1]);right,bottom=min(a[2],b[2]),min(a[3],b[3])
    area=max(0,right-left)*max(0,bottom-top)
    return area/max(1e-9,min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1])))

def crop_tensor(image,box,scale):
    l,t,r,b=map(float,box)
    if not np.isfinite([l,t,r,b]).all() or r<=l or b<=t:raise ValueError('invalid crop box')
    cx,cy=(l+r)/2,(t+b)/2;side=max(24.,max(r-l,b-t)*scale)
    left,top=math.floor(cx-side/2),math.floor(cy-side/2);right,bottom=math.ceil(cx+side/2),math.ceil(cy+side/2)
    h,w=image.shape[:2]
    if left>=w or top>=h or right<=0 or bottom<=0:raise ValueError('crop outside image')
    padded=cv2.copyMakeBorder(image[max(0,top):min(h,bottom),max(0,left):min(w,right)],
        max(0,-top),max(0,bottom-h),max(0,-left),max(0,right-w),cv2.BORDER_CONSTANT,value=(124,116,104))
    rgb=cv2.cvtColor(cv2.resize(padded,(224,224),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
    tensor=torch.from_numpy(rgb).permute(2,0,1).float()/255.
    return (tensor-torch.tensor([.485,.456,.406]).view(3,1,1))/torch.tensor([.229,.224,.225]).view(3,1,1)

def embeddings(model,image,boxes):
    tensors=[crop_tensor(image,box,scale) for box in boxes for scale in CONTEXT_SCALES]
    vectors=[]
    with torch.inference_mode():
        for start in range(0,len(tensors),8):
            result=model.forward_features(torch.stack(tensors[start:start+8]))
            patches=result['x_norm_patchtokens'].reshape(-1,16,16,384)
            cls=F.normalize(result['x_norm_clstoken'],dim=1)
            center=F.normalize(patches[:,6:10,6:10].mean(dim=(1,2)),dim=1)
            vectors.append(torch.cat((cls,center),dim=1))
    if not vectors:return torch.empty((0,1536))
    return torch.cat(vectors).reshape(len(boxes),1536)/2.

def fit_head(features,labels):
    torch.manual_seed(0)
    head=torch.nn.Linear(features.shape[1],3)
    torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
    counts=torch.bincount(labels,minlength=3).float()
    if (counts==0).any():raise ValueError('Training fold lacks a class')
    weights=counts.sum()/(3*counts)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.001)
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(head(features),labels,weight=weights).backward();optimizer.step()
    return head.eval()

def precision_gate(probabilities,labels):
    scores,classes=probabilities.max(dim=1);accepted=(scores>=PROBABILITY_GATE)&(classes>0)
    correct=accepted&(classes==labels);positive=labels>0
    tp=int(correct.sum());fp=int((accepted&~correct).sum());targets=int(positive.sum())
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/targets if targets else 0.
    return dict(tp=tp,fp=fp,targets=targets,precision=precision,recall=recall,
        qualifies=precision>=MIN_PRECISION and recall>=MIN_RECALL)
