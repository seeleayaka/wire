"""Frozen DINO patch features + a small center/box head, not a mask model.

Anchor-free center/box readout inspired by Objects as Points and FCOS.
Only bounding rectangles supervise this head: no true instance masks claimed.
"""
import math
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

INPUT=448
PATCH=14
GRID=INPUT//PATCH
EPOCHS=8
LR=.001
CROP_SCORE=.5
SOURCE_SCORE=.75
MIN_CROP_PRECISION=.90
MIN_CROP_RECALL=.25


def preprocess(image):
    if image.ndim!=3 or image.shape[2]!=3 or min(image.shape[:2])<1:raise ValueError('invalid BGR image')
    rgb=cv2.cvtColor(cv2.resize(image,(INPUT,INPUT),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
    tensor=torch.from_numpy(rgb).permute(2,0,1).float()/255.
    return (tensor-torch.tensor([.485,.456,.406]).view(3,1,1))/torch.tensor([.229,.224,.225]).view(3,1,1)


def frozen_features(encoder,images):
    with torch.inference_mode():
        output=encoder.forward_features(torch.stack([preprocess(im) for im in images]))
        tokens=output['x_norm_patchtokens']
        if tokens.shape!=(len(images),GRID*GRID,384) or not torch.isfinite(tokens).all():raise ValueError('DINO patch contract mismatch')
        return F.normalize(tokens,dim=-1).transpose(1,2).reshape(-1,384,GRID,GRID).contiguous().cpu().to(torch.float16)


def boxes_from_polygon_labels(text,width,height):
    boxes=[]
    for line in text.splitlines():
        values=list(map(float,line.split()))
        if not values:continue
        cls=values[0]
        if cls not in (0,1) or not np.isfinite(values).all():raise ValueError('invalid training label')
        coords=values[1:]
        if len(coords)==4:
            cx,cy,w,h=coords;l,t,r,b=cx-w/2,cy-h/2,cx+w/2,cy+h/2
        elif len(coords)>=6 and len(coords)%2==0:
            l,t,r,b=min(coords[::2]),min(coords[1::2]),max(coords[::2]),max(coords[1::2])
        else:raise ValueError('invalid polygon label')
        if not 0<=l<r<=1 or not 0<=t<b<=1:raise ValueError('box outside crop')
        boxes.append(dict(class_id=int(cls),box=[l*width,t*height,r*width,b*height]))
    return boxes


def encode_targets(boxes,width,height):
    heat=torch.zeros((2,GRID,GRID));reg=torch.zeros((2,4,GRID,GRID));mask=torch.zeros((2,GRID,GRID),dtype=torch.bool)
    owner_area={};collisions=0
    for row in sorted(boxes,key=lambda r:((r['box'][2]-r['box'][0])*(r['box'][3]-r['box'][1]),r['class_id'],*r['box'])):
        cls=row['class_id'];l,t,r,b=row['box']
        if cls not in (0,1) or not all(math.isfinite(v) for v in (l,t,r,b)) or not 0<=l<r<=width or not 0<=t<b<=height:
            raise ValueError('invalid target geometry')
        gx,gy=(l+r)*GRID/(2*width),(t+b)*GRID/(2*height)
        x,y=min(GRID-1,int(gx)),min(GRID-1,int(gy));key=(cls,y,x)
        if key in owner_area:collisions+=1;continue
        owner_area[key]=(r-l)*(b-t)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                yy,xx=y+dy,x+dx
                if 0<=yy<GRID and 0<=xx<GRID:heat[cls,yy,xx]=max(float(heat[cls,yy,xx]),math.exp(-(dx*dx+dy*dy)/2))
        reg[cls,:,y,x]=torch.tensor([gx-x-.5,gy-y-.5,math.log((r-l)*GRID/width),math.log((b-t)*GRID/height)])
        mask[cls,y,x]=True
    return dict(heat=heat,reg=reg,mask=mask,collisions=collisions)


class DensePortHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.context=nn.Sequential(nn.Conv2d(384,64,3,padding=1),nn.ReLU())
        self.heat=nn.Conv2d(64,2,1);self.box=nn.Conv2d(64,8,1)
        nn.init.constant_(self.heat.bias,-2.19)
        nn.init.zeros_(self.box.bias)
        with torch.no_grad():self.box.bias.reshape(2,4)[:,2:]=math.log(2.)
    def forward(self,features):
        hidden=self.context(features.float())
        return self.heat(hidden),self.box(hidden).reshape(-1,2,4,GRID,GRID)


def loss(outputs,targets):
    logits,pred_reg=outputs;heat,reg,mask=targets['heat'],targets['reg'],targets['mask']
    probability=logits.sigmoid().clamp(1e-5,1-1e-5)
    positive=heat.eq(1);negative=heat.lt(1)
    pos_loss=(probability.log()*(1-probability).pow(2)*positive).sum()
    neg_loss=((1-probability).log()*probability.pow(2)*(1-heat).pow(4)*negative).sum()
    heat_loss=-(pos_loss+neg_loss)/positive.sum().clamp(min=1)
    selected=mask.unsqueeze(2).expand_as(pred_reg)
    reg_loss=F.smooth_l1_loss(pred_reg[selected],reg[selected]) if mask.any() else pred_reg.sum()*0
    return heat_loss+reg_loss,dict(heat=float(heat_loss.detach()),box=float(reg_loss.detach()))


def decode(outputs,shapes,score=CROP_SCORE,max_boxes=100):
    logits,regs=outputs
    if logits.shape!=(len(shapes),2,GRID,GRID) or regs.shape!=(len(shapes),2,4,GRID,GRID):raise ValueError('output geometry mismatch')
    if not 0<score<1 or type(max_boxes) is not int or max_boxes<1:raise ValueError('invalid decode contract')
    if not torch.isfinite(logits).all() or not torch.isfinite(regs).all():raise ValueError('nonfinite dense output')
    probability=logits.sigmoid();peaks=probability.eq(F.max_pool2d(probability,3,1,1))&(probability>score)
    result=[]
    for index,(height,width) in enumerate(shapes):
        if min(height,width)<1:raise ValueError('invalid source shape')
        candidates=[]
        for cls,y,x in peaks[index].nonzero().tolist():
            dx,dy,lw,lh=regs[index,cls,:,y,x].detach().cpu().tolist()
            cx=(x+.5+max(-.5,min(.5,dx)))*width/GRID;cy=(y+.5+max(-.5,min(.5,dy)))*height/GRID
            bw=math.exp(max(-3,min(5,lw)))*width/GRID;bh=math.exp(max(-3,min(5,lh)))*height/GRID
            box=[cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2]
            if not 0<=box[0]<box[2]<=width or not 0<=box[1]<box[3]<=height:continue
            candidates.append(dict(class_id=cls,confidence=float(probability[index,cls,y,x]),box_xyxy=box))
        candidates.sort(key=lambda r:(-r['confidence'],*r['box_xyxy'],r['class_id']))
        selected=[]
        from inspection_agent.port_tiling import box_iou
        for row in candidates:
            if any(row['class_id']==p['class_id'] and box_iou(row['box_xyxy'],p['box_xyxy'])>=.5 for p in selected):continue
            selected.append(row)
            if len(selected)>=max_boxes:break
        result.append(selected)
    return result
