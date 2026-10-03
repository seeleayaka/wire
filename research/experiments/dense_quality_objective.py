"""Continuous joint score/IoU targets only in training; GT-free inference."""
import torch
import torch.nn.functional as F
from dense_pixel_port_probe import GRID


def decoded_grid_boxes(reg):
    if reg.ndim!=5 or reg.shape[1:]!=(2,4,GRID,GRID):raise ValueError('invalid box regression geometry')
    y,x=torch.meshgrid(torch.arange(GRID,device=reg.device,dtype=reg.dtype),torch.arange(GRID,device=reg.device,dtype=reg.dtype),indexing='ij')
    cx=x+.5+reg[:,:,0].clamp(-.5,.5);cy=y+.5+reg[:,:,1].clamp(-.5,.5)
    width=reg[:,:,2].clamp(-3,5).exp();height=reg[:,:,3].clamp(-3,5).exp()
    return torch.stack((cx-width/2,cy-height/2,cx+width/2,cy+height/2),dim=-1)


def iou_giou(a,b):
    intersection=(torch.minimum(a[...,2:],b[...,2:])-torch.maximum(a[...,:2],b[...,:2])).clamp(min=0).prod(-1)
    aa=(a[...,2:]-a[...,:2]).clamp(min=0).prod(-1);bb=(b[...,2:]-b[...,:2]).clamp(min=0).prod(-1)
    union=(aa+bb-intersection).clamp(min=1e-8);iou=intersection/union
    enclosure=(torch.maximum(a[...,2:],b[...,2:])-torch.minimum(a[...,:2],b[...,:2])).clamp(min=0).prod(-1).clamp(min=1e-8)
    return iou,iou-(enclosure-union)/enclosure


def loss(outputs,targets):
    logits,reg=outputs;heat=targets['heat'];mask=targets['mask'];true_reg=targets['reg']
    if logits.shape!=heat.shape or logits.shape[1:]!=(2,GRID,GRID) or mask.shape!=heat.shape:
        raise ValueError('invalid training targets')
    if not all(torch.isfinite(v).all() for v in (logits,reg,heat,true_reg)):raise ValueError('nonfinite training tensor')
    probability=logits.sigmoid();quality=torch.zeros_like(logits);n=mask.sum().clamp(min=1)
    if mask.any():
        pred_boxes=decoded_grid_boxes(reg)[mask];true_boxes=decoded_grid_boxes(true_reg)[mask]
        iou,giou=iou_giou(pred_boxes,true_boxes);quality[mask]=iou.detach().clamp(0,1)
        giou_loss=(1-giou).mean()
        selected=mask.unsqueeze(2).expand_as(reg);reg_loss=F.smooth_l1_loss(reg[selected],true_reg[selected])
    else:
        giou_loss=reg.sum()*0;reg_loss=reg.sum()*0
    focal=F.binary_cross_entropy_with_logits(logits,quality,reduction='none')*(probability-quality).abs().pow(2)
    weights=torch.where(mask,torch.ones_like(heat),(1-heat).pow(4))
    heat_loss=(focal*weights).sum()/n
    total=heat_loss+reg_loss+giou_loss
    return total,dict(heat=float(heat_loss.detach()),box=float(reg_loss.detach()),giou=float(giou_loss.detach()),
                      positive_quality_mean=float(quality[mask].mean()) if mask.any() else None)
