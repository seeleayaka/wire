"""GT-free registered body/ring contrast; no image IDs, labels or fitting."""
import cv2
import numpy as np
from robust_port_exposure import compensate

DIMENSIONS = 46


def _region_stats(a, b, keep):
    av, bv = a[keep], b[keep]
    delta = av-bv
    absolute = np.abs(delta)
    gray_a, gray_b = a.mean(2), b.mean(2)
    grad_a = cv2.Sobel(gray_a, cv2.CV_32F, 1, 0)+cv2.Sobel(gray_a, cv2.CV_32F, 0, 1)
    grad_b = cv2.Sobel(gray_b, cv2.CV_32F, 1, 0)+cv2.Sobel(gray_b, cv2.CV_32F, 0, 1)
    # Eroded support avoids invalid/grid-boundary derivatives.
    interior = cv2.erode(keep.astype(np.uint8), np.ones((3,3),np.uint8)).astype(bool)
    if not interior.any():
        raise ValueError('Insufficient gradient support')
    return np.concatenate((absolute.mean(0), delta.mean(0), absolute.std(0),
        np.quantile(absolute, [.25,.5,.9]),
        np.array([(absolute>.05).mean(),(absolute>.1).mean(),(absolute>.2).mean()]),
        np.array([np.abs(grad_a-grad_b)[interior].mean(),av.std(),bv.std()])))


def _grid(observed, expected, valid, box, scale):
    l,t,r,b = map(float, box)
    cx,cy=(l+r)/2,(t+b)/2
    w,h=(r-l)*scale,(b-t)*scale
    x=cx-w/2+(np.arange(64,dtype=np.float32)+.5)*w/64-.5
    y=cy-h/2+(np.arange(64,dtype=np.float32)+.5)*h/64-.5
    xx,yy=np.meshgrid(x,y)
    a=cv2.remap(observed,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).astype(np.float32)/255
    e=cv2.remap(expected,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).astype(np.float32)/255
    mask=cv2.remap(valid.astype(np.uint8),xx,yy,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0
    geometry=(xx>=-.5)&(yy>=-.5)&(xx<observed.shape[1]-.5)&(yy<observed.shape[0]-.5)
    keep=mask&geometry
    domain=np.ones((64,64),bool) if scale==1 else ~((xx+.5>=l)&(xx+.5<r)&(yy+.5>=t)&(yy+.5<b))
    coverage=float((keep&domain).sum())/int(domain.sum())
    return a,e,keep&domain,coverage


def descriptor(normalized_observed, expected, valid, box):
    if normalized_observed.dtype!=np.uint8 or expected.dtype!=np.uint8 or normalized_observed.shape!=expected.shape or normalized_observed.ndim!=3 or normalized_observed.shape[2]!=3:
        raise ValueError('Equal uint8 three-channel images required')
    if valid.shape!=expected.shape[:2] or not np.isfinite(valid).all():
        raise ValueError('Invalid registered coverage')
    box=np.asarray(box,dtype=np.float64)
    if box.shape!=(4,) or not np.isfinite(box).all() or min(box[2:]-box[:2])<2:
        raise ValueError('Invalid component box')
    a,b,core,cc=_grid(normalized_observed,expected,valid,box,1)
    ra,rb,ring,rc=_grid(normalized_observed,expected,valid,box,3)
    if cc<.85 or rc<.85 or core.sum()<16 or ring.sum()<64:
        raise ValueError('Insufficient valid body/ring coverage')
    abs_core=np.abs(a-b);abs_ring=np.abs(ra-rb)
    quadrants=[]
    for y,x in ((0,0),(0,32),(32,0),(32,32)):
        mask=core[y:y+32,x:x+32]
        if not mask.any():raise ValueError('Empty component quadrant')
        quadrants.append(abs_core[y:y+32,x:x+32][mask].mean())
    result=np.concatenate((_region_stats(a,b,core),_region_stats(ra,rb,ring),
        quadrants,abs_core[core].mean(0)-abs_ring[ring].mean(0),
        [np.log((box[2]-box[0])/(box[3]-box[1])),cc,rc])).astype(np.float32)
    if result.shape!=(DIMENSIONS,) or not np.isfinite(result).all():
        raise ValueError('Invalid descriptor')
    return result


def normalize_pair(observed, expected, valid):
    return compensate(observed, expected, valid)


def fold_standardization(train, evaluation):
    import torch
    if train.ndim!=2 or train.shape[1]!=DIMENSIONS or evaluation.ndim!=2 or evaluation.shape[1]!=DIMENSIONS or len(train)==0:
        raise ValueError('Invalid descriptor matrix')
    if not torch.isfinite(train).all() or not torch.isfinite(evaluation).all():
        raise ValueError('Nonfinite descriptor matrix')
    mean=train.mean(0);std=train.std(0,unbiased=False).clamp_min(.01)
    return ((train-mean)/std).clamp(-5,5),((evaluation-mean)/std).clamp(-5,5),mean,std
