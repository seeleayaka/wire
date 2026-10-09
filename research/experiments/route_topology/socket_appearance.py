"""Reference socket appearance novelty, not unplugged/electrical fault verdicts.

Ridge normal statistics are fit only on normal training photos. A disjoint
normal calibration subset determines the conformal rank; inspection names and
fault labels never enter descriptors, fitting or threshold selection.
"""
import cv2
import numpy as np

POLICY={'grid':[8,4],'covariance_shrinkage':.1,'diagonal_floor':1e-6,
    'appearance_candidate_pvalue_max':.05,'classification':'appearance_novelty_only',
    'wire_route_outside_socket_not_a_feature':True}


def descriptor(rgb):
    rgb=np.asarray(rgb)
    if rgb.shape!=(50,100,3) or rgb.dtype!=np.uint8:raise ValueError('registered100x50 uint8 RGB socket patch required')
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float64)/255
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float64)/255
    dx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)/8
    dy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)/8
    magnitude=np.minimum(1,np.hypot(dx,dy));angle=np.arctan2(dy,dx)%np.pi
    result=[]
    for y in range(4):
        for x in range(8):
            ys=slice(y*50//4,(y+1)*50//4);xs=slice(x*100//8,(x+1)*100//8)
            region=lab[ys,xs]
            result.extend(region.mean(axis=(0,1)));result.extend(region.std(axis=(0,1)))
            for k in range(4):result.append(float((magnitude[ys,xs]*((angle[ys,xs]>=k*np.pi/4)&
                (angle[ys,xs]<(k+1)*np.pi/4))).mean()))
    return np.asarray(result,np.float64)


def fit_normal(features):
    f=np.asarray(features,np.float64)
    if f.ndim!=2 or f.shape[1]!=320 or len(f)<20 or not np.isfinite(f).all():
        raise ValueError('at least20 finite normal socket descriptors required')
    center=f.mean(axis=0);cov=np.cov(f,rowvar=False)
    ridge=max(float(np.trace(cov)/len(center)),1e-6)
    covariance=.9*cov+.1*ridge*np.eye(len(center))
    inverse=np.linalg.inv(covariance)
    return {'center':center,'precision':inverse,'fit_count':len(f),'policy':dict(POLICY)}


def score(model,features):
    f=np.asarray(features,np.float64)
    if f.shape!=(320,) or not np.isfinite(f).all():raise ValueError('invalid descriptor')
    diff=f-model['center'];return float(np.sqrt(max(0,float(diff@model['precision']@diff))))


def novelty(model,calibration_scores,features):
    calibration=np.asarray(calibration_scores,np.float64)
    if calibration.ndim!=1 or len(calibration)<20 or not np.isfinite(calibration).all():
        raise ValueError('disjoint normal calibration scores required')
    value=score(model,features);p=float((1+(calibration>=value).sum())/(1+len(calibration)))
    return {'appearance_score':value,'normal_calibration_tail_rank':p,
        'appearance_difference_candidate':p<=.05,'plug_absent_confirmed':False,
        'port_identity_confirmed':False,'electrical_continuity':'not_assessed',
        'decision':'insufficient_evidence','new_confirmed_connections':0}
