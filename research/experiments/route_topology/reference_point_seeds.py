"""Reference-only seed locations; prompts are not observed inspection facts."""
import cv2
import numpy as np


def select_points(raw, anchors, translation):
    raw=np.asarray(raw)
    if not np.isfinite(raw).all():raise ValueError('reference pixels must be finite')
    if len(translation)!=2 or any(type(v) is not int or v<0 for v in translation):
        raise ValueError('exact reference crop translation required')
    active=raw>0
    if active.ndim!=2 or not active.any():raise ValueError('nonempty reference native mask required')
    distance=cv2.distanceTransform(active.astype(np.uint8),cv2.DIST_L2,5)
    points=[]
    for anchor in anchors:
        l,t,r,b=anchor['bbox_xyxy']
        l,r=l-translation[0],r-translation[0]
        t,b=t-translation[1],b-translation[1]
        if not (0<=l<r<active.shape[1] and 0<=t<b<active.shape[0]):
            raise ValueError('reference anchor outside exact crop')
        ys,xs=np.nonzero(active[int(t):int(b)+1,int(l):int(r)+1])
        if not len(xs):raise ValueError('reference mask does not touch anchor')
        xs,ys=xs+int(l),ys+int(t)
        maximum=distance[ys,xs].max()
        choices=[(int(x),int(y)) for x,y in zip(xs,ys) if distance[y,x]==maximum]
        cx,cy=(l+r)/2,(t+b)/2
        point=min(choices,key=lambda q:((q[0]-cx)**2+(q[1]-cy)**2,q[1],q[0]))
        points.append({'anchor_id':anchor['id'],'xy_crop':list(point),
                       'xy_original':[point[0]+translation[0],point[1]+translation[1]],
                       'native_mask_interior_margin':float(maximum),'label':1})
    height,width=active.shape
    negatives=[[10,10],[width-11,10],[10,height-11],[width-11,height-11]]
    if min(height,width)<30 or any(active[y,x] for x,y in negatives):
        raise ValueError('fixed background seeds overlap reference object; do not retune')
    return points,negatives
