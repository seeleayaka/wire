"""Two new local-component research descriptors, not electrical topology.

HSV native colour components and DINOv2 token-cluster component distributions.
The latter is ComAD-inspired, not the official ComAD implementation (DINOv1/CRF).
No gap closing, object identity assumptions, or hidden-contact inference.
"""
import cv2
import numpy as np

COLOR_NAMES=['red','yellow','green','blue','purple','dark','bright_neutral','mid_neutral']
POLICY=dict(hsv_saturation_min=80,hsv_value_min=40,grid=[4,2],
            semantic_clusters=12,random_state=0,head_l2=.02,head_steps=1000,
            head_learning_rate=.05,class_tail_alpha=.05,source_wrong_max=0,
            source_coverage_min=.75,no_component_closing=True)

def color_components(rgb):
    rgb=np.asarray(rgb)
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<4:
        raise ValueError('finite actual uint8 RGB image required')
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);h,s,v=np.moveaxis(hsv,-1,0)
    chromatic=(s>=80)&(v>=40)
    categories=np.full(h.shape,7,np.uint8)
    categories[(v<80)&~chromatic]=5
    categories[(v>=160)&~chromatic]=6
    for index,region in enumerate([(h<15)|(h>=165),(h>=15)&(h<45),
                                  (h>=45)&(h<90),(h>=90)&(h<140),(h>=140)&(h<165)]):
        categories[chromatic&region]=index
    return categories

def color_descriptor(rgb):
    categories=color_components(rgb);height,width=categories.shape;features=[]
    for y in range(2):
        for x in range(4):
            region=categories[y*height//2:(y+1)*height//2,x*width//4:(x+1)*width//4]
            features.extend(np.bincount(region.ravel(),minlength=8)/region.size)
    for k in range(8):
        mask=categories==k;count,parts=cv2.connectedComponents(mask.astype(np.uint8),connectivity=8)
        if count==1:features.extend([0.]*8);continue
        sizes=np.bincount(parts.ravel());sizes[0]=0;active=parts==sizes.argmax();ys,xs=np.nonzero(active)
        features.extend([float(mask.mean()),float(active.mean()),float((xs+.5).mean()/width),
            float((ys+.5).mean()/height),float((xs.max()-xs.min()+1)/width),
            float((ys.max()-ys.min()+1)/height),float((ys>=height*.65).mean()),
            float((count-1)/mask.size)])
    result=np.asarray(features,np.float64)
    if result.shape!=(128,) or not np.isfinite(result).all():raise ValueError('invalid component descriptor')
    return result

def semantic_descriptor(tokens,centers,grid_hw=(8,16)):
    x=np.asarray(tokens,np.float64);centers=np.asarray(centers,np.float64)
    if x.ndim!=2 or centers.shape!=(12,x.shape[1]) or x.shape[0]!=np.prod(grid_hw) or not np.isfinite(x).all() or not np.isfinite(centers).all():
        raise ValueError('finite12-prototype source-bound token grid required')
    distances=((x[:,None]-centers[None])**2).sum(2)
    labels=distances.argmin(1).reshape(grid_hw);features=[]
    for region in [labels,labels[:4,:8],labels[:4,8:],labels[4:,:8],labels[4:,8:]]:
        features.extend(np.bincount(region.ravel(),minlength=12)/region.size)
    for k in range(12):
        selected=labels.ravel()==k
        features.append(float(np.sqrt(distances[selected,k]).mean()) if selected.any() else 0.)
    return np.asarray(features),labels

def fit_head(x,y):
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    if x.ndim!=2 or y.shape!=(len(x),) or not np.isfinite(x).all() or set(y)!={0.,1.} or min((y==k).sum() for k in [0,1])<20:
        raise ValueError('finite source features with20 fit originals per class required')
    center=x.mean(0);scale=np.maximum(x.std(0),.01);z=(x-center)/scale
    w=np.zeros(x.shape[1]);b=0.;weights=np.array([len(y)/(2*(y==k).sum()) for k in y])
    for _ in range(1000):
        error=(1/(1+np.exp(-np.clip(z@w+b,-40,40)))-y)*weights
        w-=.05*(z.T@error/len(y)+.02*w);b-=.05*error.mean()
    return dict(center=center,scale=scale,weights=w,bias=float(b))

def probability(model,x):
    x=np.asarray(x,np.float64)
    if x.shape!=model['weights'].shape or not np.isfinite(x).all():raise ValueError('feature dimensions/values differ')
    return float(1/(1+np.exp(-np.clip(((x-model['center'])/model['scale'])@model['weights']+model['bias'],-40,40))))

def prediction(model,x,calibration,excluded_id=None):
    p=probability(model,x);ranks={}
    for k in [0,1]:
        scores=[r['score'] for r in calibration[k] if r['id']!=excluded_id]
        minimum=19 if excluded_id is not None else 20
        if len(scores)<minimum:raise ValueError('insufficient disjoint class-tail calibration')
        candidate=p if k==0 else 1-p
        ranks[str(k)]=float((1+sum(s>=candidate for s in scores))/(1+len(scores)))
    admitted=[k for k in [0,1] if ranks[str(k)]>.05]
    return dict(visual_label_candidate=admitted[0] if len(admitted)==1 else None,
        class_tail_ranks=ranks,probability=p,leave_self_out=excluded_id is not None,
        decision='insufficient_evidence',new_confirmed_connections=0,electrical_continuity='not_assessed')

def gate(rows):
    resolved=[r for r in rows if r['prediction']['visual_label_candidate'] is not None]
    wrong=sum(r['visual_label']!=r['prediction']['visual_label_candidate'] for r in resolved)
    return dict(passed=wrong==0 and len(resolved)/len(rows)>=.75,wrong_singletons=wrong,
        singleton_count=len(resolved),total=len(rows),coverage=len(resolved)/len(rows),
        source_development_not_field_accuracy=True)
