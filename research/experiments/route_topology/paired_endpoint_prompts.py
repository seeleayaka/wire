"""Two location prompts from verified local poses; no added connection pixels."""
import numpy as np

def endpoint_box(anchor,pose,crop):
    if not pose['localization_proposal_supported'] or not all(pose['gates'].values()):
        raise ValueError('unsupported local anchor cannot produce a prompt')
    h=np.asarray(pose['inspection_to_reference_local'],float)
    if h.shape!=(3,3) or not np.isfinite(h).all() or np.linalg.matrix_rank(h)!=3:
        raise ValueError('qualified invertible pose required')
    l,t,r,b=anchor['bbox_xyxy'];points=np.array([[l,t,1],[r,t,1],[r,b,1],[l,b,1]],float)@np.linalg.inv(h).T
    if np.any(np.abs(points[:,2])<1e-9) or not (np.all(points[:,2]>0) or np.all(points[:,2]<0)):
        raise ValueError('anchor projection crosses horizon')
    xy=points[:,:2]/points[:,2:];xy-=crop[:2]
    a,c=xy.min(0);d,f=xy.max(0);w=crop[2]-crop[0];height=crop[3]-crop[1]
    if not (w>0 and height>0 and 0<=a<d<=w and 0<=c<f<=height):
        raise ValueError('projected anchor outside original crop; no clipping')
    return [float((a+d)/(2*w)),float((c+f)/(2*height)),float((d-a)/w),float((f-c)/height)]
