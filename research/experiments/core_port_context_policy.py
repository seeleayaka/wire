"""Alternative: retain trained scale/context, recenter the same low-score seeds."""
import copy
from core_port_zoom_policy import POLICY as ZOOM_POLICY,proposals,select_zoom,confirmed
POLICY=copy.deepcopy(ZOOM_POLICY)
POLICY.update(crop_size=1280,center_offsets=[[-160,-160],[160,160]],
              rationale='Retain original training-scale context, only shift tile coverage')

def windows(proposal,shape):
    height,width=shape;size=POLICY['crop_size']
    if min(height,width)<size:raise ValueError('image_too_small_for_fixed_context')
    l,t,r,b=proposal['box_xyxy'];cx=(l+r)/2;cy=(t+b)/2;result=[]
    for dx,dy in POLICY['center_offsets']:
        x=max(0,min(width-size,round(cx+dx-size/2)))
        y=max(0,min(height-size,round(cy+dy-size/2)))
        result.append([x,y,x+size,y+size])
    return result
