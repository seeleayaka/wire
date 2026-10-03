"""Frozen cardinal direction filter, review only; no backward/gap inference."""
from collections import deque
import cv2
import numpy as np
from visible_entry_path import inspect_local_path


def inspect_directional_path(rgb, mask, bbox, forward_axis):
    if not isinstance(forward_axis, (list, tuple)) or len(forward_axis)!=2 or any(type(v) is not int for v in forward_axis) or tuple(forward_axis) not in ((0,1),(0,-1),(1,0),(-1,0)):
        raise ValueError('explicit cardinal forward axis required')
    baseline=inspect_local_path(rgb,mask,bbox)
    fx,fy=forward_axis;px,py=-fy,fx
    x1,y1,x2,y2=bbox
    # Symmetric local envelope makes declared orientation rotation-equivariant.
    lateral_width=(x2-x1) if fy else (y2-y1)
    axial_height=(y2-y1) if fy else (x2-x1)
    corners=np.array([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
    extended=np.concatenate([corners+np.array([px,py])*side*2*lateral_width+np.array([fx,fy])*end*axial_height
        for side in (-1,1) for end in (-1,6)])
    h,w=mask.shape
    left=max(0,int(extended[:,0].min()));right=min(w,int(extended[:,0].max()))
    top=max(0,int(extended[:,1].min()));bottom=min(h,int(extended[:,1].max()))
    target=np.asarray(mask[top:bottom,left:right])>0
    hsv=cv2.cvtColor(np.asarray(rgb[top:bottom,left:right]),cv2.COLOR_RGB2HSV)
    support=target|(cv2.inRange(hsv,np.array([90,80,30]),np.array([135,255,255]))>0)
    roi=np.zeros_like(support);roi[y1-top:y2-top,x1-left:x2-left]=True
    selected=[]
    for sy,sx in np.argwhere(roi&support):
        start=(int(sy),int(sx));queue=deque([start]);parents={start:None};costs={start:0};found=None
        while queue:
            y,x=queue.popleft()
            if target[y,x]:found=(y,x);break
            for dx,dy,lateral in ((fx,fy,0),(px,py,1),(-px,-py,1)):
                ny,nx=y+dy,x+dx
                if not (0<=ny<support.shape[0] and 0<=nx<support.shape[1] and support[ny,nx]):continue
                forward=(nx-int(sx))*fx+(ny-int(sy))*fy
                cost=costs[y,x]+lateral
                # Fixed scale-normalized budget, not tuned to real-image outcomes.
                if cost>.5*lateral_width+.5*forward or cost>=costs.get((ny,nx),float('inf')):continue
                costs[ny,nx]=cost;parents[ny,nx]=(y,x);queue.append((ny,nx))
        path=[]
        while found is not None:
            y,x=found;path.append([int(x+left),int(y+top)]);found=parents[found]
        if path:
            selected=list(reversed(path));break
    return dict(local_box_xyxy=[left,top,right,bottom],forward_axis=list(forward_axis),
        continuous_pixel_support=bool(selected),path_xy=selected,
        baseline_continuous_pixel_support=baseline['continuous_pixel_support'],
        confirmed_assignment=False,gap_filling_used=False,
        lateral_budget_rule='.5 * entrance transverse width + .5 * forward displacement',
        backward_steps_allowed=False)
