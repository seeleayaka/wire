"""Diagnostic blue-conductor support; never fills gaps or assigns wires."""
from collections import deque
import cv2
import numpy as np


def inspect_local_path(rgb, mask, bbox):
    rgb, mask = np.asarray(rgb), np.asarray(mask)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or mask.shape != rgb.shape[:2]:
        raise ValueError('uint8 RGB and same-frame 2D mask required')
    if len(bbox) != 4 or any(isinstance(x, bool) or not isinstance(x, int) for x in bbox):
        raise ValueError('integer pixel ROI required')
    x1,y1,x2,y2 = bbox; h,w = mask.shape
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError('ROI out of frame')
    # Freeze local scope and color gate before real-image evaluation.
    bw,bh=x2-x1,y2-y1
    left,top,right,bottom=max(0,x1-2*bw),max(0,y1-bh),min(w,x2+2*bw),min(h,y2+6*bh)
    local_rgb=rgb[top:bottom,left:right]
    hsv=cv2.cvtColor(local_rgb,cv2.COLOR_RGB2HSV)
    blue=cv2.inRange(hsv,np.array([90,80,30]),np.array([135,255,255]))>0
    target=mask[top:bottom,left:right]>0
    support=blue|target
    roi=np.zeros_like(support)
    roi[y1-top:y2-top,x1-left:x2-left]=True
    starts=np.argwhere(roi & support)
    parents={tuple(p):None for p in starts}; queue=deque(parents)
    found=None
    while queue:
        y,x=queue.popleft()
        if target[y,x]:
            found=(y,x);break
        # 4-neighbour support avoids corner-only diagonal jumps.
        for ny,nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0<=ny<support.shape[0] and 0<=nx<support.shape[1] and support[ny,nx] and (ny,nx) not in parents:
                parents[ny,nx]=(y,x);queue.append((ny,nx))
    path=[]
    while found is not None:
        y,x=found;path.append([int(x+left),int(y+top)]);found=parents[found]
    path.reverse()
    return dict(local_box_xyxy=[left,top,right,bottom], mask_roi_pixels=int((roi&target).sum()),
                blue_roi_pixels=int((roi&blue).sum()), continuous_pixel_support=bool(path),
                path_xy=path, added_blue_path_pixels=sum(not bool(mask[y,x]) for x,y in path),
                gap_filling_used=False, confirmed_assignment=False,
                claim_boundary='Color/mask connectivity is diagnostic, not cable identity or seating.')
