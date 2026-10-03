"""Strict source-box parsing and train-only, image-grouped crop planning."""
from __future__ import annotations
import hashlib
import math
from inspection_agent.port_tiling import tile_windows, near_artificial_edge


def parse_boxes(text, width, height):
    if type(width) is not int or type(height) is not int or min(width,height)<1:
        raise ValueError('positive image dimensions required')
    result=[]
    for number,line in enumerate(text.splitlines(),1):
        if not line.strip():continue
        fields=line.split()
        if len(fields)!=5:raise ValueError(f'line {number}: expected class cx cy w h')
        values=list(map(float,fields))
        if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite label')
        cls,cx,cy,bw,bh=values
        if cls!=int(cls) or int(cls) not in (1,2,3,4):raise ValueError('invalid source class')
        if not (0<=cx<=1 and 0<=cy<=1 and 0<bw<=1 and 0<bh<=1):raise ValueError('invalid normalized box')
        raw=[cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2]
        bounds=[max(0.,raw[0]),max(0.,raw[1]),min(1.,raw[2]),min(1.,raw[3])]
        if bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:raise ValueError('clipped empty box')
        result.append({'line':number,'source_class':int(cls),'normalized':bounds,
                       'box_xyxy':[bounds[0]*width,bounds[1]*height,bounds[2]*width,bounds[3]*height],
                       'clipped':raw!=bounds})
    return result


def rectangle_labels(boxes):
    result=[]
    for box in boxes:
        if box['source_class'] not in (3,4):continue
        l,t,r,b=box['normalized'];cls=box['source_class']-3
        result.append(f'{cls} {l:.6f} {t:.6f} {r:.6f} {t:.6f} {r:.6f} {b:.6f} {l:.6f} {b:.6f}')
    return result


def inner_split(names, seed='port-audit-20260929'):
    if len(names)!=len(set(names)):raise ValueError('duplicate source image')
    kinds={name.split('_')[0] for name in names}
    if not kinds<= {'damaged','disconnected','misrouted','normal'}:raise ValueError('unknown image kind')
    result={}
    for kind in sorted(kinds):
        group=sorted((n for n in names if n.split('_')[0]==kind),
                     key=lambda n:(hashlib.sha256((seed+'|'+n).encode()).hexdigest(),n))
        holdout=max(1,len(group)//5) if len(group)>=2 else 0
        for index,name in enumerate(group):result[name]='inner_val' if index<holdout else 'inner_train'
    return result


def plan_tiles(boxes,width,height):
    """Reject any tile with a cut/edge-guard port, never erase its visible label.

    A port-negative tile is not a claim that all assembly components are normal.
    This planner is for classes 3/4 only; damage/routing labels are not its targets.
    """
    ports=[(i,b) for i,b in enumerate(boxes) if b['source_class'] in (3,4)]
    result=[]
    for tile_id,window in enumerate(tile_windows(width,height)):
        x,y,r,b=window;complete=[];unsafe=[]
        for index,port in ports:
            l,t,rr,bb=port['box_xyxy']
            if min(rr,r)<=max(l,x) or min(bb,b)<=max(t,y):continue
            local=[l-x,t-y,rr-x,bb-y]
            contained=x<=l<rr<=r and y<=t<bb<=b
            if not contained or near_artificial_edge(local,window,width,height):
                unsafe.append(index)
            else:
                complete.append({'source_box_index':index,'class_id':port['source_class']-3,
                                 'box_xyxy_local':local})
        result.append({'tile_id':tile_id,'window':list(window),'labels':complete,
                       'excluded_cut_or_edge_ports':unsafe,'usable':not unsafe})
    return result
