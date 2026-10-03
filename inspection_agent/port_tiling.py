"""Generic image tiling, cut-edge rejection and conservative port deduplication."""
from __future__ import annotations
import copy
import math
from inspection_agent.port_state_hint import select_port_state_hints


def axis_starts(length, size=1280, stride=960):
    if any(type(v) is not int or v<1 for v in (length,size,stride)) or stride>size:
        raise ValueError('positive integer dimensions and stride<=size required')
    last=max(0,length-size); starts=list(range(0,last+1,stride))
    if starts[-1]!=last:starts.append(last)
    return starts


def tile_windows(width,height,size=1280,stride=960):
    return [(x,y,min(width,x+size),min(height,y+size))
            for y in axis_starts(height,size,stride) for x in axis_starts(width,size,stride)]


def near_artificial_edge(box,window,width,height,margin=16):
    x,y,r,b=window;tw,th=r-x,b-y
    l,t,rr,bb=box
    return ((x>0 and l<=margin) or (y>0 and t<=margin)
            or (r<width and rr>=tw-margin) or (b<height and bb>=th-margin))


def box_iou(a,b):
    l,t,r,bb=a;ll,tt,rr,bbb=b
    intersection=max(0,min(r,rr)-max(l,ll))*max(0,min(bb,bbb)-max(t,tt))
    area=(r-l)*(bb-t)+(rr-ll)*(bbb-tt)-intersection
    return intersection/area if area>0 else 0.


def merge_tiled_ports(predictions,threshold=.5):
    if not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or not 0<threshold<=1:
        raise ValueError('invalid NMS IoU')
    pending=copy.deepcopy(predictions)
    for p in pending:
        if type(p.get('class_id')) is not int or p['class_id'] not in (0,1):raise ValueError('invalid class')
        box=p.get('box_xyxy')
        if not isinstance(box,list) or len(box)!=4 or not all(math.isfinite(v) for v in box):raise ValueError('invalid box')
        if not 0<=box[0]<box[2] or not 0<=box[1]<box[3]:raise ValueError('invalid box')
        if not math.isfinite(p['confidence']) or not 0<=p['confidence']<=1:raise ValueError('invalid score')
    pending.sort(key=lambda p:(-p['confidence'],*p['box_xyxy'],p['class_id'],p['source_tile']))
    result=[]
    while pending:
        selected=pending.pop(0);tiles={selected['source_tile']};remaining=[]
        for other in pending:
            if other['class_id']==selected['class_id'] and box_iou(selected['box_xyxy'],other['box_xyxy'])>=threshold:
                tiles.add(other['source_tile'])
            else:remaining.append(other)
        selected['support_tiles']=sorted(tiles);result.append(selected);pending=remaining
    return result


def select_additional_tile_hint(parents,existing_hints,predictions,threshold):
    selected=select_port_state_hints(parents,predictions,threshold)
    coords=lambda b:[b[k] for k in ('left','top','right','bottom')]
    choices=[h for h in selected['hints'] if not any(
        box_iou(coords(h['box']),coords(old['box']))>=.5 for old in existing_hints)]
    choices.sort(key=lambda h:(-h['box']['confidence'],
        (h['box']['right']-h['box']['left'])*(h['box']['bottom']-h['box']['top']),
        *coords(h['box']),h['box']['class_id'],h['parent_index']))
    return {'parents':copy.deepcopy(parents),'existing_hints':copy.deepcopy(existing_hints),
            'tile_hints':copy.deepcopy(choices[:1]),'selection_audit':selected['selection_audit']}
