"""Train-only sample policy. Never used to select inference/evaluation crops."""
import hashlib
from collections import defaultdict

POLICY=dict(sizes=[640,960],seeds_per_class_per_source=1,edge_margin=16,
            maximum_hard_negative_crops=64,hard_negative_crop_size=640,
            maximum_per_negative_kind={'normal':24,'damaged':16,'misrouted':16,'disconnected':8},
            preserve_all_original_train_crops=True,preserve_all_original_inner_val_crops=True)

def stable_key(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()

def selected_seeds(name,ports):
    selected=[]
    for cls in (0,1):
        rows=[p for p in ports if p['class_id']==cls]
        rows.sort(key=lambda p:(stable_key(name+'|'+str(p['source_box_index'])),p['source_box_index']))
        selected.extend(rows[:1])
    return selected

def labels_for_window(ports,window,shape):
    """Refuse any visible partial/edge-cut target, rather than silently erasing it."""
    from inspection_agent.port_tiling import near_artificial_edge
    h,w=shape;x,y,r,b=window;labels=[]
    for p in ports:
        l,t,rr,bb=p['box_xyxy']
        if min(r,rr)<=max(x,l) or min(b,bb)<=max(y,t):continue
        local=[l-x,t-y,rr-x,bb-y]
        if not (x<=l<rr<=r and y<=t<bb<=b):return None
        if near_artificial_edge(local,window,w,h,16):return None
        labels.append(dict(class_id=p['class_id'],source_box_index=p['source_box_index'],box_xyxy_local=local))
    return labels

def positive_window(seed,ports,size,shape):
    h,w=shape;l,t,r,b=seed['box_xyxy'];cx=(l+r)/2;cy=(t+b)/2;possibilities=[]
    for dx,dy in ((0,0),(-.125,0),(.125,0),(0,-.125),(0,.125),(-.125,-.125),(.125,.125),(-.125,.125),(.125,-.125)):
        x=max(0,min(w-size,round(cx+dx*size-size/2)));y=max(0,min(h-size,round(cy+dy*size-size/2)))
        window=[x,y,x+size,y+size];labels=labels_for_window(ports,window,shape)
        if labels is not None and any(p['source_box_index']==seed['source_box_index'] for p in labels):
            possibilities.append((len(labels),window,labels))
    if not possibilities:return None
    # Max complete targets, then fixed geometry order; no model/holdout feedback.
    possibilities.sort(key=lambda p:(-p[0],*p[1]))
    _,window,labels=possibilities[0];return dict(window=window,labels=labels)

def hard_negative_selection(rows):
    choices=sorted(rows,key=lambda p:(-p['teacher_score'],stable_key(p['source_image']),p['tile_id']))
    result=[];sources=set();counts=defaultdict(int)
    for p in choices:
        kind=p['source_image'].split('_')[0]
        if p['source_image'] in sources or counts[kind]>=POLICY['maximum_per_negative_kind'][kind]:continue
        result.append(p);sources.add(p['source_image']);counts[kind]+=1
        if len(result)==POLICY['maximum_hard_negative_crops']:break
    return result
