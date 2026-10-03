"""Fixed weak seed selection, independent strong two-context center decoding."""
import copy
from inspection_agent.teacher_student_port_support import valid
from inspection_agent.context_port_recheck import complete
from inspection_agent.port_tiling import box_iou


def seeds_from_cases(current,cases):
    if len(current['primary'])>5 or len(current['all_predictions'])-len(current['primary'])>5:
        raise ValueError('invalid current budget')
    if len(current['all_predictions'])-len(current['primary'])==5:return []
    source=cases[0]['source_sha256'];shape=cases[0]['predictions']['source_shape']
    if not source or any(c['source_sha256']!=source or c['predictions']['source_shape']!=shape for c in cases):
        raise ValueError('source mismatch')
    pool=[copy.deepcopy(p) for c in cases for p in c['predictions']['merged_predictions'] if valid(p) and
        p['confidence']>.25 and complete(p,shape) and not any(box_iou(p['box_xyxy'],o['box_xyxy'])>=.5 for o in current['all_predictions'])]
    pool.sort(key=lambda p:(-p['confidence'],*p['box_xyxy'],p['class_id']));selected=[]
    for p in pool:
        if any(p['class_id']==o['class_id'] and box_iou(p['box_xyxy'],o['box_xyxy'])>=.5 for o in selected):continue
        selected.append(p)
        if len(selected)>=6:break
    return selected


def context_windows(seed,shape):
    h,w=shape
    if min(h,w)<960:raise ValueError('source too small for fixed context')
    l,t,r,b=seed['box_xyxy'];cx,cy=(l+r)/2,(t+b)/2;windows=[]
    for size in (640,960):
        x=max(0,min(w-size,round(cx-size/2)));y=max(0,min(h-size,round(cy-size/2)))
        windows.append([x,y,x+size,y+size])
    return windows


def append_confirmed(current,entries,shape,head_sha):
    out=copy.deepcopy(current);out['dense_context_additions']=[]
    remaining=5-(len(current['all_predictions'])-len(current['primary']))
    if len(current['primary'])>5 or remaining<0:raise ValueError('invalid current budget')
    choices=[]
    for entry in entries:
        if len(entry['views'])!=2 or entry['windows']!=context_windows(entry['seed'],shape):raise ValueError('view provenance mismatch')
        seed=entry['seed'];eligible=[]
        for view in entry['views']:
            eligible.append([p for p in view if valid(p) and p['class_id']==seed['class_id'] and p['confidence']>.75 and
                complete(p,shape) and box_iou(p['box_xyxy'],seed['box_xyxy'])>=.5])
        pairs=[(min(a['confidence'],b['confidence']),a,b) for a in eligible[0] for b in eligible[1] if box_iou(a['box_xyxy'],b['box_xyxy'])>=.5]
        if not pairs:continue
        _,a,b=max(pairs,key=lambda v:(v[0],v[1]['confidence'],*v[1]['box_xyxy']))
        row=copy.deepcopy(a if a['confidence']>=b['confidence'] else b)
        row.update(evidence_tier='dense_seed_context_experimental_manual_review',dense_head_sha256=head_sha,
            context_scores=[a['confidence'],b['confidence']],context_windows=entry['windows'],seed=copy.deepcopy(seed),automatic_fault_verdict=False)
        choices.append(row)
    choices.sort(key=lambda p:(-p['confidence'],*p['box_xyxy'],p['class_id']))
    for p in choices:
        if len(out['dense_context_additions'])>=remaining:break
        if any(box_iou(p['box_xyxy'],o['box_xyxy'])>=.5 for o in out['all_predictions']):continue
        out['dense_context_additions'].append(p);out['all_predictions'].append(p)
    assert out['all_predictions'][:len(current['all_predictions'])]==current['all_predictions']
    assert len(out['all_predictions'])<=len(current['primary'])+5
    return out
