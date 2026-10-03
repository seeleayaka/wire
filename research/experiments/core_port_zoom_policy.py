"""Bounded, label-free proposal zoom. Experimental, not a production switch."""
import copy
import math
from core_port_supplement_policy import supplement, complete
from core_port_precision_policy import iou

POLICY = dict(maximum_proposals=6, crop_size=640, center_offsets=[[-80, -80], [80, 80]],
              prediction_imgsz=960, proposal_maximum_score=.5, proposal_maximum_side=160,
              proposal_minimum_score=.001, proposal_overlap_iou=.25, agreement_iou=.5,
              strict_score=.75, standard_score=.5, standard_maximum_score=.75,
              maximum_total_supplementary=5, maximum_total_primary=5,
              crop_edge_margin=16, source_edge_margin=16)


def ordered(rows):
    return sorted(rows, key=lambda p: (-p['confidence'], *p['box_xyxy'], p['class_id']))


def proposals(raw):
    existing = supplement(raw)['all_predictions']
    result = []
    for p in ordered(raw['merged_predictions']):
        l,t,r,b = p['box_xyxy']
        if not all(math.isfinite(v) for v in (*p['box_xyxy'], p['confidence'])):continue
        if not (POLICY['proposal_minimum_score'] <= p['confidence'] <= .5):continue
        if p['class_id'] not in (0,1) or not 0 < min(r-l,b-t) <= max(r-l,b-t) <= 160:continue
        if not complete(p, raw['source_shape']):continue
        if any(iou(p['box_xyxy'], q['box_xyxy']) >= .5 for q in existing+result):continue
        result.append(copy.deepcopy(p))
        if len(result) == POLICY['maximum_proposals']:break
    return result


def windows(proposal, shape):
    height,width=shape;size=POLICY['crop_size']
    if min(height,width)<size:raise ValueError('image_too_small_for_fixed_zoom')
    l,t,r,b=proposal['box_xyxy'];cx=(l+r)/2;cy=(t+b)/2
    result=[]
    for dx,dy in POLICY['center_offsets']:
        x=max(0,min(width-size,round(cx+dx-size/2)))
        y=max(0,min(height-size,round(cy+dy-size/2)))
        result.append([x,y,x+size,y+size])
    return result


def confirmed(case, mode):
    if mode not in ('strict','standard'):raise ValueError('unknown_zoom_policy')
    candidates=[];shape=case['predictions']['source_shape']
    for entry in case['zoom_evidence']:
        if entry['windows'][0] == entry['windows'][1]:continue # same crop is not two votes
        seed=entry['proposal']
        threshold=POLICY['strict_score'] if mode=='strict' else POLICY['standard_score']
        eligible=[]
        for view in entry['views']:
            eligible.append([p for p in view if p['confidence']>threshold and
                p['class_id']==seed['class_id'] and complete(p,shape) and
                iou(p['box_xyxy'],seed['box_xyxy'])>=POLICY['proposal_overlap_iou']])
        pairs=[]
        for a in eligible[0]:
            for b in eligible[1]:
                if iou(a['box_xyxy'],b['box_xyxy'])<POLICY['agreement_iou']:continue
                if max(a['confidence'],b['confidence'])<=POLICY['standard_maximum_score']:continue
                pairs.append((min(a['confidence'],b['confidence']),a,b))
        if not pairs:continue
        _,a,b=max(pairs,key=lambda pair:pair[0])
        # Keep a real predicted box rather than an annotation-fitted average.
        chosen=copy.deepcopy(a if a['confidence']>=b['confidence'] else b)
        chosen.update(zoom_view_scores=[a['confidence'],b['confidence']],
                      zoom_seed=copy.deepcopy(seed), zoom_windows=entry['windows'])
        candidates.append(chosen)
    return ordered(candidates)


def select_zoom(case, mode):
    base=supplement(case['predictions']);selected=[]
    remaining=POLICY['maximum_total_supplementary']-len(base['supplementary'])
    for p in confirmed(case,mode):
        if len(selected)>=remaining:break
        if any(iou(p['box_xyxy'],q['box_xyxy'])>=.5 for q in base['all_predictions']+selected):continue
        selected.append(p)
    return dict(primary=base['primary'],supplementary=base['supplementary'],zoom=selected,
                all_predictions=base['all_predictions']+selected)
