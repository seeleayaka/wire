"""Default-off contextual confirmations, within the existing five-extra budget.

Same-photo model agreement is not independent physical/electrical evidence.
Fresh matched-position reference checks add a conservative static-cue veto.
"""
import copy
import math
from pathlib import Path
from inspection_agent.consensus_port_rescue import run_consensus_port_rescue, render_consensus_overlay
from inspection_agent.independent_port_rescue import select_rescue
from inspection_agent.port_tiling import box_iou, near_artificial_edge
from inspection_agent.optional_port_crop_review import aligned_predictions, read_image, sha

POLICY_ID='primary5_supplement5_context_recheck_v1_20261002'

def complete(p,shape):
    h,w=shape;l,t,r,b=p['box_xyxy']
    return l>16 and t>16 and r<w-16 and b<h-16

def native_baseline(raw):
    """Exact source-only selector used in the fixed development evaluations."""
    rows=[p for p in raw['merged_predictions'] if p['confidence']>.25]
    rows.sort(key=lambda p:(-p['confidence'],(p['box_xyxy'][2]-p['box_xyxy'][0])*(p['box_xyxy'][3]-p['box_xyxy'][1]),*p['box_xyxy'],p['class_id']))
    primary=(rows[:1]+[p for p in rows[1:] if p['confidence']>.5][:4]) if rows else []
    primary=[p for p in primary if p['confidence']>.5 and complete(p,raw['source_shape'])]
    extra=[]
    for p in rows:
        if p['confidence']<=.75 or not complete(p,raw['source_shape']):continue
        if any(box_iou(p['box_xyxy'],q['box_xyxy'])>=.5 for q in primary+extra):continue
        votes={q['source_tile'] for q in raw['edge_kept_predictions'] if q['confidence']>.5 and q['class_id']==p['class_id'] and box_iou(q['box_xyxy'],p['box_xyxy'])>=.5}
        if len(votes)<2:continue
        extra.append(p)
        if len(extra)==5:break
    return copy.deepcopy(primary+extra)

def recheck_proposals(raw):
    existing=native_baseline(raw);result=[]
    rows=sorted(raw['merged_predictions'],key=lambda p:(-p['confidence'],*p['box_xyxy'],p['class_id']))
    for p in rows:
        l,t,r,b=p['box_xyxy']
        if not all(math.isfinite(v) for v in (*p['box_xyxy'],p['confidence'])):continue
        if not .5<p['confidence']<=1 or p['class_id'] not in (0,1):continue
        if not 0<min(r-l,b-t)<=max(r-l,b-t)<=160 or not complete(p,raw['source_shape']):continue
        if any(box_iou(p['box_xyxy'],q['box_xyxy'])>=.5 for q in existing+result):continue
        result.append(copy.deepcopy(p))
        if len(result)==6:break
    return result

def recheck_windows(seed,shape):
    h,w=shape
    if min(h,w)<1280:raise ValueError('image_too_small_for_context')
    l,t,r,b=seed['box_xyxy'];cx=(l+r)/2;cy=(t+b)/2;result=[]
    for dx,dy in ((-160,-160),(160,160)):
        x=max(0,min(w-1280,round(cx+dx-640)));y=max(0,min(h-1280,round(cy+dy-640)))
        result.append([x,y,x+1280,y+1280])
    return result

def confirm_rechecks(entries,shape):
    result=[]
    for e in entries:
        if e['windows'][0]==e['windows'][1]:continue
        seed=e['proposal'];eligible=[]
        for view in e['views']:
            eligible.append([p for p in view if p['confidence']>.75 and p['class_id']==seed['class_id'] and complete(p,shape) and box_iou(p['box_xyxy'],seed['box_xyxy'])>=.25])
        pairs=[(min(a['confidence'],b['confidence']),a,b) for a in eligible[0] for b in eligible[1] if box_iou(a['box_xyxy'],b['box_xyxy'])>=.5]
        if not pairs:continue
        _,a,b=max(pairs,key=lambda v:v[0]);chosen=copy.deepcopy(a if a['confidence']>=b['confidence'] else b)
        chosen.update(zoom_view_scores=[a['confidence'],b['confidence']],zoom_seed=copy.deepcopy(seed),zoom_windows=e['windows'])
        result.append(chosen)
    return sorted(result,key=lambda p:(-p['confidence'],*p['box_xyxy'],p['class_id']))

def predict_seed_views(model,image,seeds):
    import torch
    shape=list(image.shape[:2]);h,w=shape;entries=[]
    for seed in seeds:
        windows=recheck_windows(seed,shape);views=[]
        crops=[image[y:b,x:r] for x,y,r,b in windows]
        outputs=model.predict(crops,imgsz=960,conf=.001,iou=.7,max_det=300,device='cpu',verbose=False,save=False)
        torch.set_num_threads(4)
        if len(outputs)!=2:raise ValueError('recheck_batch_mismatch')
        for index,(output,window) in enumerate(zip(outputs,windows)):
            x,y,_,_=window;rows=[]
            for box in output.boxes:
                local=list(map(float,box.xyxy[0].tolist()))
                if near_artificial_edge(local,window,w,h,16):continue
                l,t,r,b=local
                rows.append(dict(box_xyxy=[l+x,t+y,r+x,b+y],confidence=float(box.conf.item()),class_id=int(box.cls.item()),zoom_view=index))
            views.append(rows)
        entries.append(dict(proposal=copy.deepcopy(seed),windows=windows,views=views))
    return entries

def append_verified_rechecks(result,native_candidates,reference_rows,matrix):
    """Existing primary/extra cues remain unchanged, even on recheck failure."""
    output=copy.deepcopy(result);source=result['source_evidence'];shape=source['predictions']['source_shape']
    refshape=result['reference_evidence']['predictions']['source_shape']
    candidates=copy.deepcopy(native_candidates)
    for row in candidates:row['support_tiles']=[]
    transformed=aligned_predictions(candidates,matrix,shape,refshape)
    support={(r['class_id'],r['left'],r['top'],r['right'],r['bottom']):n for n,r in zip(candidates,transformed)}
    references=copy.deepcopy(result['reference_evidence']['aligned_predictions'])+copy.deepcopy(reference_rows)
    added=[];old=result['existing_hints']+result['rescue_hints']+result['supplementary_hints']
    coords=lambda p:[p[k] for k in ('left','top','right','bottom')]
    for _ in range(5-len(result['supplementary_hints'])):
        remaining=[p for p in transformed if not any(box_iou(coords(p),coords(h['box']))>=.5 for h in old+added)]
        hints,_=select_rescue(result['analysis_rois'],old+added,remaining,references)
        if not hints:break
        for hint in hints:
            box=hint['box'];raw=support[(box['class_id'],*coords(box))]
            hint.update(evidence_tier='contextual_double_crop_model_cue_manual_review',context_view_scores=raw['zoom_view_scores'],
                        warning='Two crops of one image are correlated; reference non-detection does not prove electrical change.')
        added.extend(hints)
    output['supplementary_hints'].extend(added)
    output['recheck_policy'].update(added_hints=len(added),confirmed_source_candidates=len(candidates))
    output['supplementary_policy'].update(additional_hints=len(output['supplementary_hints']),policy_id=POLICY_ID)
    return output

def run_context_port_recheck(report,*,project,enabled=False,scene='unknown',supplementary_enabled=False):
    original=run_consensus_port_rescue(report,project=project,enabled=enabled,scene=scene,supplementary_enabled=supplementary_enabled)
    original['recheck_policy']=dict(policy_id=POLICY_ID,enabled=bool(supplementary_enabled),maximum_proposals=6,
        seed_score=.5,both_view_score=.75,crop_size=1280,predict_imgsz=960,agreement_iou=.5,
        maximum_total_primary=5,maximum_total_supplementary=5,source_reference_fresh=original['status']=='applied',
        same_photo_not_independent_physical_evidence=True,automatic_fault_verdict=False,added_hints=0)
    original.setdefault('supplementary_policy',{}).update(policy_id=POLICY_ID,context_recheck=bool(supplementary_enabled))
    if not supplementary_enabled or original['status']!='applied' or len(original['supplementary_hints'])>=5:return original
    try:
        import numpy as np
        source=original['source_evidence'];reference=original['reference_evidence']
        seeds=recheck_proposals(source['predictions'])
        if not seeds:
            original['recheck_policy']['proposals']=0;return original
        weight=Path(project)/'output/port_crop_training_fixed_20260929/full/runs/rectports/weights/best.pt'
        if sha(weight)!=source['weight_sha256'] or sha(report['inspection'])!=source['source_sha256'] or sha(report['reference'])!=source['reference_sha256']:raise ValueError('recheck_input_changed')
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4);torch.manual_seed(20260929);model=YOLO(str(weight))
        if model.task!='segment' or dict(model.names)!={0:'unplugged_plug',1:'unplugged_jack'}:raise ValueError('recheck_model_contract')
        image=read_image(report['inspection']);ref=read_image(report['reference'])
        source_entries=predict_seed_views(model,image,seeds)
        candidates=confirm_rechecks(source_entries,source['predictions']['source_shape'])
        matrix=np.asarray(report['alignment']['source_to_reference_homography'],dtype=np.float64)
        # Query the corresponding reference locations, not just its own low-score detections.
        refs=[]
        for row in candidates:row['support_tiles']=[]
        mapped=aligned_predictions(candidates,matrix,image.shape[:2],ref.shape[:2])
        ref_seeds=[dict(box_xyxy=[p[k] for k in ('left','top','right','bottom')],class_id=p['class_id'],confidence=p['confidence']) for p in mapped]
        reference_entries=predict_seed_views(model,ref,ref_seeds)
        for entry in reference_entries:
            for view in entry['views']:
                for p in view:
                    if p['confidence']>.25 and complete(p,ref.shape[:2]):
                        refs.append(dict(zip(('left','top','right','bottom'),p['box_xyxy']),class_id=p['class_id'],confidence=p['confidence'],valid_warp_fraction=1.,support_tiles=[]))
        if sha(weight)!=source['weight_sha256'] or sha(report['inspection'])!=source['source_sha256'] or sha(report['reference'])!=source['reference_sha256']:raise ValueError('recheck_input_changed')
        output=append_verified_rechecks(original,candidates,refs,matrix)
        output['recheck_evidence']=dict(source_views=source_entries,matched_reference_views=reference_entries,
                                      source_sha256=source['source_sha256'],reference_sha256=source['reference_sha256'],weight_sha256=source['weight_sha256'])
        return output
    except Exception as error:
        original['recheck_policy']['fallback_reason']=type(error).__name__+': '+str(error)
        return original
