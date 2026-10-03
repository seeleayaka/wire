"""Optional stronger supplementary cues; preserve the five primary cues."""
import copy
from pathlib import Path
from inspection_agent.precision_port_rescue import run_precision_port_rescue,FRAME_MARGIN
from inspection_agent.independent_port_rescue import select_rescue
from inspection_agent.port_tiling import box_iou

POLICY_ID='primary5_supplement5_multitile_v1_20261002'


def expand_consensus_result(result, *, supplementary_enabled=False):
    output=copy.deepcopy(result)
    output['supplementary_hints']=[]
    output['supplementary_policy']=dict(policy_id=POLICY_ID,enabled=bool(supplementary_enabled),
        maximum_additional_hints=5,minimum_score=.75,strong_vote_threshold=.5,
        minimum_distinct_strong_tiles=2,support_iou=.5,primary_hints_unchanged=True,
        additional_review_burden_explicit=True,scores_are_probabilities=False,
        automatic_fault_verdict=False)
    if not supplementary_enabled or result['status']!='applied':return output
    source=result['source_evidence'];native=source['predictions'];aligned=source['aligned_predictions']
    if len(native['merged_predictions'])!=len(aligned):
        raise ValueError('native_aligned_prediction_count_mismatch')
    height,width=native['source_shape'];candidates=[];support={}
    coordinates=lambda p:[p[k] for k in ('left','top','right','bottom')]
    identity=lambda p:(p['class_id'],*coordinates(p))
    for raw,transformed in zip(native['merged_predictions'],aligned):
        l,t,r,b=raw['box_xyxy']
        if raw['confidence']<=.75 or not (l>FRAME_MARGIN and t>FRAME_MARGIN and r<width-FRAME_MARGIN and b<height-FRAME_MARGIN):continue
        votes=sorted({p['source_tile'] for p in native['edge_kept_predictions']
            if p['confidence']>.5 and p['class_id']==raw['class_id']
            and box_iou(p['box_xyxy'],raw['box_xyxy'])>=.5})
        if len(votes)<2:continue
        candidates.append(transformed);support[identity(transformed)]=votes
    original=copy.deepcopy(result['existing_hints'])+copy.deepcopy(result['rescue_hints'])
    added=[]
    for _ in range(5):
        remaining=[p for p in candidates if not any(box_iou(coordinates(p),coordinates(h['box']))>=.5 for h in original+added)]
        selected,_=select_rescue(result['analysis_rois'],original+added,remaining,result['reference_evidence']['aligned_predictions'])
        if not selected:break
        for hint in selected:
            hint.update(evidence_tier='supplementary_multi_tile_model_cue',strong_support_tiles=support[identity(hint['box'])])
        added.extend(selected)
    output['supplementary_hints']=added
    output['supplementary_policy'].update(additional_hints=len(added),raw_supported_candidates=len(candidates))
    return output


def run_consensus_port_rescue(report, *, project, enabled=False, scene='unknown', supplementary_enabled=False):
    primary=run_precision_port_rescue(report,project=project,enabled=enabled,scene=scene)
    try:
        return expand_consensus_result(primary,supplementary_enabled=supplementary_enabled)
    except Exception as error:
        # An optional supplement failure must never discard verified primary cues.
        output=expand_consensus_result(primary,supplementary_enabled=False)
        output['supplementary_policy'].update(requested=bool(supplementary_enabled),
            fallback_reason=type(error).__name__+': '+str(error))
        return output


def render_consensus_overlay(aligned_path,result,output_path):
    """Existing parents/primary colors; supplementary boxes are purple, separately numbered."""
    import cv2
    from inspection_agent.port_crop_gui_bridge import render_port_overlay
    from inspection_agent.optional_port_crop_review import read_image
    output_path=Path(output_path)
    render_port_overlay(aligned_path,dict(parents=result['parents'],tile_hints=result['rescue_hints']),output_path)
    image=read_image(output_path)
    for index,hint in enumerate(result.get('supplementary_hints',[]),1):
        l,t,r,b=[round(hint['box'][k]) for k in ('left','top','right','bottom')]
        cv2.rectangle(image,(l,t),(r,b),(180,70,180),5)
        cv2.putText(image,f'Extra {index}',(l,max(20,t-8)),cv2.FONT_HERSHEY_SIMPLEX,.8,(180,70,180),2)
    ok,encoded=cv2.imencode(output_path.suffix,image)
    if not ok:raise ValueError('cannot_encode_supplementary_overlay')
    encoded.tofile(str(output_path))

