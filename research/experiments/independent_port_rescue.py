"""Experimental, default-off additive port cues; never a fault verdict."""
import copy
import math
import numpy as np
from inspection_agent.port_crop_gui_bridge import run_gui_port_review
from inspection_agent.port_tiling import box_iou, select_additional_tile_hint, tile_windows, merge_tiled_ports
from inspection_agent.optional_port_crop_review import REFERENCE_SHA, CONFIG, sha


def select_rescue(roi, existing, source_predictions, reference_predictions):
    """One bounded novel model cue; reference absence is not proof of change."""
    keys=('left','top','right','bottom')
    coordinates=lambda p:[p[k] for k in keys]
    strong_reference=[p for p in reference_predictions if p['confidence']>.25]
    novel=[];static=0
    for p in source_predictions:
        if any(p['class_id']==q['class_id'] and box_iou(coordinates(p),coordinates(q))>=.5 for q in strong_reference):
            static+=1
        else:novel.append(p)
    selected=select_additional_tile_hint([roi],existing,novel,.25)
    hints=selected['tile_hints']
    for hint in hints:
        hint['parent_index']=None
        hint['role']='independent_port_model_cue_manual_review'
        hint['reference_evidence']='no_matching_strong_reference_model_cue'
        hint['warning']='Model cue only; reference non-detection is not physical change, seating or continuity proof.'
    return hints,dict(static_reference_suppressed=static,selection=selected['selection_audit'])


def run_rescue(report, *, project, reference_predictions, roi, enabled=False,
               scene='unknown', prediction_provider=None):
    original=copy.deepcopy(report.get('review_regions',[]))
    existing=copy.deepcopy(report.get('existing_port_hints',[]))
    result=dict(status='disabled',fallback_reason=None,parents=original,existing_hints=existing,
                rescue_hints=[],automatic_fault_verdict=False,experimental=True)
    if not enabled:return result
    result['status']='fallback'
    try:
        if (not isinstance(roi,dict) or any(type(roi.get(k)) not in (int,float) or not math.isfinite(roi[k])
                for k in ('left','top','right','bottom')) or
                not (0<=roi['left']<roi['right']<=3648 and 0<=roi['top']<roi['bottom']<=2736)):
            raise ValueError('invalid_review_roi')
        if reference_predictions.get('source_sha256')!=REFERENCE_SHA or sha(report['reference'])!=REFERENCE_SHA:
            raise ValueError('reference_prediction_identity_mismatch')
        if reference_predictions.get('source_shape')!=[2736,3648]:raise ValueError('reference_prediction_shape_mismatch')
        if reference_predictions.get('windows')!=[list(w) for w in tile_windows(3648,2736)]:raise ValueError('reference_prediction_windows_mismatch')
        if merge_tiled_ports(reference_predictions['edge_kept_predictions'],CONFIG['cross_tile_nms_iou'])!=reference_predictions['merged_predictions']:
            raise ValueError('reference_prediction_merge_mismatch')
        proxy=copy.deepcopy(report);proxy['review_regions']=[copy.deepcopy(roi)]
        port=run_gui_port_review(proxy,project=project,enabled=True,scene=scene,prediction_provider=prediction_provider)
        if port['status']!='applied':
            result['fallback_reason']=port['fallback_reason'];return result
        refs=[dict(zip(('left','top','right','bottom'),p['box_xyxy']),class_id=p['class_id'],confidence=p['confidence'])
              for p in reference_predictions['merged_predictions']]
        hints,audit=select_rescue(roi,existing,port['aligned_predictions'],refs)
        result.update(status='applied',rescue_hints=hints,audit=audit,aligned_predictions=port['aligned_predictions'],
                      source_sha256=port['source_sha256'],reference_sha256=port['reference_sha256'])
    except (ValueError,KeyError,TypeError,OSError) as error:
        result['fallback_reason']=str(error)
    return result
