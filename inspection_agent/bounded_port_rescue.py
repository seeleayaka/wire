"""Explicit experimental high-score expansion; production budget-one stays default."""
import copy
from inspection_agent.independent_port_rescue import run_independent_port_rescue, select_rescue
from inspection_agent.port_tiling import box_iou

def expand_verified_result(result):
    """Keep the verified baseline cue; add <=4 cues scored >.5 under same gates."""
    output=copy.deepcopy(result)
    output['budget_policy']=dict(experimental=True,maximum_hints=5,extra_threshold=.5,
        original_top1_retained=True,automatic_fault_verdict=False)
    if result['status']!='applied':return output
    rois=result['analysis_rois']
    predictions=[p for p in result['source_evidence']['aligned_predictions'] if p['confidence']>.5]
    references=result['reference_evidence']['aligned_predictions']
    existing=copy.deepcopy(result['existing_hints'])
    hints=copy.deepcopy(result['rescue_hints'])
    for _ in range(5-len(hints)):
        coordinates=lambda b:[b[k] for k in ('left','top','right','bottom')]
        # Filter previously selected boxes BEFORE the original per-ROI top1 rank.
        remaining=[p for p in predictions if not any(box_iou(coordinates(p),coordinates(old['box']))>=.5
                    for old in existing+hints)]
        addition,_=select_rescue(rois,existing+hints,remaining,references)
        if not addition:break
        hints.extend(addition)
    output['rescue_hints']=hints
    output['budget_policy']['additional_hints']=len(hints)-len(result['rescue_hints'])
    return output

def run_bounded_port_rescue(report,*,project,enabled=False,scene='unknown'):
    """Fresh source/reference inference through the unchanged original safety gates."""
    original=run_independent_port_rescue(report,project=project,enabled=enabled,scene=scene)
    return expand_verified_result(original)
