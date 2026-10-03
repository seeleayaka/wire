"""Reuse the current loose-plug-only boundary for symmetric model support.

Exclude new empty jacks AFTER native strict selection, BEFORE shared slot
allocation. Existing jack cues remain exactly unchanged.
"""
import copy
from port_support_graph_policy import strict_rows,potential_peer_support,POLICY as GRAPH_POLICY
from inspection_agent.port_tiling import box_iou
POLICY=dict(GRAPH_POLICY,policy_id='symmetric_support_graph_loose_plugs_20261003',allowed_new_classes=[0],
            class_agnostic=False,current_loose_plug_boundary_reused=True,
            class_excluded_before_final_budget=True,native_selector_unchanged=True)


def potential_plug_support(case):
    return [p for p in potential_peer_support(case) if p['class_id']==0]


def append_graph_plugs(current,candidate,peer):
    output=copy.deepcopy(current);output.update(strong_consensus_additions=[],strong_consensus_fallback_reason=None)
    try:
        if (not candidate.get('source_sha256') or not candidate.get('weight_sha256') or not peer.get('weight_sha256') or
            candidate['weight_sha256']==peer['weight_sha256']):raise ValueError('distinct_checkpoint_identity_required')
        if any(candidate.get(k)!=peer.get(k) for k in ('image','source_sha256')):raise ValueError('source_identity_mismatch')
        if candidate['predictions']['source_shape']!=peer['predictions']['source_shape']:raise ValueError('source_geometry_mismatch')
        old=current['all_predictions'];remaining=5-(len(old)-len(current['primary']))
        if len(current['primary'])>5 or remaining<0:raise ValueError('current_budget_invalid')
        choices=[]
        for model,supporter in ((candidate,peer),(peer,candidate)):
            for row in strict_rows(model):
                if row['class_id']!=0:continue
                supports=[p for p in potential_plug_support(supporter) if box_iou(p['box_xyxy'],row['box_xyxy'])>=.5]
                if not supports:continue
                best=max(supports,key=lambda p:(p['confidence'],box_iou(p['box_xyxy'],row['box_xyxy'])))
                addition=copy.deepcopy(row);addition.update(evidence_tier='support_graph_loose_plug_experimental_manual_review',
                    graph_policy_id=POLICY['policy_id'],candidate_weight_sha256=model['weight_sha256'],peer_weight_sha256=supporter['weight_sha256'],
                    peer_support_score=best['confidence'],peer_support_box=best['box_xyxy'],automatic_fault_verdict=False,
                    correlated_model_evidence=True,loose_plug_only=True)
                choices.append(addition)
        choices.sort(key=lambda r:(-r['confidence'],*r['box_xyxy'],r['class_id'],r['candidate_weight_sha256']))
        for row in choices:
            if len(output['strong_consensus_additions'])>=remaining:break
            if any(box_iou(row['box_xyxy'],p['box_xyxy'])>=.5 for p in output['all_predictions']):continue
            output['strong_consensus_additions'].append(row);output['all_predictions'].append(row)
        assert output['all_predictions'][:len(old)]==old and len(output['all_predictions'])<=len(current['primary'])+5
        return output
    except (KeyError,TypeError,ValueError,AssertionError) as error:
        output=copy.deepcopy(current);output.update(strong_consensus_additions=[],strong_consensus_fallback_reason=type(error).__name__+': '+str(error))
        return output
