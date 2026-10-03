"""Experimental symmetric support graph using the EXISTING strict cue cutoffs.

Change which distinct checkpoint may provide weak geometry support, not scores,
windows, labels or shared review capacity. Preserve all current V3 cues.
"""
import copy
from inspection_agent.teacher_student_port_support import native_selection,valid
from inspection_agent.context_port_recheck import complete
from inspection_agent.port_tiling import box_iou

POLICY=dict(policy_id='symmetric_student_feature_strict_support_graph_20261003',
            candidate_score=.75,peer_score=.25,cross_model_iou=.5,
            tile_score=.5,two_recheck_scores=.75,minimum_supporting_tiles=2,
            maximum_primary=5,maximum_extra=5,preserve_all_current_cues=True,
            same_cutoffs_as_existing_complementary_candidates=True,
            only_change='Allow either distinct student/feature checkpoint to provide geometry support instead of requiring oldest teacher',
            class_agnostic=True,automatic_fault_verdict=False,independent_physical_evidence=False)


def potential_peer_support(case):
    # The peer may support a strong candidate unseen by the oldest teacher.
    # Return the full weak geometry pool only to decide whether fresh missing
    # feature inference is necessary; this is NOT final candidate eligibility.
    return [p for p in case['predictions']['merged_predictions'] if valid(p) and p['confidence']>.25]


def strict_rows(case):
    raw=case['predictions'];selected=native_selection(case);rows=[]
    for row in selected['all_predictions']:
        if not valid(row) or row['confidence']<=.75 or not complete(row,raw['source_shape']):continue
        votes=sorted({p['source_tile'] for p in raw['edge_kept_predictions'] if valid(p) and
                      type(p.get('source_tile')) is int and p['confidence']>.5 and p['class_id']==row['class_id'] and
                      box_iou(p['box_xyxy'],row['box_xyxy'])>=.5})
        scores=row.get('zoom_view_scores',[]);windows=row.get('zoom_windows',[])
        double=(row in selected['zoom'] and len(scores)==len(windows)==2 and min(scores)>.75 and windows[0]!=windows[1])
        if len(votes)<2 and not double:continue
        candidate=copy.deepcopy(row);candidate.update(graph_support_tiles=votes,graph_double_recheck=double);rows.append(candidate)
    return rows


def append_support_graph(current,candidate,peer):
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
                supports=[p for p in potential_peer_support(supporter) if p['class_id']==row['class_id'] and
                          box_iou(p['box_xyxy'],row['box_xyxy'])>=.5]
                if not supports:continue
                best=max(supports,key=lambda p:(p['confidence'],box_iou(p['box_xyxy'],row['box_xyxy'])))
                addition=copy.deepcopy(row);addition.update(evidence_tier='symmetric_support_graph_experimental_manual_review',
                    graph_policy_id=POLICY['policy_id'],candidate_weight_sha256=model['weight_sha256'],peer_weight_sha256=supporter['weight_sha256'],
                    peer_support_score=best['confidence'],peer_support_box=best['box_xyxy'],automatic_fault_verdict=False,
                    correlated_model_evidence=True)
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
