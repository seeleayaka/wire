"""Validated score selector over the frozen, default-off port rescue backend.

Scores are model scores, not calibrated probabilities or electrical verdicts.
Original parents/existing cues stay intact. Only independent rescue cues change.
"""
import copy
from inspection_agent.bounded_port_rescue import expand_verified_result
from inspection_agent.independent_port_rescue import run_independent_port_rescue

POLICY_ID = 'high_score_complete_frame_max5_v1_20261002'
FRAME_MARGIN = 16


def precision_verified_result(result):
    """Retain at most five independently gated cues with scores strictly >0.5."""
    verified = copy.deepcopy(result)
    original_count = len(verified['rescue_hints'])
    frame_suppressed = 0
    if verified['status'] == 'applied':
        native = result['source_evidence']['predictions']
        aligned = result['source_evidence']['aligned_predictions']
        if len(native['merged_predictions']) != len(aligned):
            raise ValueError('native_aligned_prediction_count_mismatch')
        height, width = native['source_shape']
        identity = lambda p: (p['class_id'],p['left'],p['top'],p['right'],p['bottom'])
        complete = []
        for raw, transformed in zip(native['merged_predictions'],aligned):
            l,t,r,b = raw['box_xyxy']
            if l > FRAME_MARGIN and t > FRAME_MARGIN and r < width-FRAME_MARGIN and b < height-FRAME_MARGIN:
                complete.append(transformed)
        frame_suppressed = len(aligned)-len(complete)
        eligible = {identity(p) for p in complete}
        verified['source_evidence']['aligned_predictions'] = complete
        verified['rescue_hints'] = [h for h in verified['rescue_hints']
            if h['box']['confidence'] > .5 and identity(h['box']) in eligible]
    suppressed = original_count-len(verified['rescue_hints'])
    # Remove a weak first cue before expansion so it cannot consume a high-score slot.
    output = expand_verified_result(verified)
    if result['status'] == 'applied':
        # Retain complete unfiltered prediction provenance; record the selection separately.
        output['source_evidence'] = copy.deepcopy(result['source_evidence'])
    output['budget_policy'].update(policy_id=POLICY_ID, minimum_hint_score=.5,
        comparison='strictly_greater', first_hint_retained=False, original_top1_retained=False,
        suppressed_weak_hints=suppressed,
        suppressed_native_edge_candidates=frame_suppressed,
        native_frame_margin=FRAME_MARGIN, all_four_edges=True,
        scores_are_probabilities=False, original_regions_preserved=True,
        warning='Incomplete genuine ports near the frame border also abstain.')
    return output


def run_precision_port_rescue(report, *, project, enabled=False, scene='unknown'):
    """No prediction-file input: fresh source/reference through original gates."""
    original = run_independent_port_rescue(report, project=project, enabled=enabled, scene=scene)
    return precision_verified_result(original)
