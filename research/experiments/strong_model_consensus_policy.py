"""Experimental high-score cross-model cue; no teacher-required support.

Freeze this rule before grading. Two related checkpoints/two same-photo views
are correlated model evidence, never electrical or physical verification.
"""
import copy
import math
from inspection_agent.teacher_student_port_support import valid
from inspection_agent.context_port_recheck import complete, confirm_rechecks
from inspection_agent.port_tiling import box_iou

POLICY = dict(candidate_score=.9, peer_score=.9, cross_model_iou=.65,
              view_score=.75, view_iou=.5, maximum_primary=5, maximum_extra=5,
              preserve_all_current_cues=True, both_classes=True, automatic_fault_verdict=False,
              independent_physical_evidence=False, teacher_support_not_required=True)


def consistent_rows(case):
    raw = case['predictions']
    result = []
    for row in raw['merged_predictions'] + confirm_rechecks(case['zoom_evidence'], raw['source_shape']):
        if not valid(row) or row['confidence'] <= POLICY['candidate_score'] or not complete(row, raw['source_shape']):
            continue
        votes = sorted({p['source_tile'] for p in raw['edge_kept_predictions']
                        if valid(p) and type(p.get('source_tile')) is int and p['confidence'] > POLICY['view_score']
                        and p['class_id'] == row['class_id'] and box_iou(p['box_xyxy'], row['box_xyxy']) >= POLICY['view_iou']})
        scores = row.get('zoom_view_scores', [])
        windows = row.get('zoom_windows', [])
        double_crop = (len(scores) == len(windows) == 2 and windows[0] != windows[1] and
                       all(type(s) in (int, float) and math.isfinite(s) and POLICY['view_score'] < s <= 1 for s in scores))
        if len(votes) < 2 and not double_crop:
            continue
        selected = copy.deepcopy(row)
        selected.update(consensus_support_tiles=votes, consensus_double_crop=double_crop)
        result.append(selected)
    return result


def append_strong_consensus(current, candidate, peer):
    output = copy.deepcopy(current)
    output.update(strong_consensus_additions=[], strong_consensus_fallback_reason=None)
    try:
        if (not candidate.get('source_sha256') or not candidate.get('weight_sha256') or
                not peer.get('weight_sha256') or candidate['weight_sha256'] == peer['weight_sha256']):
            raise ValueError('distinct_checkpoint_identity_required')
        if any(candidate.get(k) != peer.get(k) for k in ('image', 'source_sha256')):
            raise ValueError('source_identity_mismatch')
        if candidate['predictions']['source_shape'] != peer['predictions']['source_shape']:
            raise ValueError('source_geometry_mismatch')
        old = current['all_predictions']
        remaining = 5 - (len(old) - len(current['primary']))
        if len(current['primary']) > 5 or remaining < 0:
            raise ValueError('current_budget_invalid')
        # No labels, filename patterns, source indices, or coordinate-specific
        # rules enter candidate eligibility or ordering.
        choices = []
        peers = consistent_rows(peer)
        for row in consistent_rows(candidate):
            supporting = [p for p in peers if p['class_id'] == row['class_id'] and
                          p['confidence'] > POLICY['peer_score'] and
                          box_iou(p['box_xyxy'], row['box_xyxy']) >= POLICY['cross_model_iou']]
            if not supporting:
                continue
            best = max(supporting, key=lambda p: (p['confidence'], box_iou(p['box_xyxy'], row['box_xyxy'])))
            addition = copy.deepcopy(row)
            addition.update(evidence_tier='strong_model_consensus_experimental_manual_review',
                            peer_confidence=best['confidence'], peer_box_xyxy=best['box_xyxy'],
                            peer_support_tiles=best['consensus_support_tiles'], peer_double_crop=best['consensus_double_crop'],
                            candidate_weight_sha256=candidate['weight_sha256'], peer_weight_sha256=peer['weight_sha256'],
                            automatic_fault_verdict=False, correlated_model_evidence=True)
            choices.append(addition)
        choices.sort(key=lambda r: (-min(r['confidence'], r['peer_confidence']), *r['box_xyxy'], r['class_id']))
        for row in choices:
            if len(output['strong_consensus_additions']) >= remaining:
                break
            if any(box_iou(row['box_xyxy'], p['box_xyxy']) >= .5 for p in output['all_predictions']):
                continue
            output['strong_consensus_additions'].append(row)
            output['all_predictions'].append(row)
        assert output['all_predictions'][:len(old)] == old
        assert len(output['all_predictions']) <= len(current['primary']) + 5
        return output
    except (KeyError, TypeError, ValueError, AssertionError) as error:
        output = copy.deepcopy(current)
        output.update(strong_consensus_additions=[], strong_consensus_fallback_reason=type(error).__name__ + ': ' + str(error))
        return output
