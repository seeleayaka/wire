"""Experimental append-only residual support, never a replacement detector."""
import copy

from core_port_precision_policy import iou
from teacher_student_port_policy import merge

POLICY_ID = 'accepted_pair_preserved_feature_residual_v1_20261003'


def merge_residual(teacher, accepted_student, feature_student):
    accepted = merge(teacher, accepted_student)
    output = copy.deepcopy(accepted)
    output.update(feature_additions=[], feature_fallback_reason=None)
    if accepted['fallback_reason']:
        output['feature_fallback_reason'] = 'accepted_pair_invalid'
        return output
    feature = merge(teacher, feature_student)
    if feature['fallback_reason']:
        output['feature_fallback_reason'] = feature['fallback_reason']
        return output
    # Do not replace, average, reorder or rescore any accepted cue. All models
    # share the original five supplementary slots (including old rechecks).
    remaining = 5 - (len(accepted['all_predictions']) - len(accepted['primary']))
    for cue in feature['student_additions']:
        if len(output['feature_additions']) >= remaining:
            break
        if any(iou(cue['box_xyxy'], old['box_xyxy']) >= .5
               for old in output['all_predictions']):
            continue
        cue = copy.deepcopy(cue)
        cue['evidence_tier'] = 'feature_residual_manual_review'
        output['feature_additions'].append(cue)
        output['all_predictions'].append(cue)
    assert output['all_predictions'][:len(accepted['all_predictions'])] == accepted['all_predictions']
    assert output['primary'] == accepted['primary'] and len(output['primary']) <= 5
    assert len(output['all_predictions']) <= len(output['primary']) + 5
    return output
