"""Default-off, bounded port model cues. Never an electrical fault verdict."""
import copy
import math

from inspection_agent.port_crop_gui_bridge import run_gui_port_review
from inspection_agent.port_tiling import box_iou, select_additional_tile_hint


def review_rois(report):
    """Use recorded analysis ROIs, not an editable or hard-coded scene recipe."""
    raw = report.get('analysis_check_rois')
    if not isinstance(raw, list) or not raw:
        raise ValueError('analysis_roi_provenance_missing')
    result = []
    for roi in raw:
        if (not isinstance(roi, list) or len(roi) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in roi)
                or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1)):
            raise ValueError('invalid_analysis_roi')
        result.append(dict(zip(('left','top','right','bottom'),
                              (round(roi[0]*3648), round(roi[1]*2736), round(roi[2]*3648), round(roi[3]*2736)))))
    return result


def select_rescue(rois, existing, predictions, reference_predictions):
    coordinates = lambda p: [p[k] for k in ('left','top','right','bottom')]
    strong = [p for p in reference_predictions if p['confidence'] > .25]
    novel = [p for p in predictions if not any(
        p['class_id'] == q['class_id'] and box_iou(coordinates(p), coordinates(q)) >= .5
        for q in strong)]
    selected = select_additional_tile_hint(rois, existing, novel, .25)
    for hint in selected['tile_hints']:
        hint.update(parent_index=None, role='independent_port_model_cue_manual_review',
                    reference_evidence='no_matching_strong_reference_model_cue',
                    warning='Reference non-detection is not proof of change, seating or continuity.')
    return selected['tile_hints'], dict(static_reference_suppressed=len(predictions)-len(novel),
                                        selection=selected['selection_audit'])


def run_independent_port_rescue(report, *, project, enabled=False, scene='unknown'):
    """Fresh source AND reference inference through the same frozen gates.

    No prediction-file/provider input is exposed. Original review regions survive.
    """
    original = copy.deepcopy(report.get('review_regions', []))
    existing = copy.deepcopy(report.get('existing_port_hints', []))
    existing += copy.deepcopy(report.get('optional_port_crop', {}).get('tile_hints', []))
    result = dict(status='disabled', fallback_reason=None, parents=original,
                  existing_hints=existing, rescue_hints=[], automatic_fault_verdict=False,
                  decision='possible_difference_manual_review')
    if not enabled:
        return result
    result['status'] = 'fallback'
    try:
        rois = review_rois(report)
        proxy = copy.deepcopy(report)
        proxy['review_regions'] = rois
        proxy['existing_port_hints'] = existing
        source = run_gui_port_review(proxy, project=project, enabled=True, scene=scene)
        if source['status'] != 'applied':
            result['fallback_reason'] = source['fallback_reason']
            return result
        reference = copy.deepcopy(proxy)
        reference['inspection'] = reference['reference']
        reference['image_fingerprints']['source_sha256'] = source['reference_sha256']
        reference['alignment']['source_to_reference_homography'] = [[1,0,0],[0,1,0],[0,0,1]]
        ref = run_gui_port_review(reference, project=project, enabled=True, scene=scene)
        if ref['status'] != 'applied':
            result['fallback_reason'] = 'reference_inference: ' + str(ref['fallback_reason'])
            return result
        from inspection_agent.optional_port_crop_review import sha
        if sha(report['inspection']) != source['source_sha256']:
            raise ValueError('input_changed_during_reference_prediction')
        hints, audit = select_rescue(rois, existing, source['aligned_predictions'], ref['aligned_predictions'])
        result.update(status='applied', rescue_hints=hints, audit=audit, analysis_rois=rois,
                      source_evidence=source, reference_evidence=ref,
                      warning='Model cues only. Operator-declared existing PC scene; not a fault verdict.')
    except Exception as error:
        result['fallback_reason'] = str(error)
    return result
