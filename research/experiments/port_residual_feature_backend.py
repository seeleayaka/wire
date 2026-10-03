"""Isolated live residual experiment; the accepted formal branch is unchanged."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = Path('E:/PythonProject10')
sys.path.insert(0, str(REPO))
from inspection_agent.teacher_student_port_support import (
    run_teacher_student_review, append_verified_student, native_selection,
    support_runtime_fingerprint, TEACHER_SHA, STUDENT_SHA)
from inspection_agent.optional_port_crop_review import sha, read_image, predict, aligned_predictions
from inspection_agent.context_port_recheck import predict_seed_views, recheck_proposals, complete
from port_residual_feature_support import merge_residual, POLICY_ID

WEIGHT = ROOT / 'artifacts/port_feature_adaptation_20261003/full/runs/rectports/weights/last.pt'
WEIGHT_SHA = '18ab6c4c4327cc0b3d9870f5fe36efc6a93d78a6e341ecae05ae703deb4537dc'


def append_feature(original, candidates, refs, matrix):
    output, added = append_verified_student(original, candidates, refs, matrix)
    for hint in added:
        hint.update(evidence_tier='feature_residual_manual_review', student_weight_sha256=WEIGHT_SHA,
                    accepted_student_weight_sha256=STUDENT_SHA, residual_policy_id=POLICY_ID)
    return output, added


def run_residual_review(report, *, project=REPO, feature_enabled=False, **kwargs):
    original = run_teacher_student_review(report, project=project, **kwargs)
    original['feature_residual_policy'] = dict(policy_id=POLICY_ID, enabled=bool(feature_enabled),
        added_hints=0, automatic_fault_verdict=False, existing_cues_preserved=True,
        correlated_models_not_physical_evidence=True, maximum_primary=5, maximum_extra=5)
    if not feature_enabled or original['status'] != 'applied' or len(original['supplementary_hints']) >= 5:
        return original
    if (original.get('teacher_student_policy', {}).get('fallback_reason') or
            'teacher_student_evidence' not in original):
        original['feature_residual_policy']['fallback_reason'] = 'accepted_pair_not_available'
        return original
    try:
        import numpy as np
        import torch
        from ultralytics import YOLO
        before = support_runtime_fingerprint(project)
        paths = (WEIGHT, Path(report['inspection']), Path(report['reference']),
                 Path(__file__), Path(__file__).with_name('port_residual_feature_support.py'))
        pins = {str(path): sha(path) for path in paths}
        if pins[str(WEIGHT)] != WEIGHT_SHA:
            raise ValueError('feature_weight_identity_mismatch')
        base = original['teacher_student_evidence']
        teacher, accepted = base['teacher_case'], base['student_case']
        if teacher['source_sha256'] != pins[str(Path(report['inspection']))]:
            raise ValueError('source_changed_after_accepted_review')
        if original['source_evidence']['reference_sha256'] != pins[str(Path(report['reference']))]:
            raise ValueError('reference_changed_after_accepted_review')
        image, ref = read_image(report['inspection']), read_image(report['reference'])
        torch.set_num_threads(4); model = YOLO(str(WEIGHT))
        if model.task != 'segment' or dict(model.names) != {0:'unplugged_plug', 1:'unplugged_jack'}:
            raise ValueError('feature_model_contract_mismatch')
        class Capped:
            def predict(self, *args, **kw):
                result = model.predict(*args, **kw); torch.set_num_threads(4); return result
        capped = Capped(); raw = predict(capped, image)
        feature = dict(image=teacher['image'], source_sha256=teacher['source_sha256'], weight_sha256=WEIGHT_SHA,
                       predictions=raw, zoom_evidence=[])
        if len(native_selection(feature)['supplementary']) < 5:
            feature['zoom_evidence'] = predict_seed_views(capped, image, recheck_proposals(raw))
        fused = merge_residual(teacher, accepted, feature)
        if fused['feature_fallback_reason']:
            raise ValueError(fused['feature_fallback_reason'])
        candidates = fused['feature_additions']; rows = []; reference_raw = None; entries = []
        matrix = np.asarray(report['alignment']['source_to_reference_homography'], dtype=np.float64)
        if candidates:
            reference_raw = predict(capped, ref)
            rows = aligned_predictions(reference_raw['merged_predictions'], np.eye(3), ref.shape[:2], ref.shape[:2])
            native = copy.deepcopy(candidates)
            for row in native: row['support_tiles'] = []
            mapped = aligned_predictions(native, matrix, image.shape[:2], ref.shape[:2])
            seeds = [dict(box_xyxy=[row[k] for k in ('left','top','right','bottom')],
                          class_id=row['class_id'], confidence=row['confidence']) for row in mapped]
            entries = predict_seed_views(capped, ref, seeds)
            for entry in entries:
                for view in entry['views']:
                    for row in view:
                        if row['confidence'] > .25 and complete(row, ref.shape[:2]):
                            rows.append(dict(zip(('left','top','right','bottom'),row['box_xyxy']),
                                confidence=row['confidence'], class_id=row['class_id'], valid_warp_fraction=1., support_tiles=[]))
        if {p: sha(Path(p)) for p in pins} != pins or support_runtime_fingerprint(project) != before:
            raise ValueError('inputs_changed_during_feature_review')
        output, added = append_feature(original, candidates, rows, matrix)
        output['feature_residual_policy'].update(added_hints=len(added), native_candidates=len(candidates),
            reference_inference_performed=bool(candidates), fallback_reason=None, feature_weight_sha256=WEIGHT_SHA)
        output['feature_residual_evidence'] = dict(feature_case=feature, fusion=fused,
            feature_reference_predictions=reference_raw, matched_feature_reference_views=entries,
            inputs=pins, accepted_runtime_fingerprint=before)
        return output
    except Exception as error:
        original['feature_residual_policy']['fallback_reason'] = type(error).__name__ + ': ' + str(error)
        return original
