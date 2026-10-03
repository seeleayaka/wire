"""Workspace-only extra median geometry after actual accepted paired branch."""
import copy
import json
from pathlib import Path
from paired_median_proposals import proposals
from inspection_agent.paired_port_features import expected_in_source, valid_boxes, paired_features, embeddings, select
from inspection_agent.paired_port_geometry import paired_runtime_fingerprint, HEAD_SHA, HEAD_RELATIVE, ENCODER_SHA
from inspection_agent.optional_port_crop_review import sha, read_image, predict, aligned_predictions, REFERENCE_SHA
from inspection_agent.teacher_student_port_support import append_verified_student, TEACHER_SHA, TEACHER_RELATIVE, STUDENT_SHA, STUDENT_RELATIVE
from inspection_agent.resolution_loose_plug_support import resolution_candidates
from inspection_agent.feature_residual_port_support import residual_candidates
from inspection_agent.context_port_recheck import predict_seed_views, complete
from inspection_agent.port_tiling import box_iou
POLICY = 'accepted_paired_preserved_median_geometry_prototype_20261003'


def append_median_review(report, original, *, project):
    output = copy.deepcopy(original)
    output['median_geometry_policy'] = dict(policy_id=POLICY, enabled=True, added_hints=0,
        automatic_fault_verdict=False, existing_cues_preserved=True, probability_gate=.98,
        maximum_primary=5, maximum_extra=5, model_deployed=False)
    policy = output['median_geometry_policy']
    if original['status'] != 'applied' or len(original['supplementary_hints']) >= 5: return output
    try:
        import numpy as np
        import torch
        import dino_feature_diff as dino
        from ultralytics import YOLO
        root = Path(project); frozen = paired_runtime_fingerprint(root)
        if frozen['head'] != HEAD_SHA or frozen['encoder'] != ENCODER_SHA:
            raise ValueError('median_original_model_identity_mismatch')
        for name in ('teacher_student_policy', 'feature_residual_policy', 'resolution_policy', 'paired_geometry_policy'):
            if original.get(name, {}).get('fallback_reason'): raise ValueError('accepted_branch_not_available')
        if not original.get('paired_geometry_policy', {}).get('enabled'): raise ValueError('accepted_paired_branch_required')
        fingerprints = report['image_fingerprints']
        if fingerprints['stable_during_visual_analysis'] is not True: raise ValueError('unstable_visual_inputs')
        pins = {str(p): sha(p) for p in (Path(report['inspection']), Path(report['reference']), Path(__file__), Path(__file__).with_name('paired_median_proposals.py'))}
        if (pins[str(Path(report['inspection']))] != fingerprints['source_sha256'] or
                pins[str(Path(report['reference']))] != fingerprints['reference_sha256'] or fingerprints['reference_sha256'] != REFERENCE_SHA):
            raise ValueError('source_reference_identity_mismatch')
        protected_report = json.dumps(report, sort_keys=True); protected_original = json.dumps(original, sort_keys=True)
        base = original['teacher_student_evidence']; teacher, student = base['teacher_case'], base['student_case']
        feature = original['feature_residual_evidence']['feature_case']; alternative = original['resolution_evidence']['alternative']
        evidence = original.get('paired_geometry_evidence')
        current = copy.deepcopy(evidence['native']) if evidence else resolution_candidates(teacher, residual_candidates(teacher, student, feature), alternative)
        remaining = 5 - (len(current['all_predictions']) - len(current['primary']))
        if remaining < 0: raise ValueError('invalid_existing_budget')
        if not remaining: policy['native_candidates'] = 0; return output
        native = proposals(teacher, [teacher, student, feature, alternative])
        native = [p for p in native if not any(box_iou(p['box_xyxy'], q['box_xyxy']) >= .5 for q in current['all_predictions'])]
        image, reference = read_image(report['inspection']), read_image(report['reference'])
        matrix = np.asarray(report['alignment']['source_to_reference_homography'], dtype=np.float64)
        if not report['alignment']['alignment_quality']['reliable']: raise ValueError('unreliable_registration')
        expected, mask = expected_in_source(reference, matrix, image.shape[:2])
        native = [native[i] for i in valid_boxes([p['box_xyxy'] for p in native], mask)]
        torch.set_num_threads(2); encoder = dino._model(); encoder.eval().requires_grad_(False); torch.set_num_threads(2)
        checkpoint = torch.load(root / HEAD_RELATIVE, map_location='cpu', weights_only=True)
        if (checkpoint.get('input_dimensions') != 6144 or checkpoint.get('encoder_sha256') != ENCODER_SHA or
                checkpoint.get('classes') != ['other', 'unplugged_plug', 'unplugged_jack']): raise ValueError('median_checkpoint_contract_mismatch')
        head = torch.nn.Linear(6144, 3); head.load_state_dict(checkpoint['state_dict'], strict=True); head.eval().requires_grad_(False)
        with torch.inference_mode():
            vectors = paired_features(embeddings(encoder, image, [p['box_xyxy'] for p in native]), embeddings(encoder, expected, [p['box_xyxy'] for p in native]))
            scores = head(vectors).softmax(dim=1).tolist()
        fused = select(current, native, scores, HEAD_SHA); candidates = fused['paired_semantic_additions']
        refs = []; reference_evidence = []
        if candidates:
            mapped_native = copy.deepcopy(candidates)
            for row in mapped_native: row['support_tiles'] = []
            mapped = aligned_predictions(mapped_native, matrix, image.shape[:2], reference.shape[:2])
            seeds = [dict(class_id=p['class_id'], confidence=p['confidence'], box_xyxy=[p[k] for k in ('left', 'top', 'right', 'bottom')]) for p in mapped]
            for digest, relative in ((TEACHER_SHA, TEACHER_RELATIVE), (STUDENT_SHA, STUDENT_RELATIVE)):
                torch.set_num_threads(4); model = YOLO(str(root / relative))
                if sha(root / relative) != digest or model.task != 'segment' or dict(model.names) != {0: 'unplugged_plug', 1: 'unplugged_jack'}:
                    raise ValueError('median_reference_model_contract_mismatch')
                class Capped:
                    def predict(self, *args, **kw):
                        torch.set_num_threads(4); result = model.predict(*args, **kw); torch.set_num_threads(4); return result
                capped = Capped(); raw = predict(capped, reference); views = predict_seed_views(capped, reference, seeds)
                refs.extend(aligned_predictions(raw['merged_predictions'], np.eye(3), reference.shape[:2], reference.shape[:2]))
                reference_evidence.append(dict(weight_sha256=digest, predictions=raw, views=views))
                for record in views:
                    for view in record['views']:
                        for row in view:
                            if row['confidence'] > .25 and complete(row, reference.shape[:2]):
                                refs.append(dict(zip(('left', 'top', 'right', 'bottom'), row['box_xyxy']), class_id=row['class_id'],
                                    confidence=row['confidence'], valid_warp_fraction=1., support_tiles=[]))
        if ({p: sha(Path(p)) for p in pins} != pins or paired_runtime_fingerprint(root) != frozen or
                json.dumps(report, sort_keys=True) != protected_report or json.dumps(original, sort_keys=True) != protected_original):
            raise ValueError('inputs_changed_during_median_review')
        result, added = append_verified_student(output, candidates, refs, matrix)
        for hint in added:
            hint.update(evidence_tier='paired_median_geometry_manual_review', paired_geometry_head_sha256=HEAD_SHA,
                median_geometry_policy_id=POLICY, automatic_fault_verdict=False,
                warning='Median learned localization cue. Reference non-detection does not prove physical fault or continuity.')
        for name in ('parents', 'existing_hints', 'rescue_hints'): assert result[name] == original[name]
        assert result['supplementary_hints'][:len(original['supplementary_hints'])] == original['supplementary_hints']
        result['median_geometry_policy'].update(added_hints=len(added), native_candidates=len(candidates),
            valid_weak_proposals=len(native), fallback_reason=None, head_sha256=HEAD_SHA, source_and_reference_newly_inferred=True)
        result['median_geometry_evidence'] = dict(native_current=current, native=fused, proposals=native, probabilities=scores,
            reference_evidence=reference_evidence, pins=pins, runtime_fingerprint=frozen)
        return result
    except Exception as error:
        policy['fallback_reason'] = type(error).__name__ + ': ' + str(error)
        return output
