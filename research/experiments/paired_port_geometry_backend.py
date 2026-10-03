"""Workspace-only current-V3 additive paired localization prototype, GT-free."""
import copy
import json
from pathlib import Path
from prepare_paired_semantic_geometry import NEW
from prepare_paired_port_semantics import REPO,load
from paired_port_semantics import expected_in_source,valid_boxes,paired_features
from port_semantic_verifier import embeddings
from port_semantic_model_vote import proposals
from paired_port_semantic_selection import select
from inspection_agent.optional_port_crop_review import sha,read_image,predict,aligned_predictions,REFERENCE_SHA
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint,resolution_candidates
from inspection_agent.feature_residual_port_support import residual_candidates
from inspection_agent.teacher_student_port_support import append_verified_student,TEACHER_SHA,TEACHER_RELATIVE,STUDENT_SHA,STUDENT_RELATIVE
from inspection_agent.context_port_recheck import predict_seed_views,complete
from inspection_agent.port_tiling import box_iou
HEAD_SHA='238f517bfdbe5e81f6c98e28c33750fa02de75cf25ec0c93b8876fe72657004a'
POLICY='current_V3_preserved_paired_geometry_prototype_20261003'


def append_geometry_review(report,original,*,project):
    output=copy.deepcopy(original);output['paired_geometry_policy']=dict(policy_id=POLICY,enabled=True,
        automatic_fault_verdict=False,added_hints=0,existing_cues_preserved=True,maximum_primary=5,maximum_extra=5,
        probability_gate=.98,model_deployed=False)
    policy=output['paired_geometry_policy']
    if original['status']!='applied' or len(original['supplementary_hints'])>=5:return output
    try:
        import numpy as np
        import torch
        import dino_feature_diff as dino
        from ultralytics import YOLO
        root=Path(project);frozen=resolution_runtime_fingerprint(root);trained=load(NEW/'head_full/report.json')
        if frozen!=trained['runtime_fingerprint']:raise ValueError('accepted_runtime_changed')
        weight=Path(trained['checkpoint']['path'])
        if sha(weight)!=HEAD_SHA or trained['checkpoint']['sha256']!=HEAD_SHA or not trained['no_validation_training']:
            raise ValueError('geometry_head_identity_mismatch')
        fingerprints=report['image_fingerprints']
        if fingerprints['stable_during_visual_analysis'] is not True or sha(report['reference'])!=REFERENCE_SHA:
            raise ValueError('unstable_or_unknown_reference')
        pins={str(p):sha(p) for p in (Path(report['inspection']),Path(report['reference']),weight,Path(__file__),root/'models/dinov2/weights/dinov2_vits14_pretrain.pth')}
        if pins[str(Path(report['inspection']))]!=fingerprints['source_sha256'] or pins[str(Path(report['reference']))]!=fingerprints['reference_sha256']:
            raise ValueError('source_reference_fingerprint_mismatch')
        before_report=json.dumps(report,sort_keys=True);before_original=json.dumps(original,sort_keys=True)
        for key in ('teacher_student_policy','feature_residual_policy','resolution_policy'):
            if original.get(key,{}).get('fallback_reason'):raise ValueError('accepted_branch_not_available')
        base=original['teacher_student_evidence'];teacher,student=base['teacher_case'],base['student_case']
        feature=original['feature_residual_evidence']['feature_case'];alternative=original['resolution_evidence']['alternative']
        if teacher['source_sha256']!=pins[str(Path(report['inspection']))] or student['source_sha256']!=teacher['source_sha256']:
            raise ValueError('stale_source_model_evidence')
        current=resolution_candidates(teacher,residual_candidates(teacher,student,feature),alternative)
        if current.get('resolution_fallback_reason') or current.get('feature_fallback_reason'):raise ValueError('native_current_unavailable')
        remaining=5-(len(current['all_predictions'])-len(current['primary']))
        if not remaining:policy['native_candidates']=0;return output
        native=proposals(teacher,[teacher,student]);native=[row for row in native if not any(box_iou(row['box_xyxy'],old['box_xyxy'])>=.5 for old in current['all_predictions'])]
        image,reference=read_image(report['inspection']),read_image(report['reference'])
        matrix=np.asarray(report['alignment']['source_to_reference_homography'],dtype=np.float64)
        if not report['alignment']['alignment_quality']['reliable']:raise ValueError('registration_unreliable')
        expected,valid=expected_in_source(reference,matrix,image.shape[:2]);native=[native[i] for i in valid_boxes([row['box_xyxy'] for row in native],valid)]
        torch.set_num_threads(2);encoder=dino._model();encoder.requires_grad_(False).eval();torch.set_num_threads(2)
        checkpoint=torch.load(weight,map_location='cpu',weights_only=True)
        if checkpoint['input_dimensions']!=6144 or checkpoint['encoder_sha256']!=pins[str(root/'models/dinov2/weights/dinov2_vits14_pretrain.pth')]:raise ValueError('encoder_contract_mismatch')
        head=torch.nn.Linear(6144,3);head.load_state_dict(checkpoint['state_dict'],strict=True);head.requires_grad_(False).eval()
        with torch.inference_mode():
            vectors=paired_features(embeddings(encoder,image,[row['box_xyxy'] for row in native]),embeddings(encoder,expected,[row['box_xyxy'] for row in native]))
            scores=head(vectors).softmax(dim=1).tolist()
        fused=select(current,native,scores,HEAD_SHA);candidates=fused['paired_semantic_additions'];refs=[];reference_evidence=[]
        if candidates:
            mapped_native=copy.deepcopy(candidates)
            for row in mapped_native:row['support_tiles']=[]
            mapped=aligned_predictions(mapped_native,matrix,image.shape[:2],reference.shape[:2])
            seeds=[dict(class_id=row['class_id'],confidence=row['confidence'],box_xyxy=[row[k] for k in ('left','top','right','bottom')]) for row in mapped]
            for digest,relative in ((TEACHER_SHA,TEACHER_RELATIVE),(STUDENT_SHA,STUDENT_RELATIVE)):
                torch.set_num_threads(4);model=YOLO(str(root/relative))
                if sha(root/relative)!=digest or model.task!='segment' or dict(model.names)!={0:'unplugged_plug',1:'unplugged_jack'}:raise ValueError('reference_model_contract_mismatch')
                class Capped:
                    def predict(self,*args,**kw):
                        torch.set_num_threads(4);result=model.predict(*args,**kw);torch.set_num_threads(4);return result
                capped=Capped();raw=predict(capped,reference);views=predict_seed_views(capped,reference,seeds)
                refs.extend(aligned_predictions(raw['merged_predictions'],np.eye(3),reference.shape[:2],reference.shape[:2]))
                reference_evidence.append(dict(weight_sha256=digest,predictions=raw,views=views))
                for record in views:
                    for view in record['views']:
                        for row in view:
                            if row['confidence']>.25 and complete(row,reference.shape[:2]):
                                refs.append(dict(zip(('left','top','right','bottom'),row['box_xyxy']),class_id=row['class_id'],confidence=row['confidence'],valid_warp_fraction=1.,support_tiles=[]))
        if ({p:sha(Path(p)) for p in pins}!=pins or resolution_runtime_fingerprint(root)!=frozen or
            json.dumps(report,sort_keys=True)!=before_report or json.dumps(original,sort_keys=True)!=before_original):raise ValueError('inputs_changed_during_paired_geometry_review')
        result,added=append_verified_student(output,candidates,refs,matrix)
        for hint in added:hint.update(evidence_tier='paired_geometry_manual_review',paired_geometry_head_sha256=HEAD_SHA,
            paired_geometry_policy_id=POLICY,automatic_fault_verdict=False,
            warning='Paired learned localization cue. Reference non-detection does not prove physical fault or continuity.')
        assert result['parents']==original['parents'] and result['existing_hints']==original['existing_hints'] and result['rescue_hints']==original['rescue_hints']
        assert result['supplementary_hints'][:len(original['supplementary_hints'])]==original['supplementary_hints']
        result['paired_geometry_policy'].update(added_hints=len(added),native_candidates=len(candidates),valid_weak_proposals=len(native),
            fallback_reason=None,head_sha256=HEAD_SHA,source_and_reference_newly_inferred=True)
        result['paired_geometry_evidence']=dict(native_current=current,native=fused,proposals=native,probabilities=scores,
            reference_evidence=reference_evidence,pins=pins,runtime_fingerprint=frozen)
        return result
    except Exception as error:
        policy['fallback_reason']=type(error).__name__+': '+str(error);return output
