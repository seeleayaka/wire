"""Default-off native pose classifier; preserve accepted median and safety."""
import copy
import json
from pathlib import Path
from inspection_agent.paired_native_pose_features import proposals,select
from inspection_agent.paired_port_features import expected_in_source,valid_boxes,paired_features,embeddings
from inspection_agent.paired_median_geometry import run_paired_median_review,median_runtime_fingerprint
from inspection_agent.paired_port_geometry import ENCODER_SHA
from inspection_agent.optional_port_crop_review import sha,read_image,predict,aligned_predictions,REFERENCE_SHA
from inspection_agent.teacher_student_port_support import append_verified_student,TEACHER_SHA,TEACHER_RELATIVE,STUDENT_SHA,STUDENT_RELATIVE
from inspection_agent.context_port_recheck import predict_seed_views,complete
POLICY_ID='accepted_median_preserved_native_pose_three_checkpoint_v1_20261004'
HEAD_RELATIVE='output/paired_pose_native_20261004/last_head.pt'
HEAD_SHA='9459ba29715a9f511ddc48a9a99014d2a239cfe65ccc32fe5e2de9122ab72467'
MANIFEST_NAME='paired_pose_native_20261004.json'
# Bound after source, ALL20 actual/portable, and fresh SAM/Qt candidate gates.
MANIFEST_SHA='ad8daa84d3c7d8cd499635226f1c4bc716369d2366c1f1efbeef5bb0038db9e3'


def native_pose_runtime_fingerprint(project):
    root=Path(project)
    return dict(accepted=median_runtime_fingerprint(root),head=sha(root/HEAD_RELATIVE),
        encoder=sha(root/'models/dinov2/weights/dinov2_vits14_pretrain.pth'),
        manifest=sha(root/'config'/MANIFEST_NAME),backend=sha(Path(__file__)),
        geometry=sha(root/'inspection_agent/paired_native_pose_features.py'))


def run_native_pose_review(report,*,project,native_pose_enabled=False,median_enabled=False,paired_enabled=False,**kwargs):
    original=run_paired_median_review(report,project=project,median_enabled=median_enabled,paired_enabled=paired_enabled,**kwargs)
    if native_pose_enabled:return append_native_pose_review(report,original,project=project)
    original['native_pose_policy']=dict(policy_id=POLICY_ID,enabled=False,added_hints=0,
        automatic_fault_verdict=False,existing_cues_preserved=True,maximum_primary=5,maximum_extra=5)
    return original


def validate_release(manifest,frozen):
    if not MANIFEST_SHA or frozen['manifest']!=MANIFEST_SHA:raise ValueError('native_manifest_not_accepted_or_changed')
    if frozen['head']!=HEAD_SHA or frozen['encoder']!=ENCODER_SHA:raise ValueError('native_model_identity_mismatch')
    if manifest.get('accepted_runtime')!=frozen['accepted']:raise ValueError('accepted_median_runtime_changed')
    if not manifest.get('source_gates_passed') or not manifest.get('live_diagnostic_passed'):
        raise ValueError('native_source_and_live_acceptance_required')
    expected=dict(policy_id=POLICY_ID,head_sha256=HEAD_SHA,encoder_sha256=ENCODER_SHA,
        geometry_sha256=frozen['geometry'],feature_dimensions=6144,classes=['other','unplugged_plug','unplugged_jack'],
        contexts=[1.5,3.0],probability_gate=.98,minimum_unique_checkpoint_votes=3,
        maximum_primary=5,maximum_extra=5,default_off=True,manual_review_only=True,automatic_fault_verdict=False)
    if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError('native_manifest_contract_mismatch')


def failed_review(original,policy,error):
    result=copy.deepcopy(original)
    result['native_pose_policy']=dict(policy,added_hints=0,
        fallback_reason=type(error).__name__+': '+str(error),discarded_new_hints_on_failure=True)
    return result


def append_native_pose_review(report,original,*,project):
    output=copy.deepcopy(original)
    output['native_pose_policy']=dict(policy_id=POLICY_ID,enabled=True,added_hints=0,automatic_fault_verdict=False,
        existing_cues_preserved=True,probability_gate=.98,maximum_primary=5,maximum_extra=5,min_distinct_checkpoint_votes=3)
    policy=output['native_pose_policy']
    if original['status']!='applied':return output
    if len(original.get('supplementary_hints',[]))>=5:policy['shared_budget_full']=True;return output
    try:
        import numpy as np
        import torch
        import dino_feature_diff as dino
        from ultralytics import YOLO
        root=Path(project);frozen=native_pose_runtime_fingerprint(root)
        manifest=json.loads((root/'config'/MANIFEST_NAME).read_text(encoding='utf-8'));validate_release(manifest,frozen)
        for name in ('teacher_student_policy','feature_residual_policy','resolution_policy','paired_geometry_policy','median_geometry_policy'):
            if original.get(name,{}).get('fallback_reason'):raise ValueError('accepted_prefix_not_available')
        if not original.get('median_geometry_policy',{}).get('enabled'):raise ValueError('accepted_median_branch_required')
        evidence=original.get('median_geometry_evidence') or original.get('paired_geometry_evidence')
        if not evidence:policy['upstream_without_native_evidence']=True;return output
        current=copy.deepcopy(evidence['native']);remaining=5-(len(current['all_predictions'])-len(current['primary']))
        if remaining<0:raise ValueError('invalid_existing_shared_budget')
        if not remaining:policy['shared_budget_full']=True;return output
        fp=report['image_fingerprints']
        if fp['stable_during_visual_analysis'] is not True:raise ValueError('unstable_visual_inputs')
        pins={str(p):sha(p) for p in (Path(report['inspection']),Path(report['reference']),Path(__file__),root/'inspection_agent/paired_native_pose_features.py')}
        if pins[str(Path(report['inspection']))]!=fp['source_sha256'] or pins[str(Path(report['reference']))]!=fp['reference_sha256'] or fp['reference_sha256']!=REFERENCE_SHA:
            raise ValueError('native_source_reference_identity_mismatch')
        protected_report=json.dumps(report,sort_keys=True);protected_original=json.dumps(original,sort_keys=True)
        base=original['teacher_student_evidence'];teacher,student=base['teacher_case'],base['student_case']
        feature=original['feature_residual_evidence']['feature_case'];alternative=original['resolution_evidence']['alternative']
        native=proposals(teacher,[teacher,student,feature,alternative],current)
        image,reference=read_image(report['inspection']),read_image(report['reference'])
        matrix=np.asarray(report['alignment']['source_to_reference_homography'],dtype=np.float64)
        if not report['alignment']['alignment_quality']['reliable']:raise ValueError('unreliable_native_registration')
        expected,mask=expected_in_source(reference,matrix,image.shape[:2]);native=[native[i] for i in valid_boxes([p['box_xyxy'] for p in native],mask)]
        torch.set_num_threads(2);encoder=dino._model();encoder.eval().requires_grad_(False);torch.set_num_threads(2)
        checkpoint=torch.load(root/HEAD_RELATIVE,map_location='cpu',weights_only=True)
        if checkpoint.get('input_dimensions')!=6144 or checkpoint.get('encoder_sha256')!=ENCODER_SHA or checkpoint.get('classes')!=['other','unplugged_plug','unplugged_jack']:
            raise ValueError('native_checkpoint_contract_mismatch')
        head=torch.nn.Linear(6144,3);head.load_state_dict(checkpoint['state_dict'],strict=True);head.eval().requires_grad_(False)
        with torch.inference_mode():
            vectors=paired_features(embeddings(encoder,image,[p['box_xyxy'] for p in native]),embeddings(encoder,expected,[p['box_xyxy'] for p in native]))
            scores=head(vectors).softmax(dim=1).tolist()
        fused=select(current,native,scores,HEAD_SHA);candidates=fused['paired_semantic_additions'];refs=[];reference_evidence=[]
        if candidates:
            mapped_native=copy.deepcopy(candidates)
            for row in mapped_native:row['support_tiles']=[]
            mapped=aligned_predictions(mapped_native,matrix,image.shape[:2],reference.shape[:2])
            seeds=[dict(class_id=p['class_id'],confidence=p['confidence'],box_xyxy=[p[k] for k in ('left','top','right','bottom')]) for p in mapped]
            for digest,relative in ((TEACHER_SHA,TEACHER_RELATIVE),(STUDENT_SHA,STUDENT_RELATIVE)):
                torch.set_num_threads(4);model=YOLO(str(root/relative))
                if sha(root/relative)!=digest or model.task!='segment' or dict(model.names)!={0:'unplugged_plug',1:'unplugged_jack'}:
                    raise ValueError('native_reference_model_contract_mismatch')
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
        if {p:sha(Path(p)) for p in pins}!=pins or native_pose_runtime_fingerprint(root)!=frozen or json.dumps(report,sort_keys=True)!=protected_report or json.dumps(original,sort_keys=True)!=protected_original:
            raise ValueError('native_inputs_changed_during_review')
        result,added=append_verified_student(output,candidates,refs,matrix)
        for hint in added:hint.update(evidence_tier='native_learned_pose_three_checkpoint_manual_review',paired_geometry_head_sha256=HEAD_SHA,
            native_pose_policy_id=POLICY_ID,automatic_fault_verdict=False,
            warning='Native learned localization cue. Reference non-detection does not prove physical fault or continuity.')
        for key,value in original.items():
            if key=='supplementary_hints':assert result[key][:len(value)]==value
            else:assert result[key]==value
        result['native_pose_policy'].update(added_hints=len(added),native_candidates=len(candidates),valid_weak_proposals=len(native),
            fallback_reason=None,head_sha256=HEAD_SHA,source_and_reference_newly_inferred=True)
        result['native_pose_evidence']=dict(native_current=current,native=fused,proposals=native,probabilities=scores,
            reference_evidence=reference_evidence,pins=pins,runtime_fingerprint=frozen)
        return result
    except Exception as error:
        return failed_review(original,policy,error)
