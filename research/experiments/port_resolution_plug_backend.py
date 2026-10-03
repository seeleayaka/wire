"""Isolated live strict1280 loose-plug residual, no formal deployment."""
import copy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');sys.path.insert(0,str(REPO))
from inspection_agent.feature_residual_port_support import (
    run_feature_residual_review,residual_candidates,residual_runtime_fingerprint,WEIGHT_SHA,WEIGHT_RELATIVE)
from inspection_agent.teacher_student_port_support import native_selection,append_verified_student
from inspection_agent.optional_port_crop_review import sha,read_image,predict,aligned_predictions
from inspection_agent.context_port_recheck import predict_seed_views,recheck_proposals,complete
from port_resolution_support import ResolutionModel
from port_resolution_plug_policy import append_resolution_plugs,POLICY_ID
def append_plugs(original,candidates,refs,matrix):
    if any(row['class_id']!=0 for row in candidates):raise ValueError('empty_jack_in_loose_plug_branch')
    output,added=append_verified_student(original,candidates,refs,matrix)
    for hint in added:hint.update(evidence_tier='resolution_loose_plug_manual_review',inference_imgsz=1280,
        resolution_policy_id=POLICY_ID,student_weight_sha256=WEIGHT_SHA,loose_plug_only=True,automatic_fault_verdict=False)
    return output,added
def run_plug_review(report,*,project=REPO,resolution_enabled=False,**kwargs):
    original=run_feature_residual_review(report,project=project,**kwargs)
    original['resolution_policy']=dict(policy_id=POLICY_ID,enabled=bool(resolution_enabled),added_hints=0,
        automatic_fault_verdict=False,existing_cues_preserved=True,maximum_primary=5,maximum_extra=5,
        input_resolution=1280,loose_plug_only=True)
    policy=original['resolution_policy']
    if not resolution_enabled or original['status']!='applied' or len(original['supplementary_hints'])>=5:return original
    if original.get('feature_residual_policy',{}).get('fallback_reason') or 'feature_residual_evidence' not in original:
        policy['fallback_reason']='accepted_feature_branch_not_available';return original
    try:
        import numpy as np
        import torch
        from ultralytics import YOLO
        root=Path(project);before=residual_runtime_fingerprint(root);weight=root/WEIGHT_RELATIVE
        pins={str(p):sha(p) for p in (weight,Path(report['inspection']),Path(report['reference']),Path(__file__),
            Path(__file__).with_name('port_resolution_support.py'),Path(__file__).with_name('port_resolution_plug_policy.py'))}
        if pins[str(weight)]!=WEIGHT_SHA:raise ValueError('plug_weight_identity_mismatch')
        base=original['teacher_student_evidence'];teacher=base['teacher_case']
        if teacher['source_sha256']!=pins[str(Path(report['inspection']))]:raise ValueError('source_changed')
        if original['source_evidence']['reference_sha256']!=pins[str(Path(report['reference']))]:raise ValueError('reference_changed')
        current=residual_candidates(teacher,base['student_case'],original['feature_residual_evidence']['feature_case'])
        if current['feature_fallback_reason']:raise ValueError(current['feature_fallback_reason'])
        torch.set_num_threads(4);model=YOLO(str(weight))
        if model.task!='segment' or dict(model.names)!={0:'unplugged_plug',1:'unplugged_jack'}:raise ValueError('plug_model_contract_mismatch')
        capped=ResolutionModel(model,1280,lambda:torch.set_num_threads(4))
        image,ref=read_image(report['inspection']),read_image(report['reference']);raw=predict(capped,image)
        alternative=dict(image=teacher['image'],source_sha256=teacher['source_sha256'],weight_sha256=WEIGHT_SHA,
            inference_imgsz=1280,predictions=raw,zoom_evidence=[])
        if len(native_selection(alternative)['supplementary'])<5:alternative['zoom_evidence']=predict_seed_views(capped,image,recheck_proposals(raw))
        fused=append_resolution_plugs(teacher,current,alternative)
        if fused['resolution_fallback_reason']:raise ValueError(fused['resolution_fallback_reason'])
        candidates=fused['resolution_additions'];rows=[];reference_raw=None;entries=[]
        matrix=np.asarray(report['alignment']['source_to_reference_homography'],dtype=np.float64)
        if candidates:
            reference_raw=predict(capped,ref)
            rows=aligned_predictions(reference_raw['merged_predictions'],np.eye(3),ref.shape[:2],ref.shape[:2])
            native=copy.deepcopy(candidates)
            for row in native:row['support_tiles']=[]
            mapped=aligned_predictions(native,matrix,image.shape[:2],ref.shape[:2])
            seeds=[dict(box_xyxy=[row[k] for k in ('left','top','right','bottom')],class_id=row['class_id'],confidence=row['confidence']) for row in mapped]
            entries=predict_seed_views(capped,ref,seeds)
            for entry in entries:
                for view in entry['views']:
                    for row in view:
                        if row['confidence']>.25 and complete(row,ref.shape[:2]):
                            rows.append(dict(zip(('left','top','right','bottom'),row['box_xyxy']),confidence=row['confidence'],
                                class_id=row['class_id'],valid_warp_fraction=1.,support_tiles=[]))
        if {p:sha(Path(p)) for p in pins}!=pins or residual_runtime_fingerprint(root)!=before:raise ValueError('inputs_changed_during_plug_review')
        output,added=append_plugs(original,candidates,rows,matrix)
        output['resolution_policy'].update(added_hints=len(added),native_candidates=len(candidates),
            reference_inference_performed=bool(candidates),fallback_reason=None,weight_sha256=WEIGHT_SHA)
        output['resolution_evidence']=dict(alternative=alternative,fusion=fused,reference_predictions=reference_raw,
            matched_reference_views=entries,inputs=pins,runtime_fingerprint=before)
        return output
    except Exception as error:policy['fallback_reason']=type(error).__name__+': '+str(error);return original
