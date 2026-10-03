"""Default-off1280 loose-plug residual; preserve accepted source/reference cues."""
import copy,json
from pathlib import Path
from inspection_agent.feature_residual_port_support import (
    run_feature_residual_review,residual_candidates,residual_runtime_fingerprint,WEIGHT_SHA,WEIGHT_RELATIVE)
from inspection_agent.teacher_student_port_support import native_selection,complementary_candidates,append_verified_student
from inspection_agent.optional_port_crop_review import sha,read_image,predict,aligned_predictions
from inspection_agent.context_port_recheck import predict_seed_views,recheck_proposals,complete
from inspection_agent.port_tiling import box_iou
POLICY_ID='accepted_three_model_preserved_resolution_loose_plug_v3_20261003'
MANIFEST_NAME='resolution_loose_plug_support_20261003.json'
MANIFEST_SHA='478cda48b165b5eb4142bddf57d7399dd68fa3dde559950a5d32cbd6718f4866'

def resolution_runtime_fingerprint(project):
    root=Path(project)
    return dict(accepted_feature=residual_runtime_fingerprint(root),resolution_code=sha(Path(__file__)),
        resolution_manifest=sha(root/'config'/MANIFEST_NAME),predict_code=sha(root/'inspection_agent/optional_port_crop_review.py'),
        tiling_code=sha(root/'inspection_agent/port_tiling.py'))

def resolution_candidates(teacher,current,alternative):
    output=copy.deepcopy(current);output.update(resolution_additions=[],resolution_fallback_reason=None)
    candidates=complementary_candidates(teacher,alternative)
    if candidates['fallback_reason']:
        output['resolution_fallback_reason']=candidates['fallback_reason'];return output
    remaining=5-(len(current['all_predictions'])-len(current['primary']))
    for row in candidates['student_additions']:
        if row['class_id']!=0:continue
        if len(output['resolution_additions'])>=remaining:break
        if any(box_iou(row['box_xyxy'],old['box_xyxy'])>=.5 for old in output['all_predictions']):continue
        row=copy.deepcopy(row);row.update(evidence_tier='resolution_loose_plug_manual_review',inference_imgsz=1280,
            resolution_policy_id=POLICY_ID,loose_plug_only=True,automatic_fault_verdict=False,
            allowed_class_filtered_before_final_budget=True)
        output['resolution_additions'].append(row);output['all_predictions'].append(row)
    assert output['all_predictions'][:len(current['all_predictions'])]==current['all_predictions']
    assert len(output['primary'])<=5 and len(output['all_predictions'])<=len(output['primary'])+5
    return output

def append_plugs(original,candidates,refs,matrix):
    if any(row['class_id']!=0 for row in candidates):raise ValueError('empty_jack_in_loose_plug_branch')
    output,added=append_verified_student(original,candidates,refs,matrix)
    for hint in added:hint.update(evidence_tier='resolution_loose_plug_manual_review',inference_imgsz=1280,
        resolution_policy_id=POLICY_ID,student_weight_sha256=WEIGHT_SHA,loose_plug_only=True,automatic_fault_verdict=False)
    return output,added

def run_resolution_plug_review(report,*,project,resolution_enabled=False,**kwargs):
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
        root=Path(project);before=resolution_runtime_fingerprint(root);weight=root/WEIGHT_RELATIVE
        if before['resolution_manifest']!=MANIFEST_SHA:raise ValueError('resolution_provenance_manifest_mismatch')
        manifest=json.loads((root/'config'/MANIFEST_NAME).read_text(encoding='utf-8'))
        if (manifest.get('policy_id')!=POLICY_ID or manifest.get('feature_sha256')!=WEIGHT_SHA or
            manifest.get('allowed_new_classes')!=[0] or manifest.get('inference_imgsz')!=1280 or
            not manifest.get('source_gates_passed') or not manifest.get('live_diagnostic_passed')):
            raise ValueError('resolution_manifest_contract_mismatch')
        report_before=json.dumps(report,sort_keys=True,ensure_ascii=False)
        pins={str(p):sha(p) for p in (weight,Path(report['inspection']),Path(report['reference']))}
        if pins[str(weight)]!=WEIGHT_SHA:raise ValueError('plug_weight_identity_mismatch')
        base=original['teacher_student_evidence'];teacher=base['teacher_case']
        if teacher['source_sha256']!=pins[str(Path(report['inspection']))]:raise ValueError('source_changed')
        if original['source_evidence']['reference_sha256']!=pins[str(Path(report['reference']))]:raise ValueError('reference_changed')
        current=residual_candidates(teacher,base['student_case'],original['feature_residual_evidence']['feature_case'])
        if current['feature_fallback_reason']:raise ValueError(current['feature_fallback_reason'])
        torch.set_num_threads(4);model=YOLO(str(weight))
        if model.task!='segment' or dict(model.names)!={0:'unplugged_plug',1:'unplugged_jack'}:raise ValueError('plug_model_contract_mismatch')
        class HighResolution:
            def predict(self,*args,**kw):
                torch.set_num_threads(4);kw['imgsz']=1280
                result=model.predict(*args,**kw);torch.set_num_threads(4);return result
        capped=HighResolution();image,ref=read_image(report['inspection']),read_image(report['reference']);raw=predict(capped,image)
        alternative=dict(image=teacher['image'],source_sha256=teacher['source_sha256'],weight_sha256=WEIGHT_SHA,
            inference_imgsz=1280,predictions=raw,zoom_evidence=[])
        if len(native_selection(alternative)['supplementary'])<5:alternative['zoom_evidence']=predict_seed_views(capped,image,recheck_proposals(raw))
        fused=resolution_candidates(teacher,current,alternative)
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
        if ({p:sha(Path(p)) for p in pins}!=pins or resolution_runtime_fingerprint(root)!=before or
                json.dumps(report,sort_keys=True,ensure_ascii=False)!=report_before):raise ValueError('inputs_changed_during_plug_review')
        output,added=append_plugs(original,candidates,rows,matrix)
        output['resolution_policy'].update(added_hints=len(added),native_candidates=len(candidates),
            reference_inference_performed=bool(candidates),fallback_reason=None,weight_sha256=WEIGHT_SHA,
            provenance_manifest_sha256=MANIFEST_SHA)
        output['resolution_evidence']=dict(alternative=alternative,fusion=fused,reference_predictions=reference_raw,
            matched_reference_views=entries,inputs=pins,runtime_fingerprint=before)
        return output
    except Exception as error:policy['fallback_reason']=type(error).__name__+': '+str(error);return original
