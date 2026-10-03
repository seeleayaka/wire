"""Experimental, default-off complementary model within old source/reference gates."""
import copy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
sys.path.insert(0,str(REPO))
from inspection_agent.context_port_recheck import run_context_port_recheck,predict_seed_views,complete
from inspection_agent.independent_port_rescue import select_rescue
from inspection_agent.optional_port_crop_review import sha,read_image,predict,aligned_predictions
from inspection_agent.port_tiling import box_iou
from core_port_recheck_policy import proposals,select_zoom
from core_port_supplement_policy import supplement
from teacher_student_port_policy import merge
STUDENT_WEIGHT=ROOT/'artifacts/port_training_multiscale_20261002/full/runs/rectports/weights/last.pt'
STUDENT_SHA='f519a566d98756d988372c3f24dd827480ea41ac7cdb9fd12a97a426756d55b4'
TEACHER_SHA='9ed5ae77c940a78869e084639c55294b05fc65a78e3723f84af2fdc8189ed824'
POLICY_ID='teacher_preserved_student_cross_support_v1_20261003'

def append_verified_student(original,native_candidates,reference_rows,matrix):
    output=copy.deepcopy(original);source=output['source_evidence'];reference=output['reference_evidence']
    candidates=copy.deepcopy(native_candidates)
    for row in candidates:row['support_tiles']=[]
    mapped=aligned_predictions(candidates,matrix,source['predictions']['source_shape'],reference['predictions']['source_shape'])
    refs=copy.deepcopy(reference['aligned_predictions'])+copy.deepcopy(reference_rows)
    old=output['existing_hints']+output['rescue_hints']+output['supplementary_hints'];added=[]
    coords=lambda p:[p[k] for k in ('left','top','right','bottom')]
    for _ in range(max(0,5-len(output['supplementary_hints']))):
        remaining=[p for p in mapped if not any(box_iou(coords(p),coords(h['box']))>=.5 for h in old+added)]
        hints,_=select_rescue(output['analysis_rois'],old+added,remaining,refs)
        if not hints:break
        for hint in hints:
            hint.update(evidence_tier='teacher_student_complement_manual_review',
                teacher_weight_sha256=TEACHER_SHA,student_weight_sha256=STUDENT_SHA,
                warning='Correlated model evidence only; reference non-detection does not prove a physical fault.')
        added.extend(hints)
    output['supplementary_hints'].extend(added)
    assert output['rescue_hints']==original['rescue_hints']
    assert output['supplementary_hints'][:len(original['supplementary_hints'])]==original['supplementary_hints']
    assert len(output['rescue_hints'])<=5 and len(output['supplementary_hints'])<=5
    return output,added

def run_teacher_student_review(report,*,project=REPO,enabled=False,scene='unknown',supplementary_enabled=False,student_enabled=False):
    original=run_context_port_recheck(report,project=project,enabled=enabled,scene=scene,supplementary_enabled=supplementary_enabled)
    original['teacher_student_policy']=dict(policy_id=POLICY_ID,enabled=bool(student_enabled),
        experimental=True,maximum_total_primary=5,maximum_total_supplementary=5,
        teacher_preserved=True,automatic_fault_verdict=False,added_hints=0,
        correlated_models_not_physical_evidence=True)
    if not student_enabled or not supplementary_enabled or original['status']!='applied' or len(original['supplementary_hints'])>=5:return original
    try:
        import torch,numpy as np
        from ultralytics import YOLO
        teacher=Path(project)/'output/port_crop_training_fixed_20260929/full/runs/rectports/weights/best.pt'
        inputs={str(p):sha(p) for p in (teacher,STUDENT_WEIGHT,Path(report['inspection']),Path(report['reference']))}
        if inputs[str(teacher)]!=TEACHER_SHA or inputs[str(STUDENT_WEIGHT)]!=STUDENT_SHA:raise ValueError('model_identity_mismatch')
        source=original['source_evidence'];reference=original['reference_evidence']
        if source['weight_sha256']!=TEACHER_SHA or source['source_sha256']!=inputs[str(Path(report['inspection']))] or source['reference_sha256']!=inputs[str(Path(report['reference']))]:raise ValueError('source_reference_identity_mismatch')
        old_case=dict(image=Path(report['inspection']).name,source_sha256=source['source_sha256'],weight_sha256=TEACHER_SHA,
            predictions=source['predictions'],zoom_evidence=original.get('recheck_evidence',{}).get('source_views',[]))
        image=read_image(report['inspection']);ref=read_image(report['reference']);torch.set_num_threads(4)
        model=YOLO(str(STUDENT_WEIGHT))
        if model.task!='segment' or dict(model.names)!={0:'unplugged_plug',1:'unplugged_jack'}:raise ValueError('student_contract_mismatch')
        class Capped:
            def predict(self,*args,**kw):
                result=model.predict(*args,**kw);torch.set_num_threads(4);return result
        capped=Capped();raw=predict(capped,image)
        new_case=dict(image=old_case['image'],source_sha256=old_case['source_sha256'],weight_sha256=STUDENT_SHA,predictions=raw,zoom_evidence=[])
        if len(supplement(raw)['supplementary'])<5:new_case['zoom_evidence']=predict_seed_views(capped,image,proposals(raw))
        fusion=merge(old_case,new_case,'cross_model_supported');candidates=fusion['student_additions']
        if fusion['fallback_reason']:raise ValueError('native_fusion: '+fusion['fallback_reason'])
        matrix=np.asarray(report['alignment']['source_to_reference_homography'],dtype=np.float64)
        reference_raw=None;reference_entries=[];rows=[]
        if candidates:
            reference_raw=predict(capped,ref)
            rows=aligned_predictions(reference_raw['merged_predictions'],np.eye(3),ref.shape[:2],ref.shape[:2])
            copied=copy.deepcopy(candidates)
            for row in copied:row['support_tiles']=[]
            mapped=aligned_predictions(copied,matrix,image.shape[:2],ref.shape[:2])
            seeds=[dict(box_xyxy=[p[k] for k in ('left','top','right','bottom')],confidence=p['confidence'],class_id=p['class_id']) for p in mapped]
            reference_entries=predict_seed_views(capped,ref,seeds)
            for entry in reference_entries:
                for view in entry['views']:
                    for p in view:
                        if p['confidence']>.25 and complete(p,ref.shape[:2]):
                            rows.append(dict(zip(('left','top','right','bottom'),p['box_xyxy']),confidence=p['confidence'],class_id=p['class_id'],valid_warp_fraction=1.,support_tiles=[]))
        if {p:sha(Path(p)) for p in inputs}!=inputs:raise ValueError('input_changed_during_student_review')
        output,added=append_verified_student(original,candidates,rows,matrix)
        output['teacher_student_policy'].update(added_hints=len(added),confirmed_native_candidates=len(candidates),
            reference_inference_performed=bool(candidates),fallback_reason=None)
        output['teacher_student_evidence']=dict(teacher_case=old_case,student_case=new_case,fusion=fusion,
            student_reference_predictions=reference_raw,matched_student_reference_views=reference_entries,inputs=inputs)
        return output
    except Exception as error:
        original['teacher_student_policy']['fallback_reason']=type(error).__name__+': '+str(error)
        return original
