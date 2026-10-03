"""Default-off student complements; preserve teacher cues and all existing gates."""
import copy,json,math
from pathlib import Path
from inspection_agent.context_port_recheck import (run_context_port_recheck,native_baseline,
    confirm_rechecks,recheck_proposals,predict_seed_views,complete)
from inspection_agent.independent_port_rescue import select_rescue
from inspection_agent.optional_port_crop_review import sha,read_image,predict,aligned_predictions
from inspection_agent.port_tiling import box_iou

POLICY_ID='teacher_preserved_student_cross_support_v1_20261003'
TEACHER_SHA='9ed5ae77c940a78869e084639c55294b05fc65a78e3723f84af2fdc8189ed824'
STUDENT_SHA='f519a566d98756d988372c3f24dd827480ea41ac7cdb9fd12a97a426756d55b4'
MANIFEST_NAME='teacher_student_port_support_20261003.json'
MANIFEST_SHA='362fe4ae4a4d9f14ceefee4bd2184fd54efe1de18a02a557583b669eb523196e'
STUDENT_RELATIVE='output/port_student_multiscale_20261003/last.pt'
TEACHER_RELATIVE='output/port_crop_training_fixed_20260929/full/runs/rectports/weights/best.pt'

def support_runtime_fingerprint(project):
    root=Path(project)
    return {key:sha(path) for key,path in (
        ('teacher',root/TEACHER_RELATIVE),('student',root/STUDENT_RELATIVE),
        ('manifest',root/'config'/MANIFEST_NAME),('support_code',Path(__file__)),
        ('context_code',root/'inspection_agent/context_port_recheck.py'),
        ('calibration',root/'config/port_crop_calibration_frozen_20260930.json'))}

def native_selection(case):
    raw=case['predictions'];rows=[p for p in raw['merged_predictions'] if p['confidence']>.25]
    rows.sort(key=lambda p:(-p['confidence'],(p['box_xyxy'][2]-p['box_xyxy'][0])*(p['box_xyxy'][3]-p['box_xyxy'][1]),*p['box_xyxy'],p['class_id']))
    primary=(rows[:1]+[p for p in rows[1:] if p['confidence']>.5][:4]) if rows else []
    primary=[p for p in primary if p['confidence']>.5 and complete(p,raw['source_shape'])]
    native=native_baseline(raw);extra=native[len(primary):];rechecks=[]
    for row in confirm_rechecks(case['zoom_evidence'],raw['source_shape']):
        if len(rechecks)>=5-len(extra):break
        if not any(box_iou(row['box_xyxy'],p['box_xyxy'])>=.5 for p in native+rechecks):rechecks.append(row)
    return dict(primary=copy.deepcopy(primary),supplementary=copy.deepcopy(extra),zoom=copy.deepcopy(rechecks),
        all_predictions=copy.deepcopy(native+rechecks))

def valid(p):
    box=p.get('box_xyxy',[]);score=p.get('confidence')
    return (type(p.get('class_id')) is int and p['class_id'] in (0,1) and len(box)==4 and
        all(type(v) in (int,float) and math.isfinite(v) for v in box) and 0<=box[0]<box[2] and 0<=box[1]<box[3] and
        type(score) in (int,float) and math.isfinite(score) and 0<=score<=1)

def complementary_candidates(teacher,student):
    base=native_selection(teacher);result=copy.deepcopy(base);result.update(student_additions=[],fallback_reason=None)
    if not teacher.get('source_sha256') or any(teacher.get(k)!=student.get(k) for k in ('image','source_sha256')):
        result['fallback_reason']='source_identity_mismatch';return result
    try:
        if teacher['predictions']['source_shape']!=student['predictions']['source_shape']:raise ValueError('source_geometry_mismatch')
        selected=native_selection(student);remaining=5-len(base['supplementary'])-len(base['zoom'])
        for row in sorted(selected['all_predictions'],key=lambda p:(-p['confidence'],*p['box_xyxy'],p['class_id'])):
            if len(result['student_additions'])>=remaining:break
            if not valid(row) or row['confidence']<=.75 or not complete(row,teacher['predictions']['source_shape']):continue
            if any(box_iou(row['box_xyxy'],p['box_xyxy'])>=.5 for p in result['all_predictions']):continue
            votes=sorted({p['source_tile'] for p in student['predictions']['edge_kept_predictions']
                if valid(p) and p['confidence']>.5 and p['class_id']==row['class_id'] and box_iou(p['box_xyxy'],row['box_xyxy'])>=.5})
            rechecked=(row in selected['zoom'] and len(row.get('zoom_view_scores',[]))==2 and min(row['zoom_view_scores'])>.75
                and len(row.get('zoom_windows',[]))==2 and row['zoom_windows'][0]!=row['zoom_windows'][1])
            if len(votes)<2 and not rechecked:continue
            support=[p for p in teacher['predictions']['merged_predictions'] if valid(p) and p['class_id']==row['class_id']
                and p['confidence']>.25 and box_iou(p['box_xyxy'],row['box_xyxy'])>=.5]
            if not support:continue
            addition=copy.deepcopy(row);addition.update(evidence_tier='student_complement_manual_review',
                student_support_tiles=votes,student_double_recheck=rechecked,
                teacher_support_maximum_score=max(p['confidence'] for p in support),
                teacher_weight_sha256=teacher.get('weight_sha256'),student_weight_sha256=student.get('weight_sha256'))
            result['student_additions'].append(addition);result['all_predictions'].append(addition)
        assert result['primary']==base['primary'] and result['all_predictions'][:len(base['all_predictions'])]==base['all_predictions']
        assert len(result['all_predictions'])<=10
    except (KeyError,TypeError,ValueError,AssertionError) as error:
        result=copy.deepcopy(base);result.update(student_additions=[],fallback_reason=type(error).__name__+': '+str(error))
    return result

def append_verified_student(original,candidates,reference_rows,matrix):
    output=copy.deepcopy(original);source=output['source_evidence'];reference=output['reference_evidence']
    native=copy.deepcopy(candidates)
    for row in native:row['support_tiles']=[]
    mapped=aligned_predictions(native,matrix,source['predictions']['source_shape'],reference['predictions']['source_shape'])
    refs=copy.deepcopy(reference['aligned_predictions'])+copy.deepcopy(reference_rows)
    old=output['existing_hints']+output['rescue_hints']+output['supplementary_hints'];added=[]
    coords=lambda p:[p[k] for k in ('left','top','right','bottom')]
    for _ in range(max(0,5-len(output['supplementary_hints']))):
        remaining=[p for p in mapped if not any(box_iou(coords(p),coords(h['box']))>=.5 for h in old+added)]
        hints,_=select_rescue(output['analysis_rois'],old+added,remaining,refs)
        if not hints:break
        for hint in hints:hint.update(evidence_tier='teacher_student_complement_manual_review',
            teacher_weight_sha256=TEACHER_SHA,student_weight_sha256=STUDENT_SHA,
            warning='Correlated model evidence only; reference non-detection does not prove a physical fault.')
        added.extend(hints)
    output['supplementary_hints'].extend(added)
    assert output['rescue_hints']==original['rescue_hints']
    assert output['supplementary_hints'][:len(original['supplementary_hints'])]==original['supplementary_hints']
    assert len(output['rescue_hints'])<=5 and len(output['supplementary_hints'])<=5
    return output,added

def run_teacher_student_review(report,*,project,enabled=False,scene='unknown',supplementary_enabled=False,student_enabled=False):
    original=run_context_port_recheck(report,project=project,enabled=enabled,scene=scene,supplementary_enabled=supplementary_enabled)
    original['teacher_student_policy']=dict(policy_id=POLICY_ID,enabled=bool(student_enabled),experimental=True,
        maximum_total_primary=5,maximum_total_supplementary=5,teacher_preserved=True,
        automatic_fault_verdict=False,added_hints=0,correlated_models_not_physical_evidence=True)
    if not student_enabled or not supplementary_enabled or original['status']!='applied' or len(original['supplementary_hints'])>=5:return original
    try:
        import torch,numpy as np
        from ultralytics import YOLO
        root=Path(project);manifest=root/'config'/MANIFEST_NAME
        if sha(manifest)!=MANIFEST_SHA:raise ValueError('student_provenance_manifest_mismatch')
        specification=json.loads(manifest.read_text(encoding='utf-8'))
        if specification['teacher_sha256']!=TEACHER_SHA or specification['student_sha256']!=STUDENT_SHA or specification['policy_id']!=POLICY_ID:raise ValueError('student_provenance_contract_mismatch')
        teacher=root/TEACHER_RELATIVE;student=root/STUDENT_RELATIVE
        inputs={str(p):sha(p) for p in (teacher,student,Path(report['inspection']),Path(report['reference']),manifest)}
        if inputs[str(teacher)]!=TEACHER_SHA or inputs[str(student)]!=STUDENT_SHA:raise ValueError('model_identity_mismatch')
        source=original['source_evidence'];reference=original['reference_evidence']
        if source['weight_sha256']!=TEACHER_SHA or source['source_sha256']!=inputs[str(Path(report['inspection']))] or source['reference_sha256']!=inputs[str(Path(report['reference']))]:raise ValueError('source_reference_identity_mismatch')
        old_case=dict(image=Path(report['inspection']).name,source_sha256=source['source_sha256'],weight_sha256=TEACHER_SHA,
            predictions=source['predictions'],zoom_evidence=original.get('recheck_evidence',{}).get('source_views',[]))
        image=read_image(report['inspection']);ref=read_image(report['reference']);torch.set_num_threads(4);model=YOLO(str(student))
        if model.task!='segment' or dict(model.names)!={0:'unplugged_plug',1:'unplugged_jack'}:raise ValueError('student_contract_mismatch')
        class Capped:
            def predict(self,*args,**kw):
                result=model.predict(*args,**kw);torch.set_num_threads(4);return result
        capped=Capped();raw=predict(capped,image)
        new_case=dict(image=old_case['image'],source_sha256=old_case['source_sha256'],weight_sha256=STUDENT_SHA,predictions=raw,zoom_evidence=[])
        selected=native_selection(new_case)
        if len(selected['supplementary'])<5:new_case['zoom_evidence']=predict_seed_views(capped,image,recheck_proposals(raw))
        fusion=complementary_candidates(old_case,new_case);candidates=fusion['student_additions']
        if fusion['fallback_reason']:raise ValueError('native_fusion: '+fusion['fallback_reason'])
        matrix=np.asarray(report['alignment']['source_to_reference_homography'],dtype=np.float64);reference_raw=None;reference_entries=[];rows=[]
        if candidates:
            reference_raw=predict(capped,ref);rows=aligned_predictions(reference_raw['merged_predictions'],np.eye(3),ref.shape[:2],ref.shape[:2])
            native=copy.deepcopy(candidates)
            for row in native:row['support_tiles']=[]
            mapped=aligned_predictions(native,matrix,image.shape[:2],ref.shape[:2])
            seeds=[dict(box_xyxy=[p[k] for k in ('left','top','right','bottom')],confidence=p['confidence'],class_id=p['class_id']) for p in mapped]
            reference_entries=predict_seed_views(capped,ref,seeds)
            for entry in reference_entries:
                for view in entry['views']:
                    for p in view:
                        if p['confidence']>.25 and complete(p,ref.shape[:2]):rows.append(dict(zip(('left','top','right','bottom'),p['box_xyxy']),
                            confidence=p['confidence'],class_id=p['class_id'],valid_warp_fraction=1.,support_tiles=[]))
        if {p:sha(Path(p)) for p in inputs}!=inputs:raise ValueError('input_changed_during_student_review')
        output,added=append_verified_student(original,candidates,rows,matrix)
        output['teacher_student_policy'].update(added_hints=len(added),confirmed_native_candidates=len(candidates),
            reference_inference_performed=bool(candidates),fallback_reason=None,provenance_manifest_sha256=MANIFEST_SHA)
        output['teacher_student_evidence']=dict(teacher_case=old_case,student_case=new_case,fusion=fusion,
            student_reference_predictions=reference_raw,matched_student_reference_views=reference_entries,inputs=inputs)
        return output
    except Exception as error:
        original['teacher_student_policy']['fallback_reason']=type(error).__name__+': '+str(error)
        return original
