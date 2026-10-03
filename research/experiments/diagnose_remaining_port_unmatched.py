"""No threshold changes: fresh source/reference check of the existing outer unmatched cue."""
import copy,json,os,shutil,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');OUT=ROOT/'artifacts/remaining_port_unmatched_20261003'
sys.path[:0]=[str(REPO),str(REPO/'prototype')]
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',YOLO_CONFIG_DIR=str(OUT/'config'),PYTHONPROJECT10_DINO_CACHE=str(OUT/'dino_cache'))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    if OUT.exists():raise FileExistsError('Fresh output required')
    (OUT/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    prior=load(ROOT/'artifacts/teacher_student_port_20261003/outer/report.json')
    names=sorted(p['image'] for p in prior['cases'] if p['metrics']['baseline']['unmatched'])
    assert names;name=names[0]
    save(OUT/'protocol.json',dict(sample=name,selection='first sorted existing outer source-only unmatched diagnostic',
        hypotheses=['static reference cue is vetoed by unchanged live gates','cue survives because reference model misses it or semantics differ'],
        thresholds_changed=False,independent_generalization=False,new_sam=False,no_annotation_inference=True))
    import torch;torch.set_num_threads(4)
    import cv2,numpy as np
    import assembly_auto_review_dino_v2 as entry
    gui=entry.implementation;app=gui.QApplication.instance() or gui.QApplication([])
    from inspection_agent.teacher_student_port_support import run_teacher_student_review
    from inspection_agent.context_port_recheck import render_consensus_overlay
    from inspection_agent.optional_port_crop_review import sha,SCENE
    from audit_port_multiscale_acceptance import metric
    DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
    source=DATA/'images/val01'/name;reference=DATA/'images/train01/normal_073.JPG';protected={str(p):sha(p) for p in (source,reference)}
    output=OUT/'initial';output.mkdir();old=gui.adaptive.robust.auto.base.OUT;started=time.monotonic()
    try:
        gui.adaptive.robust.auto.base.OUT=output;cv2.setRNGSeed(0)
        worker=gui.InitialReviewWorker(reference,source,[[.03,.04,.97,.96]]);payloads=[];worker.completed.connect(payloads.append);worker.run()
        assert len(payloads)==1 and payloads[0]['status']=='ready_for_sam3'
        report=payloads[0]['report'];folder=Path(payloads[0]['output']);save(folder/'initial_report.json',report);before=copy.deepcopy(report)
        result=run_teacher_student_review(report,project=REPO,enabled=True,scene=SCENE,supplementary_enabled=True,student_enabled=True)
        assert report==before and report['decision']=='sam3_fusion_running'
        save(folder/'evidence.json',result)
        if result['status']=='applied':render_consensus_overlay(folder/'aligned.jpg',result,folder/'overlay.jpg')
        # Evaluation only after actual gates and inference; transform ground-truth for aligned-cue scoring.
        targets=[];matrix=np.asarray(report['alignment']['source_to_reference_homography'],dtype=np.float64)
        label=DATA/'labels/val01'/(Path(name).stem+'.txt')
        for line in label.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,w,h=map(float,line.split())
            if cls not in (3,4):continue
            l,t,r,b=(cx-w/2)*3648,(cy-h/2)*2736,(cx+w/2)*3648,(cy+h/2)*2736
            points=cv2.perspectiveTransform(np.float32([[[l,t],[r,t],[r,b],[l,b]]]),matrix).reshape(-1,2)
            targets.append(dict(class_id=int(cls)-3,box=[points[:,0].min(),points[:,1].min(),points[:,0].max(),points[:,1].max()]))
        hints=result['rescue_hints']+result['supplementary_hints']
        converted=[dict(class_id=p['box']['class_id'],box_xyxy=[p['box'][k] for k in ('left','top','right','bottom')]) for p in hints]
        assert {p:sha(Path(p)) for p in protected}==protected
        save(OUT/'report.json',dict(status='complete',sample=name,port_status=result['status'],fallback_reason=result.get('fallback_reason'),
            source_only_old=next(p['metrics']['baseline'] for p in prior['cases'] if p['image']==name),live_aligned_port_metrics=metric(converted,targets),
            primary=len(result['rescue_hints']),supplementary=len(result['supplementary_hints']),student_policy=result['teacher_student_policy'],
            seconds=round(time.monotonic()-started,2),evidence=str(folder/'evidence.json'),initial_report=str(folder/'initial_report.json'),
            sam_pending=True,thresholds_unchanged=True,operator_confirmation=False,field_accuracy=False,source_reference_unchanged=True))
        print(json.dumps(load(OUT/'report.json')),flush=True)
    finally:gui.adaptive.robust.auto.base.OUT=old;app.processEvents()
if __name__=='__main__':main()
