"""Fresh gated source/reference review; prior complete SAM reports stay protected."""
import copy,json,os,shutil,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
OUT=ROOT/'artifacts/teacher_student_port_live_20261003'
sys.path[:0]=[str(REPO),str(REPO/'prototype')]
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',
    YOLO_CONFIG_DIR=str(OUT/'config'),PYTHONPROJECT10_DINO_CACHE=str(OUT/'dino_cache'))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    if OUT.exists():raise FileExistsError('Fresh output required')
    decision=load(ROOT/'artifacts/teacher_student_port_20261003/training_decision.json')
    assert decision['accepted'] and decision['mode']=='cross_model_supported'
    for split in ('inner','outer'):
        assert load(ROOT/'artifacts/teacher_student_port_20261003'/split/'report.json')['qualifies'][decision['mode']]
    (OUT/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    import torch;torch.set_num_threads(4)
    import cv2,numpy as np
    from teacher_student_port_backend import run_teacher_student_review,STUDENT_WEIGHT,STUDENT_SHA,TEACHER_SHA
    from inspection_agent.context_port_recheck import render_consensus_overlay
    from inspection_agent.optional_port_crop_review import sha,SCENE
    from audit_port_multiscale_acceptance import metric
    import assembly_auto_review_dino_v2 as entry
    gui=entry.implementation;app=gui.QApplication.instance() or gui.QApplication([])
    training=load(ROOT/'artifacts/teacher_student_port_20261003/train/report.json')
    gains=sorted(row['image'] for row in training['cases'] if row['additions'][decision['mode']]>0)
    assert gains;name=gains[0]
    save(OUT/'protocol.json',dict(training_selected_positive=name,selection='first training-control case with accepted frozen fusion addition',
        additional_complete_sam_cases=['normal_006.JPG','disconnected_005.JPG'],
        student_sha256=STUDENT_SHA,teacher_sha256=TEACHER_SHA,new_sam=False,
        sam_pending_on_fresh_initial=True,not_unseen_accuracy=True,source_reference_model_inference_fresh=True))
    manifest=load(ROOT/'artifacts/rescue_mainline_ab_20261002_v2/report.json');cases=[];reference=None;started=time.monotonic()
    def execute(report,image,folder,split,protected=None,sam_pending=False):
        before=copy.deepcopy(report);protected_sha=sha(protected) if protected else None;tick=time.monotonic()
        result=run_teacher_student_review(report,project=REPO,enabled=True,scene=SCENE,supplementary_enabled=True,student_enabled=True)
        assert report==before
        if protected:assert sha(protected)==protected_sha
        assert len(result['rescue_hints'])<=5 and len(result['supplementary_hints'])<=5
        save(folder/'evidence.json',result)
        if result['status']=='applied':
            aligned=read_aligned(report,folder)
            render_consensus_overlay(aligned,result,folder/'overlay.jpg')
        row=dict(image=image,status=result['status'],fallback_reason=result.get('fallback_reason'),
            primary=len(result['rescue_hints']),supplementary=len(result['supplementary_hints']),
            student_policy=result['teacher_student_policy'],evidence=str(folder/'evidence.json'),
            seconds=round(time.monotonic()-tick,2),sam_pending=sam_pending,split=split,geometry_report=report)
        cases.append(row);print(json.dumps({k:v for k,v in row.items() if k!='geometry_report'}),flush=True)
    def read_aligned(report,folder):
        from inspection_agent.optional_port_crop_review import read_image
        image=read_image(report['inspection']);matrix=np.asarray(report['alignment']['source_to_reference_homography'],dtype=np.float64)
        aligned=cv2.warpPerspective(image,matrix,(3648,2736));path=folder/'aligned.jpg'
        ok,encoded=cv2.imencode('.jpg',aligned,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok;encoded.tofile(str(path));return path
    for item in manifest['cases']:
        if item['image'] not in ('normal_006.JPG','disconnected_005.JPG'):continue
        file=Path(item['output'])/'report_off.json';report=load(file);reference=Path(report['reference'])
        folder=OUT/Path(item['image']).stem;folder.mkdir();execute(report,item['image'],folder,'test01',file)
    assert reference is not None
    target=OUT/'fresh_initial_train';target.mkdir();old=gui.adaptive.robust.auto.base.OUT
    DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
    try:
        gui.adaptive.robust.auto.base.OUT=target;cv2.setRNGSeed(0)
        worker=gui.InitialReviewWorker(reference,DATA/'images/train01'/name,[[.03,.04,.97,.96]])
        payloads=[];worker.completed.connect(payloads.append);worker.run()
        assert len(payloads)==1 and payloads[0]['status']=='ready_for_sam3'
        report=payloads[0]['report'];directory=Path(payloads[0]['output']);save(directory/'initial_report.json',report)
        execute(report,name,directory,'train01',sam_pending=True)
    finally:gui.adaptive.robust.auto.base.OUT=old
    for row in cases:
        report=row.pop('geometry_report');output=load(Path(row['evidence']));targets=[]
        matrix=np.asarray(report['alignment']['source_to_reference_homography'],dtype=np.float64)
        for line in (DATA/'labels'/row['split']/(Path(row['image']).stem+'.txt')).read_text(encoding='utf-8').splitlines():
            cls,cx,cy,w,h=map(float,line.split())
            if cls not in (3,4):continue
            l,t,r,b=(cx-w/2)*3648,(cy-h/2)*2736,(cx+w/2)*3648,(cy+h/2)*2736
            points=cv2.perspectiveTransform(np.float32([[[l,t],[r,t],[r,b],[l,b]]]),matrix).reshape(-1,2)
            targets.append(dict(class_id=int(cls)-3,box=[points[:,0].min(),points[:,1].min(),points[:,0].max(),points[:,1].max()]))
        hints=output['rescue_hints']+output['supplementary_hints'];old_hints=[h for h in hints if h.get('evidence_tier')!='teacher_student_complement_manual_review']
        converted=lambda rows:[dict(class_id=h['box']['class_id'],box_xyxy=[h['box'][k] for k in ('left','top','right','bottom')]) for h in rows]
        row['baseline_metrics']=metric(converted(old_hints),targets);row['metrics']=metric(converted(hints),targets)
    assert sha(STUDENT_WEIGHT)==STUDENT_SHA
    save(OUT/'report.json',dict(status='complete',cases=cases,seconds=round(time.monotonic()-started,2),
        new_sam=False,original_reports_unchanged=True,production_gui_not_changed=True,operator_confirmation=False,
        not_complete_fusion_acceptance=True,policy_hash=sha(ROOT/'experiments/teacher_student_port_policy.py'),
        backend_hash=sha(ROOT/'experiments/teacher_student_port_backend.py')))
    app.processEvents()
if __name__=='__main__':main()
