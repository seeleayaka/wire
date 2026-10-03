"""Training-selected real source/reference diagnostic; no deployment or new SAM."""
import copy,json,os,shutil,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
OUT=ROOT/'artifacts/port_resolution_live_20261003'
TRIAL=ROOT/'artifacts/port_high_resolution_support_20261003'
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
sys.path[:0]=[str(REPO),str(REPO/'prototype')]
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',
    YOLO_CONFIG_DIR=str(OUT/'config'),PYTHONPROJECT10_DINO_CACHE=str(OUT/'dino_cache'))
from inspection_agent.optional_port_crop_review import sha,SCENE
from inspection_agent.context_port_recheck import render_consensus_overlay
from inspection_agent.feature_residual_port_support import residual_runtime_fingerprint
from port_resolution_backend import run_resolution_review
from audit_port_multiscale_acceptance import metric,matches
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def main():
    if OUT.exists():raise FileExistsError('Preserve existing evidence')
    for mode in ('train','extended','inner','outer'):assert load(TRIAL/mode/'report.json')['qualifies']
    gains=sorted(r['image'] for r in load(TRIAL/'train/report.json')['cases'] if r['additions']>0)
    negatives=sorted(r['image'] for r in load(ROOT/'artifacts/port_extended_training_controls_20261003/report.json')['cases']
        if r['metrics']['current']['predictions']>0)
    names=[gains[0]]+negatives;reference=DATA/'images/train01/normal_073.JPG'
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('port_resolution_backend.py'),
        Path(__file__).with_name('port_resolution_support.py'),reference)}
    for name in names:pins[str(DATA/'images/train01'/name)]=sha(DATA/'images/train01'/name)
    frozen=residual_runtime_fingerprint(REPO)
    (OUT/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    save(OUT/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,images=names,
        first_training_gain=True,all_training_negative_cues=True,new_registration_dino=True,new_source_reference_inference=True,
        sam_pending=True,automatic_deployment=False,independent_accuracy=False,field_accuracy=False))
    import torch;torch.set_num_threads(4)
    import cv2,numpy as np
    import assembly_auto_review_dino_v2 as entry
    gui=entry.implementation;app=gui.QApplication.instance() or gui.QApplication([])
    previous=gui.adaptive.robust.auto.base.OUT;records=[];started=time.monotonic()
    try:
        for index,name in enumerate(names):
            target=OUT/Path(name).stem;target.mkdir();gui.adaptive.robust.auto.base.OUT=target
            save(OUT/'progress.json',dict(status='running',pid=os.getpid(),image=name,completed=index,total=len(names),phase='initial'))
            cv2.setRNGSeed(0);worker=gui.InitialReviewWorker(reference,DATA/'images/train01'/name,[[.03,.04,.97,.96]])
            payloads=[];worker.completed.connect(payloads.append);worker.run();assert len(payloads)==1
            payload=payloads[0];report=payload['report'];directory=Path(payload['output']);save(directory/'initial_report.json',report)
            assert payload['status']=='ready_for_sam3',payload['status']
            original=copy.deepcopy(report);protected=sha(directory/'initial_report.json')
            save(OUT/'progress.json',dict(status='running',pid=os.getpid(),image=name,completed=index,total=len(names),phase='ports'))
            result=run_resolution_review(report,project=REPO,enabled=True,scene=SCENE,supplementary_enabled=True,
                student_enabled=True,feature_enabled=True,resolution_enabled=True)
            assert report==original and sha(directory/'initial_report.json')==protected
            save(directory/'evidence.json',result)
            if result['status']=='applied':render_consensus_overlay(directory/'aligned.jpg',result,directory/'overlay.jpg')
            hints=result['rescue_hints']+result['supplementary_hints']
            old=[h for h in hints if h.get('evidence_tier')!='resolution_residual_manual_review']
            matrix=np.asarray(report['alignment']['source_to_reference_homography'],dtype=np.float64);targets=[]
            for line in (DATA/'labels/train01'/(Path(name).stem+'.txt')).read_text(encoding='utf-8').splitlines():
                cls,cx,cy,bw,bh=map(float,line.split())
                if cls not in (3,4):continue
                l,t,r,b=(cx-bw/2)*3648,(cy-bh/2)*2736,(cx+bw/2)*3648,(cy+bh/2)*2736
                pts=cv2.perspectiveTransform(np.float32([[[l,t],[r,t],[r,b],[l,b]]]),matrix).reshape(-1,2)
                targets.append(dict(class_id=int(cls)-3,box=[pts[:,0].min(),pts[:,1].min(),pts[:,0].max(),pts[:,1].max()]))
            convert=lambda hs:[dict(class_id=h['box']['class_id'],box_xyxy=[h['box'][k] for k in ('left','top','right','bottom')]) for h in hs]
            baseline,new=metric(convert(old),targets),metric(convert(hints),targets)
            loss=len(matches(convert(old),targets)[0]-matches(convert(hints),targets)[0])
            row=dict(image=name,status=result['status'],policy=result['resolution_policy'],baseline=baseline,metrics=new,
                lost=loss,live_gain=new['tp']>baseline['tp'],evidence=str(directory/'evidence.json'),
                initial_report=str(directory/'initial_report.json'),sam_pending=True)
            records.append(row);print(json.dumps(row),flush=True)
            assert len(result['rescue_hints'])<=5 and len(result['supplementary_hints'])<=5
            save(OUT/'partial.json',dict(cases=records,seconds=round(time.monotonic()-started,2)))
        assert {p:sha(Path(p)) for p in pins}==pins and residual_runtime_fingerprint(REPO)==frozen
        passed=records[0]['live_gain'] and all(r['status']=='applied' and not r['policy'].get('fallback_reason') and
            r['lost']==0 and r['metrics']['unmatched']<=r['baseline']['unmatched'] for r in records)
        save(OUT/'report.json',dict(status='complete',qualifies_live_diagnostic=passed,cases=records,
            seconds=round(time.monotonic()-started,2),sam_pending=True,automatic_deployment=False,field_accuracy=False))
        save(OUT/'progress.json',dict(status='complete',qualifies_live_diagnostic=passed,completed=len(records)))
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise
    finally:gui.adaptive.robust.auto.base.OUT=previous;app.processEvents()
if __name__=='__main__':main()
