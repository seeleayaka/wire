"""Fresh real registration/DINO and source/reference diagnostic, no deployment."""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]; REPO=Path('E:/PythonProject10')
OUT=ROOT/'artifacts/port_residual_feature_live_20261003'
TRIAL=ROOT/'artifacts/port_residual_feature_support_20261003'
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
sys.path[:0]=[str(REPO),str(REPO/'prototype')]
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',
                  YOLO_CONFIG_DIR=str(OUT/'config'),PYTHONPROJECT10_DINO_CACHE=str(OUT/'dino_cache'))
from inspection_agent.optional_port_crop_review import sha,SCENE
from inspection_agent.context_port_recheck import render_consensus_overlay
from port_residual_feature_backend import run_residual_review, WEIGHT, WEIGHT_SHA
from audit_port_multiscale_acceptance import metric


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temp.replace(path)


def main():
    if OUT.exists():raise FileExistsError('Preserve existing evidence')
    for stage in ('train','extended','inner','outer'):assert load(TRIAL/stage/'report.json')['qualifies']
    training=load(TRIAL/'train/report.json')
    gains=sorted(row['image'] for row in training['cases'] if row['feature_additions']>0)
    assert gains
    burden=load(ROOT/'artifacts/port_extended_training_controls_20261003/report.json')
    negatives=sorted(row['image'] for row in burden['cases'] if row['metrics']['current']['predictions']>0)
    names=[gains[0]]+negatives
    reference=DATA/'images/train01/normal_073.JPG'
    assert sha(WEIGHT)==WEIGHT_SHA
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('port_residual_feature_backend.py'),
        Path(__file__).with_name('port_residual_feature_support.py'),WEIGHT,reference)}
    for name in names:pins[str(DATA/'images/train01'/name)]=sha(DATA/'images/train01'/name)
    (OUT/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    save(OUT/'protocol.json',dict(pins=pins,images=names,positive_selection='First training-control case with residual gain',
        negative_selection='ALL broader training source-only current cues, no per-image policy',
        new_initial_registration_and_dino=True,new_source_reference_model_inference=True,new_sam=False,
        sam_pending=True,previously_seen_training_diagnostic=True,automatic_deployment=False,field_accuracy=False))
    import torch;torch.set_num_threads(4)
    import cv2,numpy as np
    import assembly_auto_review_dino_v2 as entry
    gui=entry.implementation;app=gui.QApplication.instance() or gui.QApplication([])
    before_output=gui.adaptive.robust.auto.base.OUT;records=[];started=time.monotonic()
    try:
        for index,name in enumerate(names,1):
            target=OUT/Path(name).stem;target.mkdir();gui.adaptive.robust.auto.base.OUT=target
            save(OUT/'progress.json',dict(status='running',phase='fresh_initial',image=name,completed=index-1,total=len(names)))
            cv2.setRNGSeed(0);worker=gui.InitialReviewWorker(reference,DATA/'images/train01'/name,[[.03,.04,.97,.96]])
            payloads=[];worker.completed.connect(payloads.append);worker.run()
            assert len(payloads)==1
            payload=payloads[0];report=payload['report'];directory=Path(payload['output']);save(directory/'initial_report.json',report)
            if payload['status']!='ready_for_sam3':
                records.append(dict(image=name,status=payload['status'],added=0,live_gain=False));continue
            before=copy.deepcopy(report);protected=sha(directory/'initial_report.json')
            save(OUT/'progress.json',dict(status='running',phase='fresh_port_source_reference',image=name,completed=index-1,total=len(names)))
            result=run_residual_review(report,project=REPO,enabled=True,scene=SCENE,
                supplementary_enabled=True,student_enabled=True,feature_enabled=True)
            assert report==before and sha(directory/'initial_report.json')==protected
            save(directory/'evidence.json',result)
            if result['status']=='applied':render_consensus_overlay(directory/'aligned.jpg',result,directory/'overlay.jpg')
            hints=result['rescue_hints']+result['supplementary_hints'];old=[h for h in hints if h.get('evidence_tier')!='feature_residual_manual_review']
            targets=[];matrix=np.asarray(report['alignment']['source_to_reference_homography'],dtype=np.float64)
            for line in (DATA/'labels/train01'/(Path(name).stem+'.txt')).read_text(encoding='utf-8').splitlines():
                cls,cx,cy,bw,bh=map(float,line.split())
                if cls not in (3,4):continue
                l,t,r,b=(cx-bw/2)*3648,(cy-bh/2)*2736,(cx+bw/2)*3648,(cy+bh/2)*2736
                pts=cv2.perspectiveTransform(np.float32([[[l,t],[r,t],[r,b],[l,b]]]),matrix).reshape(-1,2)
                targets.append(dict(class_id=int(cls)-3,box=[pts[:,0].min(),pts[:,1].min(),pts[:,0].max(),pts[:,1].max()]))
            convert=lambda hs:[dict(class_id=h['box']['class_id'],box_xyxy=[h['box'][k] for k in ('left','top','right','bottom')]) for h in hs]
            baseline,current=metric(convert(old),targets),metric(convert(hints),targets)
            policy=result['feature_residual_policy']
            record=dict(image=name,status=result['status'],added=policy['added_hints'],feature_policy=policy,
                baseline=baseline,metrics=current,live_gain=current['tp']>baseline['tp'],
                evidence=str(directory/'evidence.json'),initial_report=str(directory/'initial_report.json'),sam_pending=True)
            records.append(record);print(json.dumps(record),flush=True)
            assert len(result['rescue_hints'])<=5 and len(result['supplementary_hints'])<=5
            save(OUT/'partial.json',dict(cases=records,seconds=round(time.monotonic()-started,2)))
        assert {p:sha(Path(p)) for p in pins}==pins
        positive=records[0];passed=positive.get('live_gain',False) and all(
            row.get('feature_policy',{}).get('fallback_reason') is None and row['status']=='applied' and
            row['metrics']['unmatched']<=row['baseline']['unmatched'] for row in records)
        save(OUT/'report.json',dict(status='complete',qualifies_live_diagnostic=passed,cases=records,
            seconds=round(time.monotonic()-started,2),new_sam=False,sam_pending=True,
            old_models_and_reports_unchanged=True,automatic_deployment=False,field_accuracy=False))
        save(OUT/'progress.json',dict(status='complete',qualifies_live_diagnostic=passed,completed=len(records),total=len(names)))
    except Exception as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)))
        raise
    finally:
        gui.adaptive.robust.auto.base.OUT=before_output;app.processEvents()


if __name__=='__main__':main()
