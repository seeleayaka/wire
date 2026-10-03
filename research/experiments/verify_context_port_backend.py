"""Parity on all fixed cases; fresh API/geometry checks, no source label crops."""
import copy,json,os,sys,time,unittest
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');OUT=ROOT/'artifacts/context_port_backend_20261002'
sys.path[:0]=[str(REPO),str(REPO/'prototype')]
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',YOLO_CONFIG_DIR=str(OUT/'config'),PYTHONPROJECT10_DINO_CACHE=str(OUT/'dino_cache'))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    if OUT.exists():raise FileExistsError('Fresh output needed')
    (OUT/'config/Ultralytics').mkdir(parents=True)
    import torch;torch.set_num_threads(4)
    from inspection_agent.context_port_recheck import recheck_proposals,confirm_rechecks,run_context_port_recheck,render_consensus_overlay
    from inspection_agent.optional_port_crop_review import sha,SCENE
    from core_port_recheck_policy import proposals,confirmed
    from core_port_resolution_ab_20261002 import DATA,score
    os.environ['YOLO_CONFIG_DIR']=str(OUT/'config')
    suite=unittest.defaultTestLoader.discover(str(REPO/'tests'),pattern='test_context_port_recheck.py')
    tests=unittest.TextTestRunner().run(suite);assert tests.wasSuccessful()
    count=0
    for stage in ('train','inner','outer'):
        directory=ROOT/'artifacts/core_port_recheck_20261002'/stage
        for file in directory.glob('*_zoom_predictions.json'):
            case=load(file);assert recheck_proposals(case['predictions'])==proposals(case['predictions'])
            formal=confirm_rechecks(case['zoom_evidence'],case['predictions']['source_shape']);experimental=confirmed(case,'strict')
            assert formal==experimental;count+=1
    assert count==110 # 32 + 48 + 30
    save(OUT/'protocol.json',dict(parity_cases=count,tests=tests.testsRun,new_live_sources=['disconnected_028.JPG'],
        live_positive_selection='first sorted inner case with frozen-policy gain; diagnostic integration check, not independent metric',
        additional_real_sam_sources=['normal_006.JPG','disconnected_005.JPG'],new_sam=False))
    import cv2,numpy as np
    import assembly_auto_review_dino_v2 as entry
    gui=entry.implementation;app=gui.QApplication.instance() or gui.QApplication([])
    manifest=load(ROOT/'artifacts/rescue_mainline_ab_20261002_v2/report.json');rows=[]
    reference=None
    # Prior complete SAM reports stay protected, but model evidence is fresh this run.
    for item in manifest['cases']:
        if item['image'] not in ('normal_006.JPG','disconnected_005.JPG'):continue
        file=Path(item['output'])/'report_off.json';report=load(file);before=sha(file);reference=Path(report['reference'])
        folder=OUT/Path(item['image']).stem;folder.mkdir()
        output=run_context_port_recheck(report,project=REPO,enabled=True,scene=SCENE,supplementary_enabled=True)
        save(folder/'evidence.json',output);assert sha(file)==before
        rows.append(dict(image=item['image'],status=output['status'],primary=len(output['rescue_hints']),supplementary=len(output['supplementary_hints']),recheck=output['recheck_policy'],geometry_report=report,evidence=str(folder/'evidence.json')))
        print(json.dumps({k:v for k,v in rows[-1].items() if k!='geometry_report'}),flush=True)
    name='disconnected_028.JPG';target=OUT/'fresh_initial_028';target.mkdir();old=gui.adaptive.robust.auto.base.OUT
    try:
        gui.adaptive.robust.auto.base.OUT=target;cv2.setRNGSeed(0)
        worker=gui.InitialReviewWorker(reference,DATA/'images/train01'/name,[[.03,.04,.97,.96]])
        payloads=[];worker.completed.connect(payloads.append);worker.run()
        assert len(payloads)==1 and payloads[0]['status']=='ready_for_sam3'
        report=payloads[0]['report'];directory=Path(payloads[0]['output']);save(directory/'initial_report.json',report);before=copy.deepcopy(report)
        output=run_context_port_recheck(report,project=REPO,enabled=True,scene=SCENE,supplementary_enabled=True)
        save(directory/'evidence.json',output);assert report==before
        if output['status']=='applied':render_consensus_overlay(directory/'aligned.jpg',output,directory/'overlay.jpg')
        rows.append(dict(image=name,status=output['status'],primary=len(output['rescue_hints']),supplementary=len(output['supplementary_hints']),recheck=output['recheck_policy'],geometry_report=report,evidence=str(directory/'evidence.json'),sam_pending=True))
        print(json.dumps({k:v for k,v in rows[-1].items() if k!='geometry_report'}),flush=True)
    finally:gui.adaptive.robust.auto.base.OUT=old
    for row in rows:
        report=row.pop('geometry_report');output=load(Path(row['evidence']));targets=[]
        split='train01' if row['image']==name else 'test01';matrix=np.asarray(report['alignment']['source_to_reference_homography'],dtype=np.float64)
        for line in (DATA/'labels'/split/(Path(row['image']).stem+'.txt')).read_text(encoding='utf-8').splitlines():
            cls,cx,cy,w,h=map(float,line.split())
            if cls not in (3,4):continue
            l,t,r,b=(cx-w/2)*3648,(cy-h/2)*2736,(cx+w/2)*3648,(cy+h/2)*2736
            q=cv2.perspectiveTransform(np.float32([[[l,t],[r,t],[r,b],[l,b]]]),matrix).reshape(-1,2)
            targets.append(dict(class_id=int(cls)-3,box=[q[:,0].min(),q[:,1].min(),q[:,0].max(),q[:,1].max()]))
        rows_to_score=lambda hints:[dict(class_id=h['box']['class_id'],box_xyxy=[h['box'][k] for k in ('left','top','right','bottom')]) for h in hints]
        hints=output['rescue_hints']+output['supplementary_hints'];row['metrics']=score(rows_to_score(hints),targets)
        oldhints=[h for h in hints if h.get('evidence_tier')!='contextual_double_crop_model_cue_manual_review']
        row['baseline_metrics']=score(rows_to_score(oldhints),targets)
    save(OUT/'report.json',dict(status='complete',cases=rows,parity_cases=count,tests=tests.testsRun,new_sam=False,original_reports_unchanged=True))
    app.processEvents()
if __name__=='__main__':main()
