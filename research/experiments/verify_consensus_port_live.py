"""Real button path plus fresh initial geometry on two fixed additional sources."""
import copy,json,os,shutil,sys,time,unittest
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
OUT=ROOT/'artifacts/consensus_port_live_20261002'
sys.path[:0]=[str(REPO),str(REPO/'prototype')]
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',
    YOLO_CONFIG_DIR=str(OUT/'model_config'),PYTHONPROJECT10_DINO_CACHE=str(OUT/'dino_cache'))

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
    if OUT.exists():raise FileExistsError('Fresh output required')
    (OUT/'model_config/Ultralytics').mkdir(parents=True)
    import torch
    torch.set_num_threads(4)
    import cv2,numpy as np
    import assembly_auto_review_dino_v2 as entry
    gui=entry.implementation
    from inspection_agent.consensus_port_rescue import POLICY_ID,run_consensus_port_rescue,render_consensus_overlay
    from inspection_agent.port_rescue_gui import snapshot,payload_is_current,finish_port_rescue
    from inspection_agent.optional_port_crop_review import sha,SCENE
    from core_port_resolution_ab_20261002 import DATA,score
    os.environ['YOLO_CONFIG_DIR']=str(OUT/'model_config')
    suite=unittest.TestSuite()
    for name in ('test_consensus_port_rescue.py','test_precision_port_rescue.py','test_bounded_port_rescue.py',
        'test_independent_port_rescue.py','test_port_crop_gui_bridge.py','test_port_state_hint.py','test_port_tiling.py',
        'test_inspection_agent.py','test_inspection_agent_gui_contract.py','test_local_evidence_bridge.py','test_local_review_gui_bridge.py'):
        suite.addTests(unittest.defaultTestLoader.discover(str(REPO/'tests'),pattern=name))
    suite.addTests(unittest.TestLoader().discover(str(ROOT/'experiments'),pattern='test_port_rescue_freshness.py'))
    tests=unittest.TextTestRunner(verbosity=1).run(suite);assert tests.wasSuccessful()
    app=gui.QApplication.instance() or gui.QApplication([]);window=gui.DINOReview();rows=[]
    manifest=load(ROOT/'artifacts/rescue_mainline_ab_20261002_v2/report.json')
    protected={};started=time.monotonic()
    initial_names=sorted(c['image'] for c in load(ROOT/'artifacts/core_precision_full_inner_20261002/report.json')['cases'] if c['image'].startswith('disconnected_'))[:2]
    save(OUT/'protocol.json',dict(gui_sources=[c['image'] for c in manifest['cases']],new_initial_sources=initial_names,
        new_initial_selection='first two sorted original-group heldout disconnected sources',
        primary_limit=5,extra_limit=5,rule_frozen=True,no_sam_run=True,labels_used_for_selection=False))
    try:
        assert not window.rescue_switch.isChecked() and not window.rescue_supplement_switch.isChecked()
        for case in manifest['cases']:
            original=Path(case['output']);file=original/'report_off.json';protected[str(file)]=sha(file)
            report=load(file);target=OUT/Path(case['image']).stem;target.mkdir()
            shutil.copy2(original/'aligned.jpg',target/'aligned.jpg')
            window.current_output=target;window.reference.setText(report['reference']);window.inspection.setText(report['inspection'])
            window._write_report(report);window.port_scene_combo.setCurrentIndex(1)
            window.rescue_switch.setChecked(True);window.rescue_supplement_switch.setChecked(True)
            window._update_port_controls();assert window.rescue_button.isEnabled()
            assert snapshot(window)['supplementary'] and snapshot(window)['policy']==POLICY_ID
            before=sha(target/'report.json')
            old=dict(token='old',snapshot=snapshot(window),result={},evidence='unused');finish_port_rescue(window,old)
            assert sha(target/'report.json')==before
            window.rescue_button.click();assert window.rescue_worker is not None
            assert not window.rescue_supplement_switch.isEnabled()
            deadline=time.monotonic()+180
            while window.rescue_worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.02)
            assert window.rescue_worker is None,'worker timeout'
            app.processEvents();saved=load(target/'report.json');result=saved['independent_port_rescue']
            assert result['supplementary_policy']['policy_id']==POLICY_ID
            assert len(result['rescue_hints'])<=5 and len(result['supplementary_hints'])<=5
            previous=load(ROOT/'artifacts/precision_port_gui_20261002_v2'/Path(case['image']).stem/'report.json')['independent_port_rescue']
            assert result['rescue_hints']==previous['rescue_hints'],'Primary changed'
            assert saved['review_regions']==report['review_regions'] and not result['automatic_fault_verdict']
            binding=snapshot(window);window._rescue_token='switchtest'
            payload=dict(token='switchtest',snapshot=binding);assert payload_is_current(window,payload)
            window.rescue_supplement_switch.setChecked(False);assert not payload_is_current(window,payload)
            row=dict(image=case['image'],mode='actual_gui',status=result['status'],primary=len(result['rescue_hints']),
                additional=len(result['supplementary_hints']),reason=result['fallback_reason'],evidence=result['evidence_path'],
                geometry_report=report)
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='geometry_report'}),flush=True)
        old_out=gui.adaptive.robust.auto.base.OUT
        reference=Path(manifest['cases'][0]['output']);reference=Path(load(reference/'report_off.json')['reference'])
        try:
            for name in initial_names:
                source=DATA/'images/train01'/name;target=OUT/(Path(name).stem+'_new_initial');target.mkdir()
                gui.adaptive.robust.auto.base.OUT=target;cv2.setRNGSeed(0)
                worker=gui.InitialReviewWorker(reference,source,[[.03,.04,.97,.96]])
                payloads=[];worker.completed.connect(payloads.append);worker.run()
                assert len(payloads)==1 and payloads[0]['status']=='ready_for_sam3',payloads
                initial=payloads[0];report=initial['report'];output=Path(initial['output'])
                save(output/'initial_report.json',report);before=copy.deepcopy(report)
                os.environ['YOLO_CONFIG_DIR']=str(OUT/'model_config')
                result=run_consensus_port_rescue(report,project=REPO,enabled=True,scene=SCENE,supplementary_enabled=True)
                assert report==before and result['parents']==report['review_regions']
                save(output/'consensus_evidence.json',result)
                if result['status']=='applied':render_consensus_overlay(output/'aligned.jpg',result,output/'consensus_overlay.jpg')
                row=dict(image=name,mode='fresh_initial_without_sam',status=result['status'],primary=len(result['rescue_hints']),
                    additional=len(result['supplementary_hints']),reason=result['fallback_reason'],
                    evidence=str(output/'consensus_evidence.json'),geometry_report=report,
                    sam_fusion_pending=True,original_decision_unchanged=report['decision'])
                rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='geometry_report'}),flush=True)
        finally:gui.adaptive.robust.auto.base.OUT=old_out
        # All labels accessed only after all inference and selections finish.
        for row in rows:
            evidence_path=Path(row['evidence'])
            result=load(evidence_path/'evidence.json' if row['mode']=='actual_gui' else evidence_path)
            result=result['result'] if row['mode']=='actual_gui' else result
            report=row.pop('geometry_report');matrix=np.asarray(report['alignment']['source_to_reference_homography'],dtype=np.float64)
            split='test01' if row['mode']=='actual_gui' else 'train01';targets=[]
            for line in (DATA/'labels'/split/(Path(row['image']).stem+'.txt')).read_text(encoding='utf-8').splitlines():
                cls,cx,cy,w,h=map(float,line.split())
                if cls not in (3,4):continue
                l,t,r,b=(cx-w/2)*3648,(cy-h/2)*2736,(cx+w/2)*3648,(cy+h/2)*2736
                q=cv2.perspectiveTransform(np.float32([[[l,t],[r,t],[r,b],[l,b]]]),matrix).reshape(-1,2)
                targets.append(dict(class_id=int(cls)-3,box=[q[:,0].min(),q[:,1].min(),q[:,0].max(),q[:,1].max()]))
            adapt=lambda hints:[dict(class_id=h['box']['class_id'],box_xyxy=[h['box'][k] for k in ('left','top','right','bottom')]) for h in hints]
            row['metrics']=dict(primary=score(adapt(result['rescue_hints']),targets),combined=score(adapt(result['rescue_hints']+result['supplementary_hints']),targets))
        assert {p:sha(Path(p)) for p in protected}==protected
        save(OUT/'report.json',dict(status='complete',tests=tests.testsRun,cases=rows,primary_gui_default_preserved=True,
            extra_default_off=True,fresh_source_reference_inference=True,new_initial_dino=True,new_sam=False,
            old_reports_unchanged=True,seconds=round(time.monotonic()-started,2)))
    finally:window.close();app.processEvents()

if __name__=='__main__':main()
