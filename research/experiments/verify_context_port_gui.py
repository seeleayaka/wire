"""Final offscreen real-button/backend/callback checks on two complete SAM reports."""
import json,os,shutil,sys,time,unittest
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');OUT=ROOT/'artifacts/context_port_gui_20261002_v2'
sys.path[:0]=[str(REPO),str(REPO/'prototype')]
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',YOLO_CONFIG_DIR=str(OUT/'config'))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    if OUT.exists():raise FileExistsError('Fresh output needed')
    (OUT/'config/Ultralytics').mkdir(parents=True)
    import torch;torch.set_num_threads(4)
    import assembly_auto_review_dino_v2 as entry
    gui=entry.implementation
    from inspection_agent.context_port_recheck import POLICY_ID
    from inspection_agent.optional_port_crop_review import sha
    from inspection_agent.port_rescue_gui import snapshot,payload_is_current,finish_port_rescue
    suite=unittest.TestSuite()
    for name in ('test_context_port_recheck.py','test_consensus_port_rescue.py','test_precision_port_rescue.py','test_bounded_port_rescue.py','test_independent_port_rescue.py','test_port_crop_gui_bridge.py','test_port_state_hint.py','test_port_tiling.py','test_inspection_agent.py','test_inspection_agent_gui_contract.py','test_local_evidence_bridge.py','test_local_review_gui_bridge.py'):
        suite.addTests(unittest.defaultTestLoader.discover(str(REPO/'tests'),pattern=name))
    suite.addTests(unittest.TestLoader().discover(str(ROOT/'experiments'),pattern='test_port_rescue_freshness.py'))
    result=unittest.TextTestRunner().run(suite);assert result.wasSuccessful()
    app=gui.QApplication.instance() or gui.QApplication([]);window=gui.DINOReview();rows=[];protected={}
    try:
        assert not window.rescue_switch.isChecked() and not window.rescue_supplement_switch.isChecked()
        for case in load(ROOT/'artifacts/rescue_mainline_ab_20261002_v2/report.json')['cases']:
            if case['image'] not in ('normal_006.JPG','disconnected_005.JPG'):continue
            original=Path(case['output']);file=original/'report_off.json';protected[str(file)]=sha(file);report=load(file)
            target=OUT/Path(case['image']).stem;target.mkdir();shutil.copy2(original/'aligned.jpg',target/'aligned.jpg')
            window.current_output=target;window.reference.setText(report['reference']);window.inspection.setText(report['inspection'])
            window._write_report(report);window.port_scene_combo.setCurrentIndex(1)
            window.rescue_switch.setChecked(True);window.rescue_supplement_switch.setChecked(True);window._update_port_controls()
            assert window.rescue_button.isEnabled() and snapshot(window)['policy']==POLICY_ID
            before=sha(target/'report.json');finish_port_rescue(window,dict(token='old',snapshot=snapshot(window),result={},evidence='unused'))
            assert sha(target/'report.json')==before
            window.rescue_button.click();assert window.rescue_worker is not None
            assert not window.rescue_supplement_switch.isEnabled()
            deadline=time.monotonic()+240
            while window.rescue_worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.02)
            assert window.rescue_worker is None,'owned worker timeout';app.processEvents()
            saved=load(target/'report.json');current=saved['independent_port_rescue']
            assert current['recheck_policy']['policy_id']==POLICY_ID
            assert current['supplementary_policy']['policy_id']==POLICY_ID
            assert len(current['rescue_hints'])<=5 and len(current['supplementary_hints'])<=5
            assert saved['review_regions']==report['review_regions'] and not current['automatic_fault_verdict']
            previous=load(ROOT/'artifacts/consensus_port_live_20261002'/Path(case['image']).stem/'report.json')['independent_port_rescue']
            assert current['rescue_hints']==previous['rescue_hints'] and current['supplementary_hints']==previous['supplementary_hints']
            window._rescue_token='switchtest';binding=snapshot(window);payload=dict(token='switchtest',snapshot=binding)
            assert payload_is_current(window,payload);window.rescue_supplement_switch.setChecked(False);assert not payload_is_current(window,payload)
            rows.append(dict(image=case['image'],status=current['status'],primary=len(current['rescue_hints']),supplementary=len(current['supplementary_hints']),recheck=current['recheck_policy'],evidence=current['evidence_path']))
            print(json.dumps(rows[-1]),flush=True)
        assert {p:sha(Path(p)) for p in protected}==protected
        save(OUT/'report.json',dict(status='complete',tests=result.testsRun,cases=rows,default_off=True,
            real_button_worker_callback=True,stale_result_rejected=True,original_reports_unchanged=True,new_sam=False))
    finally:window.close();app.processEvents()
if __name__=='__main__':main()
