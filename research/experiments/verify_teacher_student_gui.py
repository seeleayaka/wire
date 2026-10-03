"""Real Qt button/worker/callback and formal backend gain parity; fresh output only."""
import copy,json,os,shutil,sys,time,unittest
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');OUT=ROOT/'artifacts/teacher_student_port_gui_20261003_v2'
sys.path[:0]=[str(REPO),str(REPO/'prototype')]
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',YOLO_CONFIG_DIR=str(OUT/'config'))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    if OUT.exists():raise FileExistsError('Fresh output required')
    (OUT/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    import torch;torch.set_num_threads(4)
    import assembly_auto_review_dino_v2 as entry
    gui=entry.implementation
    from inspection_agent.teacher_student_port_support import POLICY_ID,run_teacher_student_review
    from inspection_agent.context_port_recheck import render_consensus_overlay
    from inspection_agent.optional_port_crop_review import sha,SCENE
    from inspection_agent.port_rescue_gui import snapshot,payload_is_current,finish_port_rescue
    suite=unittest.TestSuite()
    for name in ('test_teacher_student_port_support.py','test_context_port_recheck.py','test_consensus_port_rescue.py','test_precision_port_rescue.py','test_bounded_port_rescue.py','test_independent_port_rescue.py','test_port_crop_gui_bridge.py','test_port_state_hint.py','test_port_tiling.py','test_inspection_agent.py','test_inspection_agent_gui_contract.py','test_local_evidence_bridge.py','test_local_review_gui_bridge.py'):
        suite.addTests(unittest.TestLoader().discover(str(REPO/'tests'),pattern=name))
    for name in ('test_port_rescue_freshness.py','test_teacher_student_gui_freshness.py'):
        suite.addTests(unittest.TestLoader().discover(str(ROOT/'experiments'),pattern=name))
    tests=unittest.TextTestRunner().run(suite);assert tests.wasSuccessful()
    app=gui.QApplication.instance() or gui.QApplication([]);window=gui.DINOReview();rows=[];protected={};started=time.monotonic()
    try:
        assert not any(s.isChecked() for s in (window.rescue_switch,window.rescue_supplement_switch,window.rescue_student_switch))
        assert not window.rescue_student_switch.isEnabled()
        for case in load(ROOT/'artifacts/rescue_mainline_ab_20261002_v2/report.json')['cases']:
            if case['image'] not in ('normal_006.JPG','disconnected_005.JPG'):continue
            original=Path(case['output']);file=original/'report_off.json';protected[str(file)]=sha(file);report=load(file)
            target=OUT/Path(case['image']).stem;target.mkdir();shutil.copy2(original/'aligned.jpg',target/'aligned.jpg')
            window.current_output=target;window.reference.setText(report['reference']);window.inspection.setText(report['inspection'])
            window._write_report(report);window.port_scene_combo.setCurrentIndex(1)
            window.rescue_switch.setChecked(True);window.rescue_supplement_switch.setChecked(True);window.rescue_student_switch.setChecked(True);window._update_port_controls()
            assert window.rescue_button.isEnabled() and snapshot(window)['policy']==POLICY_ID
            before=sha(target/'report.json');finish_port_rescue(window,dict(token='stale',snapshot=snapshot(window),result={},evidence='unused'))
            assert sha(target/'report.json')==before
            window.rescue_button.click();assert window.rescue_worker is not None
            assert not window.rescue_student_switch.isEnabled()
            deadline=time.monotonic()+240
            while window.rescue_worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.02)
            assert window.rescue_worker is None,'owned worker timeout';app.processEvents()
            saved=load(target/'report.json');current=saved['independent_port_rescue']
            assert current['teacher_student_policy']['policy_id']==POLICY_ID
            assert current['teacher_student_policy']['enabled']
            assert len(current['rescue_hints'])<=5 and len(current['supplementary_hints'])<=5
            assert saved['review_regions']==report['review_regions'] and saved['decision']==report['decision']
            assert not current['automatic_fault_verdict']
            previous=load(ROOT/'artifacts/consensus_port_live_20261002'/Path(case['image']).stem/'report.json')['independent_port_rescue']
            assert current['rescue_hints']==previous['rescue_hints'] and current['supplementary_hints']==previous['supplementary_hints']
            window._rescue_token='switchtest';binding=snapshot(window);payload=dict(token='switchtest',snapshot=binding)
            assert payload_is_current(window,payload)
            window.rescue_student_switch.setChecked(False);assert not payload_is_current(window,payload)
            rows.append(dict(image=case['image'],status=current['status'],primary=len(current['rescue_hints']),supplementary=len(current['supplementary_hints']),student=current['teacher_student_policy'],evidence=current['evidence_path']))
            print(json.dumps(rows[-1]),flush=True)
        assert {p:sha(Path(p)) for p in protected}==protected
        # Fresh formal-model execution on a prior training-selected diagnostic initial report.
        live=load(ROOT/'artifacts/teacher_student_port_live_20261003/report.json')
        gained=next(p for p in live['cases'] if p['split']=='train01')
        folder=Path(gained['evidence']).parent;report=load(folder/'initial_report.json');before=copy.deepcopy(report)
        result=run_teacher_student_review(report,project=REPO,enabled=True,scene=SCENE,supplementary_enabled=True,student_enabled=True)
        gain=OUT/'training_diagnostic';gain.mkdir();save(gain/'evidence.json',result)
        previous=load(Path(gained['evidence']))
        assert report==before and report['decision']=='sam3_fusion_running'
        assert result['teacher_student_policy']['fallback_reason'] is None
        assert result['rescue_hints']==previous['rescue_hints'] and result['supplementary_hints']==previous['supplementary_hints']
        assert result['teacher_student_policy']['added_hints']==1
        render_consensus_overlay(folder/'aligned.jpg',result,gain/'overlay.jpg')
        save(OUT/'report.json',dict(status='complete',tests=tests.testsRun,cases=rows,default_off=True,
            real_button_worker_callback=True,stale_result_rejected=True,original_reports_unchanged=True,new_sam=False,
            training_diagnostic=dict(image=gained['image'],fresh_model_inference=True,original_sam_pending=True,
                independent_accuracy=False,formal_matches_experimental=True,added_hints=1,evidence=str(gain/'evidence.json')),
            seconds=round(time.monotonic()-started,2)))
    finally:window.close();app.processEvents()
if __name__=='__main__':main()
