"""Fresh port inference via actual GUI on copies of real SAM outputs."""
import os,sys,json,shutil,copy,unittest,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path('E:/PythonProject10');WORK=Path(__file__).resolve().parents[1]
OUT=WORK/'artifacts/port_rescue_gui_20261002';OUT.mkdir(exist_ok=True)
os.environ.update(QT_QPA_PLATFORM='offscreen',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'prototype'))
import torch
import assembly_auto_review_dino_v2 as entry
gui=entry.implementation
from inspection_agent.port_rescue_gui import payload_is_current,finish_port_rescue,snapshot
from inspection_agent.optional_port_crop_review import sha
suite=unittest.TestSuite()
for name in ('test_independent_port_rescue.py','test_port_crop_gui_bridge.py','test_port_state_hint.py',
             'test_port_tiling.py','test_inspection_agent.py','test_inspection_agent_gui_contract.py',
             'test_local_evidence_bridge.py','test_local_review_gui_bridge.py'):
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=name))
tests=unittest.TextTestRunner(verbosity=1).run(suite)
assert tests.wasSuccessful()
app=gui.QApplication.instance() or gui.QApplication([])
window=gui.DINOReview();rows=[]
try:
    assert not window.rescue_switch.isChecked() and not window.rescue_button.isEnabled()
    live=WORK/'artifacts/main_scenario_live_20261002'
    manifest=json.loads((live/'report.json').read_text(encoding='utf-8'))
    for case in manifest['cases']:
        started=time.monotonic();original=Path(case['output'])
        report=json.loads((original/'report.json').read_text(encoding='utf-8'))
        target=OUT/Path(case['image']).stem;target.mkdir(exist_ok=True)
        task_before=sha(Path(case['task']));original_before=sha(original/'report.json')
        shutil.copy2(original/'aligned.jpg',target/'aligned.jpg')
        window.current_output=target
        window.reference.setText(report['reference']);window.inspection.setText(report['inspection'])
        window._write_report(report)
        window.rescue_switch.setChecked(True);window.port_scene_combo.setCurrentIndex(1)
        window._update_port_controls()
        assert not window.rescue_button.isEnabled(), 'Old report must not guess an ROI'
        # Explicit verifier-only ROI, identical to the real pilot analysis.
        report['analysis_check_rois']=[[.03,.04,.97,.96]]
        window._write_report(report);assert window.rescue_button.isEnabled()
        before=sha(target/'report.json')
        finish_port_rescue(window,dict(token='old',snapshot=snapshot(window),result={},evidence='unused'))
        assert sha(target/'report.json')==before
        window.rescue_button.click()
        assert window.rescue_worker is not None
        assert not window.port_hint_button.isEnabled() and not window.run_button.isEnabled()
        deadline=time.monotonic()+180
        while window.rescue_worker is not None and time.monotonic()<deadline:
            app.processEvents();time.sleep(.02)
        assert window.rescue_worker is None,'worker timeout'
        app.processEvents()
        saved=json.loads((target/'report.json').read_text(encoding='utf-8'))
        result=saved['independent_port_rescue']
        assert saved['review_regions']==report['review_regions']
        assert len(result['rescue_hints'])<=1 and not result['automatic_fault_verdict']
        assert sha(Path(case['task']))==task_before and sha(original/'report.json')==original_before
        assert not window.agent_guidance_button.isEnabled()
        window._rescue_token='check';binding=snapshot(window);payload=dict(token='check',snapshot=binding)
        assert payload_is_current(window,payload)
        changed=copy.deepcopy(saved);changed['test_change']=True;window._port_visual_report=changed
        assert not payload_is_current(window,payload)
        window._port_visual_report=saved
        with (target/'report.json').open('a',encoding='utf-8') as stream:stream.write('\n')
        assert not payload_is_current(window,payload)
        window._write_report(saved);window.rescue_switch.setChecked(False)
        assert not payload_is_current(window,payload)
        rows.append(dict(image=case['image'],status=result['status'],reason=result['fallback_reason'],
            hints=len(result['rescue_hints']),seconds=round(time.monotonic()-started,2),
            original_candidates_unchanged=True,original_task_unchanged=True))
        print(json.dumps(rows[-1]),flush=True)
    window.show();window.workbench_optional_toggle.click();app.processEvents()
    window.grab().save(str(OUT/'desktop.png'))
    (OUT/'verification.json').write_text(json.dumps(dict(tests=tests.testsRun,failures=0,errors=0,
        actual_button_thread_callback=True,fresh_source_and_reference=True,new_sam_dino=False,
        old_report_roi_rejected=True,stale_token_report_disk_switch_rejected=True,cases=rows),ensure_ascii=False,indent=2),encoding='utf-8')
finally:
    window.close();app.processEvents()
