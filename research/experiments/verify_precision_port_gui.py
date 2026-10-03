"""Fresh prediction through the real existing button, on copies of real reports."""
import copy, json, os, shutil, sys, time, unittest
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REPO = Path('E:/PythonProject10')
OUT = ROOT/'artifacts/precision_port_gui_20261002_v2'
os.environ.update(QT_QPA_PLATFORM='offscreen',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False')
sys.path[:0] = [str(REPO),str(REPO/'prototype')]

def main():
    if OUT.exists():raise FileExistsError('Fresh output required')
    OUT.mkdir()
    import torch
    import assembly_auto_review_dino_v2 as entry
    gui = entry.implementation
    from inspection_agent.port_rescue_gui import snapshot,finish_port_rescue,payload_is_current
    from inspection_agent.precision_port_rescue import POLICY_ID
    from inspection_agent.optional_port_crop_review import sha
    suite=unittest.TestSuite()
    for name in ('test_precision_port_rescue.py','test_bounded_port_rescue.py','test_independent_port_rescue.py',
                 'test_port_crop_gui_bridge.py','test_port_state_hint.py','test_port_tiling.py',
                 'test_inspection_agent.py','test_inspection_agent_gui_contract.py',
                 'test_local_evidence_bridge.py','test_local_review_gui_bridge.py'):
        suite.addTests(unittest.defaultTestLoader.discover(str(REPO/'tests'),pattern=name))
    suite.addTests(unittest.TestLoader().discover(str(ROOT/'experiments'),pattern='test_port_rescue_freshness.py'))
    tests=unittest.TextTestRunner(verbosity=1).run(suite)
    assert tests.wasSuccessful()
    app=gui.QApplication.instance() or gui.QApplication([])
    window=gui.DINOReview()
    manifest=json.loads((ROOT/'artifacts/rescue_mainline_ab_20261002_v2/report.json').read_text(encoding='utf-8'))
    rows=[]; started=time.monotonic()
    try:
        assert not window.rescue_switch.isChecked() and not window.rescue_button.isEnabled()
        for case in manifest['cases']:
            original=Path(case['output']); file=original/'report_off.json'
            report=json.loads(file.read_text(encoding='utf-8'))
            digest=sha(file); target=OUT/Path(case['image']).stem;target.mkdir()
            shutil.copy2(original/'aligned.jpg',target/'aligned.jpg')
            window.current_output=target
            window.reference.setText(report['reference']);window.inspection.setText(report['inspection'])
            window._write_report(report)
            window.rescue_switch.setChecked(True);window.port_scene_combo.setCurrentIndex(1)
            window._update_port_controls()
            assert window.rescue_button.isEnabled()
            before=sha(target/'report.json')
            finish_port_rescue(window,dict(token='old',snapshot=snapshot(window),result={},evidence='unused'))
            assert sha(target/'report.json')==before
            window.rescue_button.click()
            assert window.rescue_worker is not None
            assert not window.run_button.isEnabled()
            deadline=time.monotonic()+180
            while window.rescue_worker is not None and time.monotonic()<deadline:
                app.processEvents();time.sleep(.02)
            assert window.rescue_worker is None,'worker timeout'
            app.processEvents()
            saved=json.loads((target/'report.json').read_text(encoding='utf-8'))
            result=saved['independent_port_rescue']
            assert result['budget_policy']['policy_id']==POLICY_ID
            assert result['budget_policy']['first_hint_retained'] is False
            assert all(h['box']['confidence']>.5 for h in result['rescue_hints'])
            assert len(result['rescue_hints'])<=5 and not result['automatic_fault_verdict']
            assert saved['review_regions']==report['review_regions'] and sha(file)==digest
            if result['status']=='applied':assert window.rescue_view.item is not None
            assert not window.agent_guidance_button.isEnabled()
            window._rescue_token='freshness';binding=snapshot(window)
            payload=dict(token='freshness',snapshot=binding)
            assert payload_is_current(window,payload)
            old_policy=copy.deepcopy(payload);old_policy['snapshot']['policy']='old_selector'
            assert not payload_is_current(window,old_policy)
            window.rescue_switch.setChecked(False)
            assert not payload_is_current(window,payload)
            rows.append(dict(image=case['image'],status=result['status'],reason=result['fallback_reason'],
                hints=len(result['rescue_hints']),budget_policy=result['budget_policy'],
                evidence=result['evidence_path'],original_report_unchanged=True))
            print(json.dumps(rows[-1]),flush=True)
        (OUT/'report.json').write_text(json.dumps(dict(status='complete',tests=tests.testsRun,
            policy=POLICY_ID,cases=rows,default_off=True,actual_button_thread_callback=True,
            fresh_source_and_reference_inference=True,new_dino_sam=False,
            seconds=round(time.monotonic()-started,2)),indent=2)+'\n',encoding='utf-8')
    finally:
        window.close();app.processEvents()

if __name__=='__main__':main()
