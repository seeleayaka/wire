"""Actual optional Qt worker/callback on a protected completed-SAM report."""
import copy,json,os,shutil,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
OUT=ROOT/'artifacts/port_residual_feature_gui_20261003_v3'
sys.path[:0]=[str(REPO),str(REPO/'prototype')]
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',YOLO_CONFIG_DIR=str(OUT/'config'))
from inspection_agent.optional_port_crop_review import sha,SCENE
from inspection_agent.feature_residual_port_support import run_feature_residual_review,residual_runtime_fingerprint,POLICY_ID
from inspection_agent.context_port_recheck import render_consensus_overlay
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
    if OUT.exists():raise FileExistsError('Preserve previous checks')
    live=load(ROOT/'artifacts/port_residual_feature_live_20261003/report.json');assert live['qualifies_live_diagnostic']
    complete=ROOT/'artifacts/student_complete_sam_flow_20261003'
    assert load(complete/'acceptance.json')['sam_complete']
    protected={str(p):sha(p) for p in (complete/'report.json',complete/'before_optional_ports.json',complete/'aligned.jpg')}
    (OUT/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    import torch;torch.set_num_threads(4)
    from inspection_agent.port_rescue_gui import snapshot,payload_is_current,finish_port_rescue
    import assembly_auto_review_dino_v2 as entry
    gui=entry.implementation;app=gui.QApplication.instance() or gui.QApplication([]);window=gui.DINOReview();started=time.monotonic()
    try:
        assert not any(control.isChecked() for control in (window.rescue_switch,window.rescue_supplement_switch,window.rescue_student_switch,window.rescue_feature_switch))
        target=OUT/'complete_sam_008';target.mkdir();shutil.copy2(complete/'aligned.jpg',target/'aligned.jpg')
        report=load(complete/'before_optional_ports.json');assert report['decision']=='possible_difference_manual_review'
        window.current_output=target;window.reference.setText(report['reference']);window.inspection.setText(report['inspection']);window._write_report(report)
        window.port_scene_combo.setCurrentIndex(1)
        for control in (window.rescue_switch,window.rescue_supplement_switch,window.rescue_student_switch,window.rescue_feature_switch):control.setChecked(True)
        window._update_port_controls();assert window.rescue_button.isEnabled();assert snapshot(window)['policy']==POLICY_ID
        before=sha(target/'report.json');finish_port_rescue(window,dict(token='stale',snapshot=snapshot(window),result={},evidence='unused'))
        assert sha(target/'report.json')==before
        window.rescue_button.click();assert window.rescue_worker is not None and not window.rescue_feature_switch.isEnabled()
        deadline=time.monotonic()+300
        while window.rescue_worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.02)
        assert window.rescue_worker is None,'Owned Qt worker timeout'
        app.processEvents();saved=load(target/'report.json');current=saved['independent_port_rescue']
        assert current['feature_residual_policy']['enabled'] and current['feature_residual_policy']['fallback_reason'] is None
        assert current['feature_residual_policy']['added_hints']==0
        old=load(complete/'report.json')['independent_port_rescue']
        assert current['rescue_hints']==old['rescue_hints'] and current['supplementary_hints']==old['supplementary_hints']
        assert saved['review_regions']==report['review_regions'] and saved['decision']==report['decision']
        assert current['automatic_fault_verdict'] is False
        window._rescue_token='feature_switch_test';payload=dict(token='feature_switch_test',snapshot=snapshot(window))
        assert payload_is_current(window,payload);window.rescue_feature_switch.setChecked(False);assert not payload_is_current(window,payload)
        # New gain was checked through real registration/DINO but SAM is not
        # complete. The actual UI must not allow a port button on that report.
        positive=live['cases'][0];initial=load(Path(positive['initial_report']));original=copy.deepcopy(initial)
        positive_out=OUT/'formal_training_gain';positive_out.mkdir()
        result=run_feature_residual_review(initial,project=REPO,enabled=True,scene=SCENE,supplementary_enabled=True,student_enabled=True,feature_enabled=True)
        assert initial==original and result['feature_residual_policy']['fallback_reason'] is None
        assert result['feature_residual_policy']['added_hints']==1
        experimental=load(Path(positive['evidence']))
        assert result['rescue_hints']==experimental['rescue_hints'] and result['supplementary_hints']==experimental['supplementary_hints']
        save(positive_out/'evidence.json',result);render_consensus_overlay(Path(positive['initial_report']).parent/'aligned.jpg',result,positive_out/'overlay.jpg')
        window.current_output=positive_out;window._write_report(initial);window.inspection.setText(initial['inspection']);window.reference.setText(initial['reference'])
        window._update_port_controls();assert not window.rescue_button.isEnabled(),'Never relabel SAM-pending reports to make UI pass'
        assert {p:sha(Path(p)) for p in protected}==protected
        save(OUT/'report.json',dict(status='complete',real_qt_button_worker_callback=True,default_off=True,stale_rejected=True,
            completed_sam_report_reused=True,new_sam=False,complete_case_old_cues_preserved=True,
            complete_case_feature_added=0,formal_training_gain_added=1,formal_matches_experimental=True,sam_pending_ui_blocked=True,
            runtime_fingerprint=residual_runtime_fingerprint(REPO),protected_unchanged=True,field_accuracy=False,
            seconds=round(time.monotonic()-started,2)))
        print(json.dumps(dict(status='complete',seconds=round(time.monotonic()-started,2))),flush=True)
    finally:window.close();app.processEvents()
if __name__=='__main__':main()
